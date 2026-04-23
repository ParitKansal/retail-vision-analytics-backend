# Retail Vision Analytics Backend

> Real-time computer vision microservices backend for Quick Service Restaurant operations. Processes multiple concurrent RTSP feeds to automate hygiene SLA enforcement, staff compliance monitoring, and customer intelligence — replacing manual observation entirely.

---

## Table of Contents

- [What This System Does](#what-this-system-does)
- [Tech Stack](#tech-stack)
- [System Architecture](#system-architecture)
- [Business Use Cases](#business-use-cases)
  - [1. Hygiene & SLA Enforcement](#1-hygiene--sla-enforcement)
  - [2. Staff Productivity & Policy Compliance](#2-staff-productivity--policy-compliance)
  - [3. Customer Journey & Demographics](#3-customer-journey--demographics)
  - [4. Real-Time Customer Satisfaction](#4-real-time-customer-satisfaction)
  - [5. Store Opening & Closing Compliance](#5-store-opening--closing-compliance)
  - [6. Automated Alert Engine](#6-automated-alert-engine)
- [Microservices Reference](#microservices-reference)
- [Hardware & Deployment](#hardware--deployment)
- [Getting Started](#getting-started)
- [Network Configuration](#network-configuration)
- [CPU Core Assignment](#cpu-core-assignment)
- [Monitoring & Observability](#monitoring--observability)
- [License](#license)

---

## What This System Does

Every RTSP camera feed in the store is ingested, decoded, and distributed to a fleet of specialized microservices. Each service solves a distinct operational problem — from detecting a dirty table to flagging a staff member using their phone at the counter — and raises structured, evidence-backed alerts with attached frame images when thresholds are breached. All data is persisted in MongoDB and visualized in real time on a Grafana dashboard accessible to store managers.

The system runs entirely on-premise on a single NVIDIA GPU server. No cloud inference. No manual review.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Video Ingestion | OpenCV, RTSP, FFmpeg |
| Object Detection | YOLOv8, Grounding DINO (HQQ 4-bit) |
| Behavioral Classification | XCLIP (zero-shot Vision-Language Model) |
| Multi-Object Tracking | ByteTrack, StrongSORT (OSNet ReID) |
| Emotion Detection | FER+ ONNX (4-bit BnB quantized) |
| Floor Analysis | Vision Transformer (ViT) + LoRA + HQQ 4-bit |
| Staff/Customer Classification | scikit-learn (custom trained) |
| Message Broker | Redis Streams with Consumer Groups |
| Object Storage | MinIO (S3-compatible) |
| Database | MongoDB Replica Set (local + cloud sync) |
| Logging | Grafana Loki |
| Dashboards | Grafana |
| Inference Serving | LitServe (HTTP), FastAPI |
| Orchestration | Docker Compose with NVIDIA Container Runtime |
| Frontend | Next.js + FastAPI |

---

## System Architecture

```
  IP Cameras (192.168.10.x)  ←── Dedicated wired LAN
           │
           ▼
  ┌──────────────────────┐
  │  stream_handling     │   Decode → MinIO (frames) → Redis Streams stream:cam:*
  │  service             │
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                      Redis Streams                           │
  │              Consumer Groups (cg:cam:*) per service          │
  └────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬───────┘
       │      │      │      │      │      │      │      │
       ▼      ▼      ▼      ▼      ▼      ▼      ▼      ▼
   [Counter [Table  [Video  [Floor [Entry [Kiosk [Crew  [Store
   Staff]  Occu.] Class.] Dirty]  Exit]  Person] Room] Open/Close]
   [Cafe   [Dustbin][Drawer][Path  [Emotion
   Counter]         ]      Track] Detect]
       │
       ▼
  ┌─────────────┐     ┌──────────────┐
  │  db_service │────►│   MongoDB    │◄──── db_sync_service ────► Cloud MongoDB
  └─────────────┘     └──────┬───────┘
                             │
                      ┌──────▼───────┐
                      │   Grafana    │  ← Loki (logs) + MongoDB (analytics)
                      └──────────────┘
```

**Redis Streams** provide exactly-once, fault-tolerant delivery. Each service joins a consumer group — if a container crashes and restarts, it resumes from its last acknowledged message.

**MinIO** stores raw frame binaries so Redis messages carry only lightweight pointer keys, not image blobs.

**MongoDB Replica Set** is the single source of truth. `db_service` dynamically routes each payload to the correct collection via a `target_collection` field embedded in every message.

**`db_sync_service`** mirrors the local MongoDB to a centralized cloud cluster for cross-store reporting.

---

## Business Use Cases

### 1. Hygiene & SLA Enforcement

| Feature | Service | How It Works |
|---|---|---|
| Table turnaround tracking | `table_occupancy_*` (×4 zones) | Starts a dirty SLA timer when a customer leaves a table. Logs clean time when staff resolves it. Alerts when SLA is breached. |
| Dirty floor / spill detection | `dirty_floor_service` | ViT model (HQQ 4-bit + LoRA) evaluates masked floor regions. Raises high-priority alert on confirmed hazard. |
| Active mopping validation | `video_classification_service` | XCLIP zero-shot classifies temporal clips against the prompt "mopping the floor" to confirm active cleaning vs. idle presence. |
| Dustbin fill monitoring | `dustbin_status_tracking` | Pixel-sum + contour analysis on polygon regions. Reports `operational`, `full`, or `undetectable` (blocked view). Majority-vote smoothing prevents flicker. |
| Aggregate dirty tables | `table_occupancy_cafe` | Separate alert fires when more than 4 tables in the McCafe zone are dirty simultaneously. |

**Table auto-recalibration**: Every 30 minutes all table services run a YOLO/Grounding DINO pass and compare detections against stored configs using IoU. Tables with 10–80% IoU drift are automatically updated on disk and hot-reloaded. Tables below 10% IoU are skipped as occluded. No manual reconfiguration needed when furniture is moved.

---

### 2. Staff Productivity & Policy Compliance

**Counter Staff Presence**

The `counter_staff_detection_service` (main counter) and `cafe_counter_staff_detection_service` (McCafe counter) each define two polygon zones per camera:

- **Staff Zone** — behind the counter. Any person here is assumed staff.
- **Customer Zone** — ordering area. Each person is sent to `staff_customer_classification_service` for AI-based staff/customer classification.

A 15-frame rolling majority vote prevents flickering. Three alert types:

| Alert | Condition |
|---|---|
| Unattended Customer | Customer in Customer Zone, no staff in Staff Zone |
| High Customer Traffic | More than 3 customers at the counter (majority vote) |
| Customer in Staff Zone | A customer is detected behind the counter |

Each alert selects the best representative frame from the history buffer rather than a potentially misleading latest frame.

**Crew Room Monitoring**

`crew_room_service` uses a custom YOLO model + StrongSORT ReID tracking (`osnet_x1_0_msmt17.pt`) to track each person in the break room with a persistent ID. An alert fires when any individual's dwell time exceeds `DURATION_THRESHOLD_MINUTES`.

**Behavioral Policy Violations**

`video_classification_service` builds temporal clips from Redis frame buffers and submits them to `xclip_service` for zero-shot classification across configurable mask regions:

| Region | Monitored Behavior |
|---|---|
| Back Lobby | Idle staff with no active task |
| Counter Floor | Active mopping in progress |
| Counter Workstation | Mobile phone usage |
| Crew Room | Sleeping, loitering, prohibited behavior |

**Cash Drawer Security**

`drawer_detection` tracks two rigid colored marker strips (bright + dark) using PCA-aligned Geometric Marker Fitting. If the drawer is OPEN and no customer is present in the region, a `SecureDrawer` alert fires. A 10-frame mode filter absorbs transient noise from lighting changes or motion blur.

---

### 3. Customer Journey & Demographics

**Entry / Exit Counting**

`entry_exit_service` connects directly to RTSP streams, sends frames to the remote YOLO service, and applies ByteTrack locally. Four configurable polygon gates define crossing boundaries. ENTRY and EXIT events are published to Redis with timestamps and persistent track IDs.

**Customer Path Tracking**

`customer_path_tracking_service` uses StrongSORT to follow individual trajectories across multiple camera views — entrance → kiosks → counter → seating. Spatial data feeds store layout analysis and congestion zone identification. Suppresses output when the store is closed.

**Kiosk Utilization**

`kiosk_person_service` detects person presence at self-service terminals and tracks per-person dwell duration. Publishes engagement events and queue buildup metrics to the analytics pipeline.

---

### 4. Real-Time Customer Satisfaction

`exit_emotion_detection_service` detects faces at exit-zone cameras and classifies emotions using the FER+ ONNX model (4-bit BnB quantized). Emotion scores per departing customer are aggregated into a passive CSAT metric — no feedback kiosk or customer participation required.

---

### 5. Store Opening & Closing Compliance

`store_opening_closing_service` monitors lighting transitions in a polygon region covering the main store area. It compares the detected transition time against the expected schedule:

| Category | Time Window |
|---|---|
| Early Opening | 7:00 – 7:50 |
| On-Time Opening | 7:50 – 8:15 |
| Late Opening | 8:15 – 9:00 |
| Early Closing | 12:00 – 12:50 |
| On-Time Closing | 12:50 – 1:05 |
| Late Closing | 1:05 – 2:00 |

Requires 80% of a 60-frame buffer to agree on a state before firing. Three transition frames are captured as evidence in the alert payload: last fully-dark, first partial-light, and first fully-lit.

---

### 6. Automated Alert Engine

Every service feeds structured payloads into the same pipeline. Each alert document contains:

```
ticket_id                  — UUID, persistent across "raised" and "resolved" events
type                       — "raised" or "resolved"
description                — human-readable text for the manager UI
frame                      — base64 JPEG with bounding box overlays
frame_without_visualization — raw JPEG for evidence archiving
camera_id, timestamp
```

Full alert catalogue across all services:

| Alert | Service | Trigger |
|---|---|---|
| Unattended Customer | `counter_staff_detection_service` | Customer at counter, no staff visible |
| High Customer Traffic | `counter_staff_detection_service` | >3 customers at counter (majority vote) |
| Customer in Staff Zone | `counter_staff_detection_service` | Customer detected behind the counter |
| McCafe Unattended | `cafe_counter_staff_detection_service` | Same logic, McCafe counter |
| Table Dirty SLA Breach | `table_occupancy_*` | Table dirty beyond SLA time limit |
| >4 Tables Dirty | `table_occupancy_cafe` | Aggregate: 4+ tables dirty simultaneously |
| Mopping Not Confirmed | `video_classification_service` | Dirty zone, XCLIP does not confirm mopping |
| Idle Staff | `video_classification_service` | Staff standing idle in lobby/counter zone |
| Mobile Phone Usage | `video_classification_service` | XCLIP detects phone use at workstation |
| Crew Room Loitering | `crew_room_service` | Person in break room beyond time limit |
| Floor Spill / Hazard | `dirty_floor_service` | ViT detects dirt or liquid on floor |
| Dustbin Full | `dustbin_status_tracking` | Pixel analysis confirms full dustbin |
| Drawer Open Unattended | `drawer_detection` | Drawer open, no customer present |
| Store Late / Early | `store_opening_closing_service` | Lights transition outside schedule window |
| Kiosk Queue Buildup | `kiosk_person_service` | Extended dwell time at kiosk terminal |

---

## Microservices Reference

### Infrastructure

| Service | Role |
|---|---|
| `mongodb` | Replica-set MongoDB. Central data sink for all analytics and alerts. |
| `redis` | Bitnami Redis. Streams-based message broker with per-service consumer groups. |
| `minio` | S3-compatible object store for raw frame binaries. TTL-managed by `delete_service`. |
| `loki` | Log aggregation. All services push structured JSON logs with `application` tags. |
| `grafana` | Dashboard and alerting UI. Sources: Loki (logs), MongoDB (analytics data). |

### Core Pipeline

| Service | Role |
|---|---|
| `stream_handling_service` | Decodes RTSP feeds, writes frames to MinIO, publishes pointer messages to `stream:cam:*`. |
| `delete_service` | Garbage collection worker. Purges expired frame objects from MinIO. |
| `db_service` | Subscribes to all Redis output queues. Routes each document to the correct MongoDB collection. |
| `db_sync_service` | Replicates local MongoDB to cloud cluster for cross-store reporting. |

### AI Inference

| Service | Model | Role |
|---|---|---|
| `yolo_service` | YOLOv8 (CUDA) | Person and object detection. HTTP endpoint consumed by most analytics services. |
| `grounding_dino_service` | Grounding DINO (HQQ) | Zero-shot free-text object detection via LitServe. |
| `xclip_service` | XCLIP (VLM) | Zero-shot video action classification from temporal frame buffers. |
| `staff_customer_classification_service` | scikit-learn | Binary staff/customer classifier from cropped person images. |

### Analytics Services

| Service | Camera Zone | Problem Solved |
|---|---|---|
| `counter_staff_detection_service` | Main counter | Staff presence, customer queue, zone violations |
| `cafe_counter_staff_detection_service` | McCafe counter | Same logic, separate zone |
| `crew_room_service` | Break room | Loitering detection with ReID tracking |
| `video_classification_service` | Multiple (masked regions) | Phone use, idle staff, mopping validation |
| `table_occupancy_back_lobby` | Back lobby | Table dirty/clean SLA tracking |
| `table_occupancy_left_lobby` | Left lobby | Table dirty/clean SLA tracking |
| `table_occupancy_cafe` | McCafe dining | Table SLA + aggregate dirty alert |
| `table_occupancy_kiosk` | Kiosk seating | Table dirty/clean SLA tracking |
| `dirty_floor_service` | Floor zones | ViT-based spill and dirt detection |
| `dustbin_status_tracking` | Dustbin area | Fill level monitoring |
| `drawer_detection` | Cash counter | Cash drawer open/unattended security |
| `entry_exit_service` | Entrance doors | Bidirectional footfall counting |
| `customer_path_tracking_service` | Multiple | Spatial trajectory mapping |
| `kiosk_person_service` | Self-service kiosks | Engagement time and queue buildup |
| `exit_emotion_detection_service` | Exit zones | Passive CSAT via facial emotion |
| `store_opening_closing_service` | Main area | Opening/closing schedule compliance |

### System Reliability

| Service | Role |
|---|---|
| `ip_monitor_service` | Polls public IP every 5 min. Sends email alert on change. 1-hour notification cooldown. |
| `uptime_scheduler` | Starts/stops the stack on a schedule to reduce off-hours GPU wear and storage growth. |
| `frontend_service` | Next.js manager dashboard + FastAPI backend. Separate entrypoints, single container. |

---

## Hardware & Deployment

| Requirement | Detail |
|---|---|
| GPU | NVIDIA CUDA-capable, minimum 8 GB VRAM |
| CPU | 16-core recommended (see core assignment below) |
| Container Runtime | Docker Compose v2 + `nvidia-container-runtime` |
| Network | Dual interface: WiFi (internet/cloud sync), Ethernet (cameras only) |
| Camera LAN | Static IP `192.168.10.200/24` on wired interface |

Verified camera IPs: `192.168.10.12`, `192.168.10.13`, `192.168.10.14`, `192.168.10.15`, `192.168.10.21`

---

## Getting Started

```bash
# 1. Clone and enter the backend directory
git clone <repo-url>
cd backend

# 2. Set your camera RTSP URLs in docker-compose.yml (INPUT_RTSP per service)

# 3. One-time camera network setup
./scripts/fix_camera_network.sh

# 4. Build and start the full stack
docker-compose up -d --build

# 5. Verify all containers are running
docker-compose ps
```

- Grafana dashboard: `http://localhost:3000`
- Store manager UI: `http://localhost:3001`

---

## Network Configuration

The host needs split routing: internet traffic via WiFi, camera streams via wired LAN.

```bash
# Lock wired interface to never be the default internet gateway
sudo nmcli con mod "enx00e04c411b80" ipv4.method manual
sudo nmcli con mod "enx00e04c411b80" ipv4.addresses 192.168.10.200/24
sudo nmcli con mod "enx00e04c411b80" ipv4.never-default yes
sudo nmcli con mod "enx00e04c411b80" ipv6.method ignore

# Ensure WiFi is the default gateway
sudo nmcli con mod "ACTFIBERNET_5G" ipv4.never-default no

# Apply
sudo nmcli con down enx00e04c411b80 && sudo nmcli con up enx00e04c411b80
```

Verify:
```bash
ip route              # "default" must show wlp2s0 (WiFi), not enx...
ip route get 8.8.8.8  # must show wlp2s0
ping 192.168.10.13    # must receive replies from camera
```

An hourly cron job (`0 * * * *`) runs `scripts/monitor_camera_network.sh` to detect and auto-fix any camera interface drops. Maximum recovery time: 59 minutes. Logs at `/var/log/camera_network_monitor.log`.

Full documentation: [backend/COMPLETE_NETWORK_DOCUMENTATION.md](backend/COMPLETE_NETWORK_DOCUMENTATION.md)

---

## CPU Core Assignment

Explicit `cpuset` pinning prevents inference services from starving analytics services under load.

| Cores | Assigned Services | Load Profile |
|---|---|---|
| 0–1 | `stream_handling_service`, `entry_exit_service` | Heavy — video decode + frame distribution |
| 2 | `yolo_service` | Heavy — GPU post-processing, HTTP handling |
| 3 | `grounding_dino_service` | Heavy — large model inference |
| 4–5 | `mongodb` | Heavy / Critical — high I/O, all writers + readers |
| 6 | `redis`, all lightweight analytics services | Low — mostly waiting on Redis messages |
| 7 | `app-backend` (FastAPI) | Moderate — dedicated for UI responsiveness |
| 8 | `kiosk_person_service` | Moderate — ResNet50 inference |
| 9–12 | *(Reserved)* | Headroom for additional cameras or MongoDB scaling |
| 13 | `db_service`, `app-frontend`, `loki`, `grafana`, `minio` | Low — infrastructure services |
| 14–15 | *(Reserved)* | OS kernel tasks |

Monitor actual CPU usage over 30 seconds:
```bash
python3 backend/monitor_docker_stats.py
```

---

## Monitoring & Observability

| Tool | Access | Purpose |
|---|---|---|
| Grafana | `localhost:3000` | Operational dashboards, alert history, service health |
| Loki | via Grafana Explore | Structured log aggregation from all containers |
| Camera Network Monitor | `/var/log/camera_network_monitor.log` | Hourly health check + auto-reconnect for camera LAN |
| Docker Stats Profiler | `python3 monitor_docker_stats.py` | 30-second CPU avg/max per container |
| IP Monitor | Email alert | Notifies on public IP change for remote access continuity |

---

## License

All Rights Reserved. Proprietary portfolio project. See `LICENSE` for usage terms.

---

*Internal project scope documentation link reserved for internal tracing.*
