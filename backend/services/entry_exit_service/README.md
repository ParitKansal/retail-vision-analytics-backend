# Entry/Exit Counting Service Documentation

A Redis-based video analytics service that processes RTSP camera streams to track and count customer entry/exit events using Remote YOLO detection and Local ByteTrack tracking.

---

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  RTSP Camera    │────►│   FrameReader   │────►│   Frame Queue   │
│  (IP Camera)    │     │   (Thread)      │     │   (max 30)      │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
                        ┌─────────────────────────────────────────────┐
                        │          Main Processing Loop               │
                        │  1. Get frame from queue                   │
                        │  2. Send to YOLO Service (HTTP)            │
                        │  3. Get detections (bounding boxes)        │
                        │  4. Update ByteTrack (local tracking)      │
                        │  5. Check region crossings                 │
                        │  6. Publish ENTRY/EXIT events to Redis     │
                        └─────────────────────────────────────────────┘
                                          │
                        ┌─────────────────┼─────────────────┐
                        ▼                 ▼                 ▼
                 ┌──────────┐      ┌──────────┐      ┌──────────┐
                 │   YOLO   │      │  Redis   │      │   Loki   │
                 │ Service  │      │ (Events) │      │ (Logs)   │
                 │  (GPU)   │      │          │      │          │
                 └──────────┘      └──────────┘      └──────────┘
```

---

## Process Flow

### 1. Frame Reading (FrameReader Thread)

| Action | Description |
|--------|-------------|
| Connect to RTSP | Opens stream via `cv2.VideoCapture(rtsp_url)` |
| Read frames | Continuously reads frames from camera |
| Queue frames | Puts frames into `Queue(maxsize=30)` |
| Handle disconnect | Auto-reconnects every 5 seconds if stream fails |
| Drop oldest | If queue full, drops oldest frame to make room |

### 2. Detection via Remote YOLO

| Action | Description |
|--------|-------------|
| Encode frame | Compress to JPEG |
| HTTP POST | Send to `http://yolo_service:8082/predict` |
| Parse response | Extract bounding boxes `[x1, y1, x2, y2, conf, class]` |
| Filter persons | Only keep detections with `class == 0` (person) |

### 3. Local Tracking (ByteTrack)

| Action | Description |
|--------|-------------|
| Track objects | Assigns consistent IDs to each person across frames |
| Handle occlusion | Maintains track even if person briefly hidden |
| Track buffer | Keeps track alive for 30 frames after last seen |

### 4. Region Crossing Detection

The service uses 4 polygon regions defined in `config.json`:

```
    ┌───────────────────────────────────────┐
    │   ┌────────┐  ┌────────┐             │
    │   │ R1     │  │  R2    │             │
    │   │(Entry) │  │        │             │
    │   └────────┘  └────────┘             │
    │   ┌────────┐  ┌────────┐             │
    │   │ R3     │  │  R4    │             │
    │   │        │  │(Inside)│             │
    │   └────────┘  └────────┘             │
    └───────────────────────────────────────┘
```

### 5. Entry/Exit Logic

| Path Example | Event |
|--------------|-------|
| Person walks R1→R2→R3→R4 | **ENTRY** detected |
| Person walks R4→R3→R2→R1 | **EXIT** detected |

### 6. Event Publishing

Events are published to Redis stream:

```json
{
  "camera_id": "entrance",
  "timestamp": 1703087342.123,
  "event": {
    "event_type": "ENTRY",
    "track_id": 42,
    "path": [1, 2, 3]
  },
  "target_collection": "entry_exit_collection"
}
```

---

## Monitoring Commands

### Check Queue Size (frames buffered from RTSP)

```bash
docker exec entry_exit_service bash -c "kill -10 \$(pgrep -f python)" 2>/dev/null
```

Then view the output:
```bash
docker logs entry_exit_service --tail 5 | grep MONITOR
```

**Output:**
```
[MONITOR] Current Queue Size: 8 / 30
```

### View Service Logs

```bash
docker logs -f entry_exit_service
```

### Check Container Status

```bash
docker ps --filter "name=entry_exit" --format "table {{.Names}}\t{{.Status}}"
```

---

## Log Events Reference

| Event | Meaning |
|-------|---------|
| `STREAM_CONNECT_ATTEMPT` | Trying to connect to RTSP |
| `STREAM_CONNECTED` | Successfully connected, shows resolution/fps |
| `STREAM_DISCONNECT` | Stream lost, shows frames read |
| `STREAM_RECONNECT_ATTEMPT` | Attempting to reconnect |
| `STREAM_RECONNECTED` | Reconnection successful, shows downtime |
| `STREAM_UNSTABLE` | 30 consecutive failures, marking disconnected |
| `FRAME_READ_FAILURE` | Corrupt frame received |
| `QUEUE_FULL` | Frame buffer full, dropping frames |
| `[MONITOR] Queue Size` | Manual queue check via signal |
| `✓ Pushed ENTRY event` | Person entered |
| `✓ Pushed EXIT event` | Person exited |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `INPUT_RTSP` | - | RTSP stream URL |
| `YOLO_SERVICE_URL` | `http://yolo_service:8082/predict` | YOLO API endpoint |
| `REDIS_HOST` | `localhost` | Redis server host |
| `REDIS_PORT` | `6379` | Redis server port |
| `FRAME_QUEUE_SIZE` | `30` | Max frames in buffer |
| `OUTPUT_QUEUE_NAME_1` | `entry_exit_output_queue` | Redis output stream |
| `TARGET_COLLECTION_NAME` | `entry_exit_collection` | MongoDB collection |
| `LOKI_URL` | `http://localhost:3100/...` | Loki logging endpoint |

---

## Docker Commands

### Start Service
```bash
docker compose up -d entry_exit_service
```

### Rebuild and Start
```bash
docker compose build entry_exit_service && docker compose up -d entry_exit_service
```

### Stop Service
```bash
docker compose stop entry_exit_service
```

### View Logs
```bash
docker logs -f entry_exit_service
```

### Check Queue Size
```bash
docker exec entry_exit_service bash -c "kill -10 \$(pgrep -f python)" 2>/dev/null
docker logs entry_exit_service --tail 5 | grep MONITOR
```

---

## Troubleshooting

### Stream Keeps Disconnecting
- Check network connectivity to camera
- Verify RTSP URL is correct
- Camera may have connection limits

### Queue Always Empty (0/30)
- Stream is being processed faster than it arrives
- Normal for stable processing

### Queue Always Full (30/30)
- YOLO service is slow or overloaded
- Network latency to YOLO service
- Consider increasing `FRAME_QUEUE_SIZE`

### "Connection refused" to YOLO Service
- Check if `yolo_service` container is running
- Verify network connectivity between containers

### HEVC Codec Warnings
```
[hevc @ ...] Could not find ref with POC X
```
- Normal for RTSP streams with packet loss
- Service recovers automatically
