
# System Architecture

## Redis Queues and Data Flow

This document outlines the mapping between RTSP Camera feeds, Redis Queues, and the Services that consume them.

### Input Streams (Camera Feeds)

The `stream_handling_service` ingests RTSP streams and pushes frames to Redis Streams (`stream:cam:X`). These streams are then consumed by various detection services.

| Redis Queue | RTSP Source URL | Description | Consuming Service |
| :--- | :--- | :--- | :--- |
| **`stream:cam:1`** | `rtsp://192.168.2.127:8554/counter1` | Counter 1 Feed | `counter_staff_detection_service1` |
| **`stream:cam:2`** | `rtsp://192.168.2.127:8554/counter2` | Counter 2 Feed | `counter_staff_detection_service2` |
| **`stream:cam:3`** | `rtsp://192.168.2.127:8554/lobby_video1` | Lobby Area 1 | `table_detection_and_cleaning1` |
| **`stream:cam:4`** | `rtsp://192.168.2.127:8554/lobby_video2` | Lobby Area 2 | `table_detection_and_cleaning2` |
| **`stream:cam:5`** | `rtsp://192.168.2.127:8554/crew_room` | Crew Room Feed | `crew_room_service` |
| **`stream:cam:6`** | `rtsp://192.168.2.127:8554/kiosk` | Dustbin Status Tracking | `dustbin_status_tracking` |

### Direct RTSP Services

Some services connect directly to the RTSP stream and do not consume from `stream:cam:*` queues. They push their results to the output event queue.

| Service Name | RTSP Source URL | Output Queue |
| :--- | :--- | :--- |
| **`entry_exit_service`** | `rtsp://192.168.2.127:8554/entrance` | `database:queue:0` |
| **`exit_emotion_detection_service`** | `rtsp://192.168.2.127:8554/kiosk` | `database:queue:0` |

### Output/Event Queues

These queues are used to store the analysis results (events) which are then processed by the `db_service` for storage in MongoDB.

| Queue Name | Producers | Consumers | Content Type |
| :--- | :--- | :--- | :--- |
| **`database:queue:0`** | `entry_exit_service`, `exit_emotion_detection_service`, and others | `db_service` | JSON Events (Entry/Exit, Emotions, Detections) |

## Service Configuration Sources

- **Stream Mappings**: Configured in `services/stream_handling_service/settings.json`.
- **Exit Emotion Service**: Configured in `services/exit_service/.env`.
- **Entry/Exit Service**: Configured via `environment` variables in `docker-compose.yml`.

---

# Operational Guide

This section describes how to manage the application lifecycle, including updates, restarts, and database recovery.

## 1. Daily Operation
**Command:** `docker compose up -d`

-   Automatically checks for and downloads the latest code from GitHub (Runtime Updates).
-   Starts all services.
-   Persists user data.

## 2. Applying Configuration Changes
**Command:** `docker compose up -d`

-   If you modify `docker-compose.yml` (e.g., adding ports or changing environment variables), run this command again.
-   Docker will recreate only the affected containers.

## 2.5. Applying Python File Code Changes
**Command:** `docker compose build <service_name> && docker compose up -d <service_name>`

-   If you modify any `.py` source code files for a specific service (like `service.py`), you must explicitly rebuild that service's Docker image before restarting it.
-   Example for rebuilding the dirty floor service:
    ```bash
    docker compose build dirty_floor_service && docker compose up -d dirty_floor_service
    ```

## 3. Database Recovery (Full Reset)
**Scenerio:** You ran `docker compose down -v` and lost all data.

1.  Start the application: `docker compose up -d`
2.  Wait 30 seconds for the backend to initialize
3.  **Seed the demo user**:
    ```bash
    docker exec app-backend python /workspace/seed_user.py
    ```
4.  Log in with the credentials set via `SEED_USER_EMAIL` and `SEED_USER_PASSWORD` environment variables.

---

# System Monitoring

This section provides instructions on how to check the status of various system components, including Redis streams, CPU memory, and GPU memory.

## Redis Stream Lengths

To check the amount of data (queue length) in all available input streams dynamically, you can use the following bash script snippet. This will find all keys matching the pattern `stream:cam:*` and print their lengths.

```bash
# Get all stream keys and check their length
docker exec redis redis-cli KEYS "stream:cam:*" | while read key; do
  # Remove any potential carriage returns from the key name
  key=$(echo "$key" | tr -d '\r')
  if [ ! -z "$key" ]; then
    len=$(docker exec redis redis-cli XLEN "$key")
    echo "$key: $len"
  fi
done
```

### One-liner version

You can also run this as a single command line:

```bash
docker exec redis redis-cli KEYS "stream:cam:*" | tr -d '\r' | while read key; do [ ! -z "$key" ] && echo "$key: $(docker exec redis redis-cli XLEN "$key")"; done
```

## Consumer Lag Monitoring

**Important:** Stream length shows total messages processed, but **lag** shows how many frames are waiting to be processed (performance metric).

### Quick Lag Check

Use the included `check_lag.sh` script to monitor all services:

```bash
./scripts/check_lag.sh
```

This will show:
- ✅ No lag (0 frames waiting) - Service is real-time
- ⚠️ Minor lag (<10 frames) - Service is slightly behind
- ❌ Significant lag (≥10 frames) - Service needs optimization

### Understanding Lag

- **Lag = 0**: Service is processing frames in real-time ✅
- **Lag > 0**: Service is falling behind, frames are queuing up ❌
- **Example**: Lag of 60 at 1 FPS = 60 seconds of delay

### Manual Lag Check

To check lag for a specific camera:

```bash
docker exec redis redis-cli XINFO GROUPS stream:cam:1
```

Look for the `lag` field for each consumer group.

## System Resources

### CPU Memory (RAM)

To check the system memory usage, use the `free` command.

```bash
free -h
```

Output explanation:
- **total**: Total installed RAM.
- **used**: RAM currently in use.
- **free**: Unused RAM.
- **available**: RAM available for new applications (includes buffers/cache).

### GPU Memory (NVIDIA)

To check the NVIDIA GPU status, memory usage, and running processes, use:

```bash
nvidia-smi
```

Key fields to look at:
- **Memory-Usage**: Shows used vs total VRAM (e.g., `2161MiB / 4096MiB`).
- **GPU-Util**: Percentage of GPU processing power currently being used.
- **Processes**: List of processes currently using the GPU.

## Database Access

### MongoDB Compass Connection
To connect to the local MongoDB instance using MongoDB Compass, use the following connection string:

```
mongodb://admin:<MONGO_PASSWORD>@localhost:27018/?authSource=admin
```
