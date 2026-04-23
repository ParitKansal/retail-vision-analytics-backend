# RTSP Stream Timeout Analysis

## Summary
The `entry_exit_service` was experiencing stream disconnections due to OpenCV timing out while waiting for frames from the IP camera.

---

## Root Cause: Network Latency

**Diagnostic Results (2025-12-19):**
```
Ping to camera (192.168.10.2):
  min/avg/max = 149ms / 219ms / 412ms
```

This latency is **~20x higher** than expected for a local network camera (should be <10ms).

---

## Why Timeouts Occur

### 1. OpenCV Default Timeout
OpenCV uses FFmpeg internally for RTSP with default timeouts of ~5-10 seconds. When network latency spikes to 400ms+, combined with:
- Packet retransmissions
- Frame reassembly delays
- HEVC decoder waiting for reference frames

The cumulative delay can exceed OpenCV's timeout threshold.

### 2. UDP Packet Loss (Default RTSP Transport)
RTSP defaults to **UDP** for video transport:
- UDP doesn't guarantee delivery
- Lost packets = corrupted frames
- Decoder errors: `Could not find ref with POC X`

### 3. HEVC/H.265 Codec Dependencies
HEVC uses inter-frame dependencies:
- P-frames depend on previous I-frames
- B-frames depend on both past and future frames
- **One lost packet can corrupt multiple frames**

Observed errors:
```
[hevc @ ...] Could not find ref with POC 2
[hevc @ ...] Duplicate POC in a sequence: 1
```

### 4. Camera Resource Limits
IP cameras typically support:
- 2-4 simultaneous RTSP connections max
- Limited encoding bandwidth
- Internal buffer overflows under load

---

## Implemented Fix

Added **automatic reconnection logic** in `service.py`:
- Attempts reconnection up to 10 times
- 5-second delay between attempts
- Resets counter on successful reconnect

```python
# Reconnection loop
while reconnect_attempts < max_reconnect_attempts:
    reconnect_attempts += 1
    time.sleep(reconnect_delay)
    cap = cv2.VideoCapture(source)
    if cap.isOpened() and cap.read()[0]:
        break
```

---

## Recommended Improvements

### 1. Use TCP Transport (More Reliable)
```python
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
```

### 2. Increase Timeouts
```python
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;30000000|stimeout;30000000"
```

### 3. Network Infrastructure
- Check ethernet cable to camera
- Use dedicated switch/VLAN for cameras
- Avoid WiFi bridges for video streams

---

## References
- OpenCV FFmpeg Backend: https://docs.opencv.org/
- RTSP Protocol: RFC 2326
- HEVC/H.265 Frame Dependencies: ITU-T H.265
