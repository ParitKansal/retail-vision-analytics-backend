# Drawer Detection Service

This service monitors the cash drawer status using geometric marker fitting and triggers security alerts when the drawer is left open and unattended.

## 1. Data Flow Architecture
The service operates as a Redis stream consumer:
1.  **Ingestion**: Reads frames from `INPUT_QUEUE_NAME` (e.g., `stream:cam:1`) via Redis Consumer Groups.
2.  **Throttling**: Implements frame rate limiting using `TARGET_FPS`. It skips frames if they arrive faster than the processing capacity or the target rate.
3.  **Processing**: Decodes the frame and passes it to the `DrawerDetector`.
4.  **Output**: Publishes raw status results to `drawer_status_output` and visualization frames to `stream:visualization:drawer:{CAMERA_ID}`.

---

## 2. Core Detection Logic (`detector.py`)
The detection logic uses **Geometric Marker Fitting** to identify if the drawer is closed.

### Key Concepts:
- **Rigid Pair**: The drawer is marked with a pair of strips—one **Bright** (Yellow/Green) and one **Dark** (Cyan/Pink). These are treated as a rigid unit with a fixed relative distance.
- **PCA Alignment**: During initialization, PCA is performed on the reference marker points to determine the primary "length axis" of the drawer.
- **Search & Fit**: At runtime, the algorithm scans a "Checking Region" (ROI). For each pixel:
    - It transforms the rigid pair template (Scale + Rotation).
    - It verifies if the pixels under the **Bright** mask are safe (> average threshold) and under the **Dark** mask are safe (< average threshold).
- **Result**: **PASS** (Draw Closed) if a valid fit is found; **FAIL** (Drawer Open) otherwise.

---

## 3. Results Smoothing: Mode Filter
To filter transient noise (motion blur, lighting spikes), the service uses a **Majority Vote** (Mode Filter):
- **Window**: A rolling buffer (`deque`) stores the last `HISTORY_BUFFER_SIZE` frames (default: 10).
- **Stable State**: The state reported (PASS/FAIL) is the most frequent value in that window. This prevents "flickering" alerts.

---

## 4. Person-in-Region Alert Mechanism
The service triggers a "Secure Drawer" alert based on a composite condition:

### The Composite Logic:
1.  **AI Detection**: Calls the `yolo_service` (port 8082) to find `person` objects.
2.  **ROI Overlap**: Checks if those persons are within the **Customer Region** polygons defined in `setting.json`.
3.  **Alert Condition**:
    - `Drawer_Status_Stable == FAIL` (Drawer is OPEN)
    - **AND** `Person_Stable == False` (NO Customer in the region)
4.  **State Management**:
    - **Raised**: Sent to `database:queue:0` when the condition becomes True.
    - **Resolved**: Sent when the drawer is closed OR a customer appears.

---

## 5. Visualization Overlay
If `VISUALIZE=True` is set, the service generates an annotated JPEG frame:
- **Cyan/Yellow Boxes**: Show where the geometric fitter found the markers.
- **Text Labels**: Show the smoothed (Stable) status of Front/Back and Person presence.
- **Red Border**: Flashes when an active **ALERT** is raised.
- **Queueing**: Frames are pushed to a Redis fixed-size queue for local viewing via `visualize.py`.

---

## 6. Configuration

### Environment Variables
- `TARGET_FPS`: Processing frequency (e.g., 0.1).
- `HISTORY_BUFFER_SIZE`: Smoothing window size.
- `YOLO_MODEL_URL`: Endpoint for person detection.
- `VISUALIZE`: Enable/disable visualization frame pushing.

### Key Files
- `setting.json`: Contains region polygons for marker checking and person monitoring.
- `detector.py`: Core geometric fitting implementation.
- `service.py`: Main loop, smoothing, and alerting logic.
- `visualize.py`: Local script to view annotated frames from Redis.

## Running Locally
```bash
# Start the service (ensure Redis and YOLO service are reachable)
python service.py

# Run visualization to see the detections in real-time
python visualize.py
```
