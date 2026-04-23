# Table Auto-Update Logic

This document details the automatic configuration update system implemented in `service.py`. This system ensures table bounding boxes remain accurate even if tables are slightly moved or bumped, without requiring manual intervention.

## Automatic Update Workflow

### 1. Timing (When it runs)
The update check is **periodic**, not continuous.
- **Trigger Condition:** `(current_time - last_update_time) >= CONFIG_UPDATE_INTERVAL`
- **Default Interval:** Every **30 minutes** (1800 seconds).
- **Environment Variable:** `CONFIG_UPDATE_INTERVAL`

### 2. Detection (What it sees)
When the timer triggers, the system inspects the current video frame.
- It filters for objects detected as `"dining table"` or `"table"`.
- It ignores any detection with a confidence score lower than `TABLE_CONF_THRESHOLD` (Default: 0.25).

### 3. Matching (Finding the tables)
The system attempts to match every currently configured table (e.g., Table 7, Table 12) to the new detections.
- **Method:** It calculates the **IoU (Intersection over Union)** between the *Configured Box* (from `table_configs.json`) and every *Detected Box* in the frame.
- **Selection:** It selects the detection with the **highest overlap (Best Match)** for each configured table.

### 4. Decision Rules (To Update or Not?)
Once the "Best Match" is found, the system decides whether to update the configuration based on the overlap percentage (IoU).

| Condition (IoU Overlap) | Interpretation | Action |
| :--- | :--- | :--- |
| **IoU < 10% (0.1)** | **Safety Check Failed.** The best match is too far away. The table is likely occluded, missing, or the detection is a hallucination. | **IGNORE** (No Change) |
| **10% <= IoU < 80% (0.8)** | **Drift Detected.** The table is roughly in the same spot (same object) but the box has shifted significantly. | **UPDATE CONFIGURATION** |
| **IoU >= 80% (0.8)** | **Accurate.** The current configuration matches the detection very closely. | **KEEP EXISTING** (No Change) |

*Note: The upper threshold is configurable via `CONFIG_UPDATE_IOU_THRESHOLD`.*

### 5. Execution (Saving Changes)
If an update is triggered:
1.  **Memory Update:** The internal coordinates for that table are updated immediately.
2.  **Disk Write:** The new coordinates are written to `artifacts/table_configs.json`.
3.  **Hot Reload:** The system (via the separate hot-reload logic) detects the file change and re-initializes the table processing logic to use the new accurate positions.
