# 🖥️ CPU Core Assignments Documentation

**Total System Cores:** 16 (Indexed 0-15)
**Allocation Strategy:** Service Isolation & Load Balancing
**Last Updated:** Dec 2025

| Core(s) | Service Name | Category | Load Characteristic | Status / Notes |
| :--- | :--- | :--- | :--- | :--- |
| **0, 1** | `stream_handling_service` | Video Pipeline | **Heavy / Saturated** | Primary bottleneck. Handles video decoding & frame distribution. Shared with Entry/Exit. |
| **0, 1** | `entry_exit_service` | Analytics | Low/Moderate | Variable load. Shares capacity with stream handling. |
| **2** | `yolo_service` | AI Inference | **Heavy** | dedicated core for object detection model (Wait/Post-process logic). |
| **3** | `grounding_dino_service` | AI Inference | **Heavy** | dedicated core for heavy specific object detection. |
| **4, 5** | `mongodb` | Database | **Heavy / Critical** | **Dual-Core**. Central data store. High I/O and compute from writers (Stream/AI) and readers (App). |
| **6** | `redis` | Database | Low | Message broker for all services. |
| **6** | `counter_staff_detection_service1` | Analytics | Idle/Low | Lightweight logic loop. |
| **6** | `cafe_counter_staff_detection_service` | Analytics | Idle/Low | Lightweight logic loop. |
| **6** | `table_occupancy_back_lobby` | Analytics | Idle/Low | Lightweight logic loop. |
| **6** | `table_occupancy_left_lobby` | Analytics | Idle/Low | Lightweight logic loop. |
| **6** | `table_detection_and_cleaning_mccafe` | Analytics | Idle/Low | Lightweight logic loop. |
| **6** | `table_detection_and_cleaning_kiosk` | Analytics | Idle/Low | Lightweight logic loop. |
| **6** | `table_detection_and_cleaning_birthday_party`| Analytics | Idle/Low | Lightweight logic loop. |
| **6** | `crew_room_service` | Analytics | Idle/Low | Lightweight logic loop. |
| **6** | `dustbin_status_tracking` | Analytics | Idle/Low | Lightweight logic loop. |
| **7** | `app-backend` | Application | **Moderate/Heavy** | dedicated core for API responsiveness. Avoids UI lag. |
| **8** | `kios_service` | AI Inference | Moderate | Dedicated core for Kiosk On/Off detection (ResNet50). | |
| **9** | *(Empty)* | Spare | - | Available. |
| **10** | *(Empty)* | Spare | - | Available. |
| **11** | *(Empty)* | Spare | - | Available. |
| **12** | *(Empty)* | Spare | - | Available. |
| **13** | `db_service` | Infrastructure | Low | Batch data writer to Mongo. |
| **13** | `app-frontend` | Infrastructure | Low | Static asset serving (Node.js). |
| **13** | `loki` | Infrastructure | Low | Log aggregation. |
| **13** | `grafana` | Infrastructure | Low | Visualization dashboard. |
| **13** | `minio` | Infrastructure | Low | Object storage (Images/Frames). |
| **14** | *(Reserved)* | System | - | Left free for OS kernel/background tasks. |
| **15** | *(Reserved)* | System | - | Left free for OS kernel/background tasks. |

## 📊 Summary Statistics

*   **Active Cores:** 9 (Cores 0-7, 13)
*   **Free Cores:** 7 (Cores 8-12, 14-15)

## 💡 Optimization Logic

1.  **Critical Path Isolation:** Video processing (`stream`, `yolo`, `dino`) and Database (`mongo`) get dedicated or dual cores to prevent bottlenecks that would starve downstream services.
2.  **Analytics Clustering:** Light analytics services (which mostly wait for Redis messages) are packed onto a single core (Core 6) to avoid wasting resources.
3.  **UI Responsiveness:** The Backend API gets a dedicated core (Core 7) to ensure the dashboard remains snappy even if the video pipeline is under heavy load.
4.  **Hardware Headroom:** Keeping Cores 8-12 empty allows for:
    *   Adding 5+ more camera streams.
    *   Scaling MongoDB to 4 cores if needed.
    *   Running ad-hoc analysis scripts without affecting production.

## 🛠️ Performance Monitoring

A custom script is included to monitor CPU usage over time (providing a more accurate picture than instantaneous snapshots).

### `monitor_docker_stats.py`

This script samples Docker stats every second for 30 seconds and calculates the **Average** and **Maximum** CPU usage for each service. This is crucial for verifying that the core assignments match the actual load.

**Usage:**
```bash
python3 monitor_docker_stats.py
```

**Output Example:**
```text
📊 RESULTS (Avg / Max over 30s):
SERVICE NAME                                  | AVG CPU %  | MAX CPU % 
---------------------------------------------------------------------------
mongodb                                       |   127.69% |   209.82%
stream_handling_service                       |   150.47% |   161.25%
app-backend                                   |    90.46% |   104.59%
...
```
