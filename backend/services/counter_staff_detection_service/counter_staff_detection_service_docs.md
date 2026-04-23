# Counter Staff Detection Service Documentation

## 1. Overview
The **`counter_staff_detection_service`** monitors a service counter area to ensure customers are not left unattended. It uses a combination of object detection, zone-based filtering, and visual classification to accurately detect the presence of staff and customers.

## 2. Core Logic Workflow

### Step 1: Frame Acquisition & Detection
- **Input**: Reads video frames from Redis Streams (e.g., `stream:cam:1`).
- **Detection**: Sends frames to the **YOLO Service** to detect "persons".

### Step 2: Zone Filtering
The service uses configured polygons (in [setting.json](file:///home/xelomoc/Desktop/McD_backend-1/backend/services/counter_staff_detection_service/setting.json)) to define two critical zones:
1.  **Staff Zone**: The area behind the counter.
2.  **Customer Zone**: The ordering area in front of the counter.

### Step 3: Identity Classification (The "Smart" Filter)
To prevent false alarms (e.g., staff cleaning the lobby being counted as customers), the service applies different logic per zone:

*   **Staff Zone**:
    *   Any person detected here is **automatically assumed to be Staff**.
    *   *Reasoning*: Only staff should be behind the counter.

*   **Customer Zone**:
    *   Any person detected here is a **Customer Candidate**.
    *   **Visual Verification**: The system crops the person's image and sends it to the **`staff_customer_classification_service`**.
    *   **Classification**:
        *   If the AI predicts **"Customer"**: Validated as a Customer. [(Counted)](file:///home/xelomoc/Desktop/McD_backend-1/backend/services/counter_staff_detection_service/service.py#75-84)
        *   If the AI predicts **"Staff"**: Identified as Staff. [(Ignored)](file:///home/xelomoc/Desktop/McD_backend-1/backend/services/counter_staff_detection_service/service.py#75-84)

### Step 4: Statistical Smoothing
To avoid flickering alerts, the service maintains a history of the last **15 frames** and uses a majority vote (presence in > 50% of frames) to determine the true state.

### Step 5: Alerts & Logic
The service now supports **three distinct alert types**:

#### 1. Unattended Customer
*   **Trigger**:
    *   **Staff Absent**: No staff detected in Staff Zone (in any of last 15 frames).
    *   **Customer Present**: Customer detected in Customer Zone (> 50% of last 15 frames).
*   **Resolution**: Staff returns OR Customer leaves.

#### 2. High Customer Traffic
*   **Trigger**:
    *   **Customer Count > 3**: More than 3 customers detected in the Customer Zone (> 50% of last 15 frames).
*   **Resolution**: Customer count drops to 3 or less.

#### 3. Customer in Staff Zone (Violation)
*   **Trigger**:
    *   **Intruder Detected**: A person in the Staff Zone is classified as a "Customer".
    *   **Persistence**: This occurs in > 50% of last 15 frames.
*   **Resolution**: The intruder leaves the Staff Zone (or is re-classified as Staff).

### Step 6: Optimized Alert Image
When an alert is raised, specific logic ensures the attached image is representative:
*   Instead of simply using the *latest* frame (which might be a glitch), the system searches the history.
*   It selects a frame that visually matches the alert condition (e.g., shows > 3 customers for a traffic alert).


## 3. Configuration ([docker-compose.yml](file:///home/xelomoc/Desktop/McD_backend-1/backend/docker-compose.yml))

| Variable | Description |
| :--- | :--- |
| `INPUT_QUEUE_NAME` | Redis Stream key (e.g., `stream:cam:1`) |
| `YOLO_MODEL_URL` | URL of the object detection service |
| `STAFF_CUSTOMER_CLASSIFICATION_URL` | URL of the classification service |
| `VISUALIZE` | Set to `True` to enable the visualization queue |

## 4. Visualization & Debugging
A visualization script is provided to verify the logic in real-time.

**Command:**
```bash
python3 backend/services/counter_staff_detection_service/vizualize.py
```

**What you see:**
*   **Green Boxes**: Detected Staff.
*   **Orange Boxes**: Verified Customers.
*   **Statistics**: Mean, Median, and Mode of customer counts displayed on-screen.

## 5. Redis Payload Schema

This section details the structure of the JSON payloads pushed to the `OUTPUT_QUEUE_1` (Redis Stream).

### 1. Standard Detection Output
Published for every processed frame.

**Redis Stream Key**: `OUTPUT_QUEUE_1` (env var)
**Field**: `data` (JSON string)

```json
{
  "camera_id": "string",             // e.g., "1", "2"
  "timestamp": "string",             // ISO 8601 timestamp from source
  
  // STAFF ZONE DETECTIONS
  "valid_detection": [               // List of STAFF detected INSIDE STAFF ZONE
    {
      "class": 0,
      "bbox": [x1, y1, x2, y2],      // Normalized coordinates [0-1]
      "confidence": 0.95,
      "classification": "staff"
    }
  ],
  "customer_in_staff_detections": [   // List of CUSTOMERS detected INSIDE STAFF ZONE (Violation)
     {
      "class": 0,
      "bbox": [x1, y1, x2, y2],      // Normalized coordinates [0-1]
      "confidence": 0.88,
      "classification": "customer"
    }
  ],

  // CUSTOMER ZONE DETECTIONS
  "customer_detection": [            // List of CUSTOMERS detected INSIDE CUSTOMER ZONE
    {
      "class": 0,
      "bbox": [x1, y1, x2, y2],      // Normalized coordinates [0-1]
      "confidence": 0.88,
      "classification": "customer"
    }
  ],
  "staff_in_customer_detections": [  // List of STAFF detected INSIDE CUSTOMER ZONE
      {
          "class": 0,
          "bbox": [x1, y1, x2, y2],
          "confidence": 0.92,
          "classification": "staff"
      }
  ],

  "is_staff_detected": boolean,      // True if staff detected in Staff Zone in current frame history
  "is_customer_detected": boolean,   // True if customer detected (majority vote)
  "customer_count": integer,         // Number of customers in current frame
  "customer_count_mean": float,      // Mean customer count over last 15 frames
  "customer_count_median": float,    // Median customer count over last 15 frames
  "customer_count_mode": float,      // Mode customer count over last 15 frames (CUSTOMER ZONE ONLY)
  "target_collection": "string",     // Target MongoDB collection (all services use this)
  "camera_id": "string",             // Unique Camera Identifier
  "timestamp": "string"              // Timestamp of the frame
}
```

### 2. Alert Payloads
Alerts are special payloads pushed to the same stream when specific conditions are met. They usually target the `alert_collection`.

#### Common Fields for All Alerts
```json
{
  "camera_id": "string",
  "timestamp": "string",
  "frame": "string",                 // Base64 encoded JPEG image (with definition/box)
  "frame_without_visualization": "string", // Base64 encoded JPEG raw image (only in Raise)
  "description": "string",           // Human-readable alert description
  "target_collection": "alert_collection",
  "type": "string",                  // "raised" or "resolved"
  "ticket_id": "string"              // UUIDv4, persists between Raise and Resolve
}
```

#### A. Unattended Customer Alert
*   **Condition**: Customer present at counter, but NO staff detected.
*   **Description (Raised)**: `"Customer waiting unattended at counter."`
*   **Description (Resolved)**: `"Unattended customer situation resolved (Staff returned or customer left)."`

#### B. High Customer Traffic Alert
*   **Condition**: More than 3 customers detected at the counter (majority vote of history).
*   **Description (Raised)**: `"High customer traffic detected (More than 3 customers)."`
*   **Description (Resolved)**: `"High customer traffic resolved."`

#### C. Customer in Staff Zone Alert
*   **Condition**: Customer detected inside the designated Staff Zone.
*   **Description (Raised)**: `"Violation: Customer detected in Staff Zone."`
*   **Description (Resolved)**: `"Customer in Staff Zone violation resolved."`
