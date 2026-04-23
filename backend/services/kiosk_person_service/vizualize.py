import redis
import json
import base64
import cv2
import numpy as np
import time
import os
from pathlib import Path

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
CAMERA_ID = os.getenv("CAMERA_ID", "6")  # Default to 6 as seen in settings
QUEUE_NAME = os.getenv("VIS_QUEUE_NAME", f"stream:visualization:kiosk_person:{CAMERA_ID}")

# Redis Config
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# Load Settings
def load_settings():
    # Try artifacts first, then local (relative to script)
    base_dir = Path(__file__).parent.resolve()
    paths = [
        base_dir / "artifacts" / "settings.json",
        base_dir / "settings.json"
    ]
    for p in paths:
        if p.exists():
            with open(p, "r") as f:
                return json.load(f)
    print("Warning: settings.json not found in artifacts or local dir.")
    return {}

SETTINGS = load_settings()

def decode_frame(frame_b64: str) -> np.ndarray:
    img_bytes = base64.b64decode(frame_b64)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def draw_kiosks(img, camera_id, kiosk_status, roi_stats):
    cam_config = SETTINGS.get(str(camera_id), {})
    kiosks_config = cam_config.get("kiosks", {})
    
    for key, rect in kiosks_config.items():
        x = int(rect["xMin"])
        y = int(rect["yMin"])
        w = int(rect["width"])
        h = int(rect["height"])
        x2 = x + w
        y2 = y + h
        
        # Color based on status
        status = kiosk_status.get(key, "UNKNOWN")
        
        # Colors (BGR)
        COLOR_ON = (0, 255, 0)      # Green
        COLOR_OFF = (0, 0, 255)     # Red
        COLOR_UNKNOWN = (0, 255, 255) # Yellow
        
        if status == "ON":
            color = COLOR_ON
        elif status == "OFF":
            color = COLOR_OFF
        else:
            color = COLOR_UNKNOWN

        cv2.rectangle(img, (x, y), (x2, y2), color, 2)
        
        label = f"{key}: {status}"
        cv2.putText(
            img,
            label,
            (x, max(0, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
        
        # Stats
        stats = roi_stats.get(key)
        if stats:
            desc = f"G:{int(stats.get('gray_mean', 0))}"
            cv2.putText(
                img,
                desc,
                (x, y2 + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )

def draw_polygon(img, camera_id):
    cam_config = SETTINGS.get(str(camera_id), {})
    pts = cam_config.get("target_polygon", [])
    if pts:
        points = np.array([ [int(p["x"]), int(p["y"])] for p in pts ], np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.polylines(img, [points], True, (255, 255, 0), 2) # Cyan

def draw_persons(img, detections, color=(255, 0, 0), thickness=2):
    if not detections:
        return
    for bbox in detections:
        # bbox might be [x1, y1, x2, y2]
        x1, y1, x2, y2 = [int(c) for c in bbox]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

def draw_status_panel(img, kiosk_status):
    """Draw a status panel on the right side of the frame."""
    height, width = img.shape[:2]
    panel_width = 200
    panel_height = 50 + len(kiosk_status) * 30
    
    # Draw semi-transparent background box
    overlay = img.copy()
    cv2.rectangle(overlay, (width - panel_width, 0), (width, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    
    # Title
    cv2.putText(
        img, "KIOSK STATUS", (width - panel_width + 10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
    )
    
    # Kiosk entries
    for i, (key, status) in enumerate(kiosk_status.items()):
        y_pos = 65 + i * 30
        color = (0, 255, 0) if status == "ON" else (0, 0, 255)
        
        cv2.putText(
            img, f"{key}:", (width - panel_width + 10, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        
        cv2.putText(
            img, status, (width - panel_width + 120, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

def visualize():
    print(f"Redis frame visualizer started for queue: {QUEUE_NAME}")
    print("Press 'q' to quit.")
    window_initialized = False
    while True:
        try:
            if window_initialized:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                if cv2.getWindowProperty("Kiosk Service Visualization", cv2.WND_PROP_VISIBLE) < 1:
                    break

            msg = r.blpop(QUEUE_NAME, timeout=0.1)
            if msg is None:
                continue

            # Collect all messages including the first one
            messages = [msg]
            
            # Drain queue
            while True:
                next_val = r.lpop(QUEUE_NAME)
                if next_val is None:
                    break
                messages.append((QUEUE_NAME, next_val))
            
            # Find the LATEST valid message for our service
            valid_payload = None
            
            for _, payload in reversed(messages):
                try:
                    d = json.loads(payload)
                    if d.get("service") == "kiosk_person_service":
                        valid_payload = payload
                        break
                except json.JSONDecodeError:
                    continue

            if valid_payload is None:
                continue
                
            data = json.loads(valid_payload)

            frame = decode_frame(data["frame"])
            
            camera_id = data.get("camera_id", CAMERA_ID)
            
            # Draw Logic
            draw_polygon(frame, camera_id)
            draw_kiosks(frame, camera_id, data.get("kiosk_status", {}), data.get("roi_stats", {}))
            
            # Draw All Persons (Blue, Thin)
            draw_persons(frame, data.get("all_detections", []), color=(255, 0, 0), thickness=1)
            
            # Draw Target Persons (Magenta, Thick)
            draw_persons(frame, data.get("target_person_detections", []), color=(255, 0, 255), thickness=2)
            
            # Draw Right-side Status Panel
            draw_status_panel(frame, data.get("kiosk_status", {}))
            
            # Info
            timestamp = data.get("timestamp", "")
            count = data.get("target_person_count", 0)
            
            cv2.putText(
                frame,
                f"Camera: {camera_id} | Time: {timestamp} | Count: {count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            
            cv2.imshow("Kiosk Service Visualization", frame)
            window_initialized = True
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Check if window was closed by user
            if cv2.getWindowProperty("Kiosk Service Visualization", cv2.WND_PROP_VISIBLE) < 1:
                break


                
        except Exception as e:
            print(f"[Viz Error] {e}")
            time.sleep(1)
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    visualize()
