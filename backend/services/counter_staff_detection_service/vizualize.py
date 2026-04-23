import redis
import json
import base64
import cv2
import numpy as np
import time
import os
import requests

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
CAMERA_ID = os.getenv("CAMERA_ID", "1")
QUEUE_NAME = os.getenv("VIS_QUEUE_NAME", f"stream:visualization:counter_staff:{CAMERA_ID}")

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def decode_frame(frame_b64: str) -> np.ndarray:
    """Decode base64 encoded frame to numpy array."""
    img_bytes = base64.b64decode(frame_b64)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def download_frame(frame_url: str) -> np.ndarray:
    """Download frame from URL and decode to numpy array."""
    try:
        response = requests.get(frame_url, timeout=10)
        response.raise_for_status()
        img_bytes = response.content
        np_arr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[Error] Failed to download frame from {frame_url}: {e}")
        return None


def draw_polygons(img, polygons, label, color, thickness=2):
    """
    Draw polygons on the image.
    
    Args:
        img: Image to draw on
        polygons: List of polygon coordinate lists (pixel coordinates)
        label: Label text for the polygon
        color: BGR color tuple
        thickness: Line thickness
    """
    height, width = img.shape[:2]
    
    for idx, polygon in enumerate(polygons):
        # Convert normalized coordinates to pixel coordinates if needed
        points = []
        for pt in polygon:
            if isinstance(pt, dict):
                # Normalized coordinates
                x = int(pt["x"] * width)
                y = int(pt["y"] * height)
            else:
                # Already pixel coordinates
                x, y = pt
            points.append([x, y])
        
        if points:
            pts = np.array(points, dtype=np.int32)
            # Draw polygon outline
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
            
            # Draw semi-transparent fill
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)
            
            # Add label at the first point
            if len(points) > 0:
                label_text = f"{label} {idx + 1}"
                cv2.putText(
                    img,
                    label_text,
                    (points[0][0], points[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )


def draw_staff_detections(img, staff_detections):
    """
    Draw staff detection bounding boxes in Green.
    """
    COLOR_STAFF = (0, 255, 0)  # Green
    height, width = img.shape[:2]
    
    for idx, detection in enumerate(staff_detections):
        bbox = detection["bbox"]
        
        # Check if bbox is normalized (values between 0 and 1)
        if all(0 <= coord <= 1 for coord in bbox):
            # Denormalize
            x1 = int(bbox[0] * width)
            y1 = int(bbox[1] * height)
            x2 = int(bbox[2] * width)
            y2 = int(bbox[3] * height)
        else:
            # Already pixel coordinates
            x1, y1, x2, y2 = [int(c) for c in bbox]
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_STAFF, 2)
        
        # Get confidence if available
        conf = detection.get("confidence", 0)
        label = f"Staff {idx + 1}"
        if conf > 0:
            label += f" ({conf:.2f})"
        
        cv2.putText(
            img,
            label,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_STAFF,
            2,
        )


def draw_customer_detections(img, customer_detections):
    """
    Draw customer detection bounding boxes in Orange.
    """
    COLOR_CUSTOMER = (0, 165, 255)  # Orange
    height, width = img.shape[:2]
    
    for idx, detection in enumerate(customer_detections):
        bbox = detection["bbox"]
        
        # Check if bbox is normalized (values between 0 and 1)
        if all(0 <= coord <= 1 for coord in bbox):
            # Denormalize
            x1 = int(bbox[0] * width)
            y1 = int(bbox[1] * height)
            x2 = int(bbox[2] * width)
            y2 = int(bbox[3] * height)
        else:
            # Already pixel coordinates
            x1, y1, x2, y2 = [int(c) for c in bbox]
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_CUSTOMER, 2)
        
        # Get confidence if available
        conf = detection.get("confidence", 0)
        label = f"Customer {idx + 1}"
        if conf > 0:
            label += f" ({conf:.2f})"
        
        cv2.putText(
            img,
            label,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_CUSTOMER,
            2,
        )


def draw_staff_in_customer_detections(img, staff_in_customer_detections):
    """
    Draw detections that were in customer zone but classified as staff in Grey.
    """
    COLOR_GREY = (128, 128, 128)  # Grey
    height, width = img.shape[:2]
    
    for idx, detection in enumerate(staff_in_customer_detections):
        bbox = detection["bbox"]
        
        # Check if bbox is normalized (values between 0 and 1)
        if all(0 <= coord <= 1 for coord in bbox):
            # Denormalize
            x1 = int(bbox[0] * width)
            y1 = int(bbox[1] * height)
            x2 = int(bbox[2] * width)
            y2 = int(bbox[3] * height)
        else:
            # Already pixel coordinates
            x1, y1, x2, y2 = [int(c) for c in bbox]
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_GREY, 2)
        
        label = f"Staff (Cust Zone) {idx + 1}"
        cv2.putText(
            img,
            label,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_GREY,
            2,
        )


def draw_customer_in_staff_detections(img, customer_in_staff_detections):
    """
    Draw customer in staff detection bounding boxes in Red.
    """
    COLOR_RED = (0, 0, 255)  # Red
    height, width = img.shape[:2]
    
    for idx, detection in enumerate(customer_in_staff_detections):
        bbox = detection["bbox"]
        
        # Check if bbox is normalized (values between 0 and 1)
        if all(0 <= coord <= 1 for coord in bbox):
            # Denormalize
            x1 = int(bbox[0] * width)
            y1 = int(bbox[1] * height)
            x2 = int(bbox[2] * width)
            y2 = int(bbox[3] * height)
        else:
            # Already pixel coordinates
            x1, y1, x2, y2 = [int(c) for c in bbox]
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_RED, 2)
        
        label = f"Cust in Staff Zone {idx + 1}"
        cv2.putText(
            img,
            label,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_RED,
            2,
        )


def visualize():
    print(f"Redis frame visualizer started for queue: {QUEUE_NAME}")
    print(f"Camera ID: {CAMERA_ID}")
    print("Press 'q' to quit")
    window_initialized = False
    while True:
        try:
            if window_initialized:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if cv2.getWindowProperty("Counter Staff Detection Visualization", cv2.WND_PROP_VISIBLE) < 1:
                    break

            msg = r.blpop(QUEUE_NAME, timeout=0.1)
            if msg is None:
                continue

            _, payload = msg
            data = json.loads(payload)

            # Get frame - either from base64 or URL
            frame = None
            if "frame" in data:
                frame = decode_frame(data["frame"])
            elif "frame_url" in data:
                frame = download_frame(data["frame_url"])
            
            if frame is None:
                print("[Warning] No frame available in message")
                continue

            # Draw staff polygons (Green)
            staff_polygons = data.get("staff_polygons", [])
            if staff_polygons:
                draw_polygons(frame, staff_polygons, "Staff Zone", (0, 255, 0))

            # Draw customer polygons (Orange)
            customer_polygons = data.get("customer_polygons", [])
            if customer_polygons:
                draw_polygons(frame, customer_polygons, "Customer Zone", (0, 165, 255))

            # Draw staff detections
            staff_detections = data.get("staff_detection", data.get("valid_detection", []))
            if staff_detections:
                draw_staff_detections(frame, staff_detections)

            # Draw customer detections
            customer_detections = data.get("customer_detection", [])
            if customer_detections:
                draw_customer_detections(frame, customer_detections)

            # Draw staff found in customer zone
            staff_in_customer_detections = data.get("staff_in_customer_detections", [])
            if staff_in_customer_detections:
                draw_staff_in_customer_detections(frame, staff_in_customer_detections)

            # Draw customers found in staff zone
            # customer_in_staff_detections = data.get("customer_in_staff_detections", [])
            # if customer_in_staff_detections:
            #     draw_customer_in_staff_detections(frame, customer_in_staff_detections)

            # Overlay metadata
            camera_id = data.get("camera_id", "unknown")
            timestamp = data.get("timestamp", "")
            is_staff_detected = data.get("is_staff_detected", False)
            is_customer_detected = data.get("is_customer_detected", False)
            customer_count = data.get("customer_count", len(customer_detections))
            
            # Status information
            y_offset = 30
            cv2.putText(
                frame,
                f"Camera: {camera_id} | Time: {timestamp}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            
            y_offset += 60
            staff_status = "PRESENT" if is_staff_detected else "ABSENT"
            staff_color = (0, 255, 0) if is_staff_detected else (0, 0, 255)
            cv2.putText(
                frame,
                f"Staff: {staff_status} ({len(staff_detections)} detected)",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                staff_color,
                2,
            )
            
            y_offset += 30
            customer_status = "PRESENT" if is_customer_detected else "ABSENT"
            customer_color = (0, 165, 255) if is_customer_detected else (128, 128, 128)
            cv2.putText(
                frame,
                f"Customers: {customer_status}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            
            y_offset += 25
            stats = data.get("stats", {})
            mode_val = stats.get("mode", 0)
            cv2.putText(
                frame,
                f"Count: {customer_count} (Mode: {mode_val})",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            
            # # Display Statistics
            # stats = data.get("stats", {})
            # if stats:
            #     y_offset += 25
            #     mean_val = stats.get("mean", 0)
            #     median_val = stats.get("median", 0)
            #     mode_val = stats.get("mode", 0)
                
            #     stats_text = f"Mean: {mean_val:.1f} | Median: {median_val:.1f} | Mode: {mode_val}"
            #     cv2.putText(
            #         frame,
            #         stats_text,
            #         (10, y_offset),
            #         cv2.FONT_HERSHEY_SIMPLEX,
            #         0.5,
            #         (0, 255, 0),
            #         1,
            #     )
            
            # Alert status
            is_unattended = (not is_staff_detected) and is_customer_detected
            if is_unattended:
                y_offset += 30
                cv2.putText(
                    frame,
                    "ALERT: UNATTENDED CUSTOMER!",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    3,
                )

            cv2.imshow("Counter Staff Detection Visualization", frame)

            window_initialized = True   
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            
                                    # Check if window was closed by user
            if cv2.getWindowProperty("Counter Staff Detection Visualization", cv2.WND_PROP_VISIBLE) < 1:
                break



        except Exception as e:
            print(f"[Visualizer Error] {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    visualize()
