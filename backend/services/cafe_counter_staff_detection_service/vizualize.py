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
CAMERA_ID = os.getenv("CAMERA_ID", "2")
QUEUE_NAME = os.getenv("VIS_QUEUE_NAME", f"stream:visualization:cafe_counter:{CAMERA_ID}")

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# Colors for different regions (BGR)
REGION_COLORS = [
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
]



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


def draw_text_with_bg(img, text, position, font_scale=0.5, font_thickness=2, text_color=(0, 255, 0), bg_color=(0, 0, 0), padding=2):
    """
    Draw text with a background rectangle for better legibility.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    
    x, y = position
    # Ensure text is within bounds (rudimentary)
    x = max(padding, x)
    y = max(text_h + padding, y)
    
    # Draw background rectangle
    cv2.rectangle(img, (x - padding, y - text_h - padding), (x + text_w + padding, y + padding), bg_color, -1)
    # Draw text
    cv2.putText(img, text, (x, y), font, font_scale, text_color, font_thickness)
    
    return text_w + 2 * padding


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
                label_text = f"{label}"
                cv2.putText(
                    img,
                    label_text,
                    (points[0][0], points[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )



def draw_line(img, line_points, label, color, thickness=3):
    """
    Draw a line on the image.
    
    Args:
        img: Image to draw on
        line_points: List of two points [(x1, y1), (x2, y2)]
        label: Label text for the line
        color: BGR color tuple
        thickness: Line thickness
    """
    if len(line_points) >= 2:
        p1 = line_points[0]
        p2 = line_points[1]
        
        # Convert to int if needed
        x1, y1 = int(p1[0]), int(p1[1])
        x2, y2 = int(p2[0]), int(p2[1])
        
        # Draw the line
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
        
        # Add label at midpoint
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        cv2.putText(
            img,
            label,
            (mid_x, mid_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )


def draw_staff_detections(img, staff_detections):
    """
    Draw staff detection bounding boxes. Use region_id for coloring if available.
    """
    height, width = img.shape[:2]
    
    for idx, detection in enumerate(staff_detections):
        bbox = detection["bbox"]
        region_id = detection.get("region_id", -1)
        
        # Select color based on region_id
        if region_id >= 0:
            color = REGION_COLORS[region_id % len(REGION_COLORS)]
        else:
            color = (0, 255, 0)  # Default Green
            
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
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Get confidence if available
        conf = detection.get("confidence", 0)
        label = f"Staff {idx + 1}"
        if region_id >= 0:
            label += f" [R{region_id + 1}]"
        if conf > 0:
            label += f" ({conf:.2f})"
        
        cv2.putText(
            img,
            label,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
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
        depth = detection.get("depth")
        
        main_label = f"Customer {idx + 1}"
        if conf > 0:
            main_label += f" ({conf:.2f})"
        
        # Draw main label
        cv2.putText(
            img,
            main_label,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_CUSTOMER,
            2,
        )
        
        # Draw depth with background if available
        if depth is not None:
            depth_text = f"D:{depth:.1f}"
            # Draw it below the main label or inside the box
            draw_text_with_bg(img, depth_text, (x1, max(0, y1 - 25)))


def draw_all_hits(img, hits):
    """
    Draw all person detections that overlap with a customer region.
    Drawn in Yellow with a dashed-like appearance (thin lines).
    """
    COLOR_ALL_HITS = (0, 255, 255)  # Yellow
    height, width = img.shape[:2]

    for idx, detection in enumerate(hits):
        bbox = detection["bbox"]

        # Check if bbox is normalized
        if all(0 <= coord <= 1 for coord in bbox):
            x1 = int(bbox[0] * width)
            y1 = int(bbox[1] * height)
            x2 = int(bbox[2] * width)
            y2 = int(bbox[3] * height)
        else:
            x1, y1, x2, y2 = [int(c) for c in bbox]

        # Draw thin bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_ALL_HITS, 1)

        depth = detection.get("depth")
        # Draw overlap label (thin)
        label = f"Overlap {idx + 1}"
        cv2.putText(
            img,
            label,
            (x1, min(height, y2 + 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            COLOR_ALL_HITS,
            1,
        )
        
        if depth is not None:
            depth_text = f"D:{depth:.1f}"
            draw_text_with_bg(img, depth_text, (x1, min(height, y2 + 35)), font_scale=0.4, font_thickness=1)


def visualize():
    print(f"Redis frame visualizer started for queue: {QUEUE_NAME}")
    print(f"Cafe Counter Staff Detection - Camera ID: {CAMERA_ID}")
    print("Press 'q' to quit")
    window_initialized = False
    
    while True:
        try:
            if window_initialized:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                if cv2.getWindowProperty("Cafe Counter Staff Detection Visualization", cv2.WND_PROP_VISIBLE) < 1:
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

            # Draw staff polygons (Multi-color)
            staff_polygons = data.get("staff_polygons", [])
            for i, poly in enumerate(staff_polygons):
                color = REGION_COLORS[i % len(REGION_COLORS)]
                draw_polygons(frame, [poly], f"Staff Zone {i + 1}", color)

            # Draw customer polygons (Orange)
            customer_polygons = data.get("customer_polygons", [])
            for i, poly in enumerate(customer_polygons):
                color = (0, 165, 255) # Orange
                draw_polygons(frame, [poly], f"Customer Zone {i + 1}", color)


            # Draw customer detection line (Orange/Cyan)
            customer_line = data.get("customer_line", [])
            if customer_line and len(customer_line) >= 2:
                draw_line(frame, customer_line, "Customer Line", (0, 255, 255))

            # Draw staff detections
            staff_detections = data.get("staff_detection", data.get("valid_detection", []))
            if staff_detections:
                draw_staff_detections(frame, staff_detections)

            # Draw customer detections
            customer_detections = data.get("customer_detection", [])
            if customer_detections:
                draw_customer_detections(frame, customer_detections)

            # Draw all overlapping hits (Yellow)
            all_hits = data.get("all_customer_hits", [])
            if all_hits:
                draw_all_hits(frame, all_hits)

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
            
            y_offset += 30
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
                f"Customers: {customer_status} (Count: {customer_count})",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                customer_color,
                2,
            )
            
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

            cv2.imshow("Cafe Counter Staff Detection Visualization", frame)

            window_initialized = True
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Check if window was closed by user
            if cv2.getWindowProperty("Cafe Counter Staff Detection Visualization", cv2.WND_PROP_VISIBLE) < 1:
                break



        except Exception as e:
            print(f"[Visualizer Error] {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    visualize()
