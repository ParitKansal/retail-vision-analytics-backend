"""
Visualization for Total Occupancy Count - LEFT LOBBY

This module provides visualization tools to display:
- Person bounding boxes with their in-region status
- Region polygon overlays
- Total occupancy count display
"""

import redis
import json
import base64
import cv2
import numpy as np
import time
import os
from typing import List, Dict, Any

from total_occupancy_patch import (
    load_region_config,
    count_persons_in_region,
    get_bbox_bottom_points,
    point_in_polygon,
    calculate_bbox_region_overlap_ratio
)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
CAMERA_ID = os.getenv("CAMERA_ID", "4")
QUEUE_NAME = os.getenv("VIS_QUEUE_NAME", f"stream:visualization:{CAMERA_ID}")
REGION_KEY = os.getenv("REGION_KEY", "left")

COLORS = {
    "region_polygon": (255, 255, 0),
    "region_fill": (255, 255, 0),
    "person_inside": (0, 255, 0),
    "person_outside": (0, 0, 255),
    "person_bottom_line": (255, 0, 255),
    "text_bg": (0, 0, 0),
    "text_fg": (255, 255, 255),
    "count_display": (0, 255, 255),
    "table_empty": (0, 255, 0),
    "table_occupied": (0, 0, 255),
    "table_dirty": (0, 255, 255),
    "plate": (255, 255, 0),
}

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def decode_frame(frame_b64: str) -> np.ndarray:
    img_bytes = base64.b64decode(frame_b64)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def draw_region_polygon(img: np.ndarray, regions: List[Dict], fill_opacity: float = 0.15):
    overlay = img.copy()
    for region in regions:
        points = region.get("points", [])
        if not points:
            continue
        pts = np.array([[int(p['x']), int(p['y'])] for p in points], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], COLORS["region_fill"])
        cv2.polylines(img, [pts], True, COLORS["region_polygon"], 2)
        region_name = region.get("name", region.get("id", "Region"))
        if points:
            min_x = min(p['x'] for p in points)
            min_y = min(p['y'] for p in points)
            cv2.putText(img, region_name, (int(min_x) + 5, int(min_y) + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["region_polygon"], 2)
    cv2.addWeighted(overlay, fill_opacity, img, 1 - fill_opacity, 0, img)


def draw_person_with_status(img: np.ndarray, person: Dict, idx: int, detail: Dict, regions: List[Dict]):
    bbox = person.get("bbox", [])
    if len(bbox) != 4:
        return
    x1, y1, x2, y2 = [int(c) for c in bbox]
    is_inside = detail.get("is_inside", False)
    reason = detail.get("reason", "unknown")
    color = COLORS["person_inside"] if is_inside else COLORS["person_outside"]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    bottom_left, bottom_right, bottom_mid = get_bbox_bottom_points(bbox)
    point_color = COLORS["person_bottom_line"]
    cv2.circle(img, (int(bottom_left[0]), int(bottom_left[1])), 5, point_color, -1)
    cv2.circle(img, (int(bottom_right[0]), int(bottom_right[1])), 5, point_color, -1)
    cv2.circle(img, (int(bottom_mid[0]), int(bottom_mid[1])), 5, point_color, -1)
    cv2.line(img, (x1, y2), (x2, y2), point_color, 2)
    status_text = "IN" if is_inside else "OUT"
    label = f"P{idx}: {status_text}"
    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.rectangle(img, (x1, y1 - label_h - 6), (x1 + label_w + 4, y1), COLORS["text_bg"], -1)
    cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    if is_inside:
        reason_text = f"Reason: {reason}"
    else:
        reason_text = "Outside region"
        if regions:
            for region in regions:
                polygon = region.get("points", [])
                if polygon:
                    overlap = calculate_bbox_region_overlap_ratio(bbox, polygon)
                    reason_text = f"Overlap: {overlap:.1%}"
                    break
    (reason_w, reason_h), _ = cv2.getTextSize(reason_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    cv2.rectangle(img, (x1, y2 + 2), (x1 + reason_w + 4, y2 + reason_h + 8), COLORS["text_bg"], -1)
    cv2.putText(img, reason_text, (x1 + 2, y2 + reason_h + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["text_fg"], 1)


def draw_total_count_display(img: np.ndarray, total_count: int, persons_inside: List[int], total_persons: int):
    count_text = f"TOTAL IN REGION: {total_count}/{total_persons}"
    img_height, img_width = img.shape[:2]
    (text_w, text_h), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
    x = img_width - text_w - 20
    y = 50
    panel_padding = 15
    cv2.rectangle(img, (x - panel_padding, y - text_h - panel_padding),
                 (x + text_w + panel_padding, y + panel_padding), COLORS["text_bg"], -1)
    cv2.rectangle(img, (x - panel_padding, y - text_h - panel_padding),
                 (x + text_w + panel_padding, y + panel_padding), COLORS["count_display"], 2)
    cv2.putText(img, count_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLORS["count_display"], 3)
    if persons_inside:
        indices_text = f"Persons: {', '.join(map(str, persons_inside))}"
        cv2.putText(img, indices_text, (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text_fg"], 1)


def draw_table_boxes(img: np.ndarray, table_bboxes: List[Dict], table_status: Dict):
    for table in table_bboxes:
        bbox = table.get("bbox", [])
        table_id = table.get("id", "?")
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(c) for c in bbox]
        status = table_status.get(str(table_id), table_status.get(table_id, "EMPTY"))
        if status == "OCCUPIED":
            color = COLORS["table_occupied"]
        elif status == "DIRTY":
            color = COLORS["table_dirty"]
        else:
            color = COLORS["table_empty"]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"T{table_id}: {status}", (x1, max(0, y1 - 6)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_plates(img: np.ndarray, plates: List[Dict]):
    for idx, plate in enumerate(plates):
        bbox = plate.get("bbox", [])
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(c) for c in bbox]
        score = plate.get("score", 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), COLORS["plate"], 2)
        cv2.putText(img, f"Plate ({score:.2f})" if score else f"Plate {idx}",
                   (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["plate"], 2)


def visualize_total_occupancy():
    print(f"Total Occupancy Visualizer started for queue: {QUEUE_NAME}")
    print(f"Using region key: {REGION_KEY}")
    regions = load_region_config(region_key=REGION_KEY)
    if not regions:
        print(f"WARNING: No regions found for key '{REGION_KEY}'")
    else:
        print(f"Loaded {len(regions)} region(s)")
    
    window_initialized = False
    while True:
        try:
            msg = r.blpop(QUEUE_NAME, timeout=0.1)
            
            if msg is None:
                if window_initialized:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    try:
                        if cv2.getWindowProperty("Total Occupancy Visualization - LEFT LOBBY", cv2.WND_PROP_VISIBLE) < 1:
                            break
                    except:
                        break
                continue

            if window_initialized:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if cv2.getWindowProperty("Total Occupancy Visualization - LEFT LOBBY", cv2.WND_PROP_VISIBLE) < 1:
                    break

            last_msg = msg
            while True:
                next_msg = r.lpop(QUEUE_NAME)
                if next_msg is None:
                    break
                last_msg = (QUEUE_NAME, next_msg)
            _, payload = last_msg
            data = json.loads(payload)
            frame = decode_frame(data["frame"])
            persons = data.get("persons", [])
            occupancy_result = data.get("occupancy_result")
            if occupancy_result is None:
                occupancy_result = count_persons_in_region(persons, regions=regions, overlap_threshold=0.6)
            viz_regions = data.get("occupancy_regions", regions)
            draw_region_polygon(frame, viz_regions)
            draw_table_boxes(frame, data.get("table_bboxes", []), data.get("table_status", {}))
            draw_plates(frame, data.get("plates", []))
            for idx, person in enumerate(persons):
                detail = occupancy_result["details"][idx] if idx < len(occupancy_result["details"]) else {}
                draw_person_with_status(frame, person, idx, detail, viz_regions)
            draw_total_count_display(frame, occupancy_result["total_count"], occupancy_result["persons_inside"], len(persons))
            camera_id = data.get("camera_id", "unknown")
            timestamp = data.get("timestamp", "")
            cv2.putText(frame, f"Camera: {camera_id} | Time: {timestamp}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["text_fg"], 2)
            print(f"\n--- Frame Stats ---")
            print(f"Total persons detected: {len(persons)}")
            print(f"Persons in region: {occupancy_result['total_count']}")
            cv2.imshow("Total Occupancy Visualization - LEFT LOBBY", frame)
            window_initialized = True
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if cv2.getWindowProperty("Total Occupancy Visualization - LEFT LOBBY", cv2.WND_PROP_VISIBLE) < 1:
                break
        except Exception as e:
            print(f"[Visualizer Error] {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    visualize_total_occupancy()
