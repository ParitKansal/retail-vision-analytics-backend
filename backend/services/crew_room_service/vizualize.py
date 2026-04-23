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
CAMERA_ID = os.getenv("CAMERA_ID", "5")
QUEUE_NAME = os.getenv("VIS_QUEUE_NAME", f"stream:visualization:{CAMERA_ID}")

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ─────────────────────────────────────────────────────────────
# Color scheme
# ─────────────────────────────────────────────────────────────
COLOR_GREEN  = (0, 255, 0)      # < 50% of threshold
COLOR_YELLOW = (0, 255, 255)    # 50–75%
COLOR_ORANGE = (0, 165, 255)    # 75–100%
COLOR_RED    = (0, 0, 255)      # exceeded threshold
COLOR_WHITE  = (255, 255, 255)
COLOR_BLACK  = (0, 0, 0)
COLOR_CYAN   = (255, 255, 0)

THRESHOLD_MINUTES = int(os.getenv("DURATION_THRESHOLD_MINUTES", "30"))
THRESHOLD_SECONDS = THRESHOLD_MINUTES * 60


def get_duration_color(duration_seconds):
    """Return color based on how close person is to threshold."""
    progress = duration_seconds / THRESHOLD_SECONDS
    if progress >= 1.0:
        return COLOR_RED
    elif progress >= 0.75:
        return COLOR_ORANGE
    elif progress >= 0.5:
        return COLOR_YELLOW
    else:
        return COLOR_GREEN


def format_duration(seconds):
    """Format seconds into MM:SS string."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def decode_frame(frame_b64: str) -> np.ndarray:
    img_bytes = base64.b64decode(frame_b64)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def download_frame(frame_url: str) -> np.ndarray:
    try:
        response = requests.get(frame_url, timeout=10)
        response.raise_for_status()
        np_arr = np.frombuffer(response.content, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[Error] Failed to download frame: {e}")
        return None


# def draw_tracked_persons(img, tracked_persons):
#     """
#     Draw tracked persons with duration info and color-coded boxes.
#     tracked_persons format:
#     [
#         {
#             'track_id': 1,
#             'bbox': [x1, y1, x2, y2],
#             'duration_seconds': 120.5,
#             'duration_minutes': 2.0,
#             'exceeded_threshold': False,
#             'total_frames': 100
#         },
#         ...
#     ]
#     """
#     height, width = img.shape[:2]

#     for person in tracked_persons:
#         bbox             = person.get("bbox", [])
#         track_id         = person.get("track_id", "?")
#         duration_seconds = person.get("duration_seconds", 0)
#         exceeded         = person.get("exceeded_threshold", False)
#         total_frames     = person.get("total_frames", 0)

#         if not bbox or len(bbox) < 4:
#             continue

#         # Handle normalized vs pixel coordinates
#         if all(0 <= c <= 1 for c in bbox):
#             x1 = int(bbox[0] * width)
#             y1 = int(bbox[1] * height)
#             x2 = int(bbox[2] * width)
#             y2 = int(bbox[3] * height)
#         else:
#             x1, y1, x2, y2 = [int(c) for c in bbox]

#         # Pick color based on duration
#         color = get_duration_color(duration_seconds)

#         # Draw bounding box - thicker if exceeded
#         thickness = 4 if exceeded else 2
#         cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

#         # ── Label background + text ──────────────────────────
#         duration_str = format_duration(duration_seconds)
#         progress_pct = min(100, int((duration_seconds / THRESHOLD_SECONDS) * 100))

#         line1 = f"ID:{track_id}  {duration_str}"
#         line2 = f"{progress_pct}% of {THRESHOLD_MINUTES}min"
#         if exceeded:
#             line2 = f"⚠ EXCEEDED {duration_str}"

#         font       = cv2.FONT_HERSHEY_SIMPLEX
#         font_scale = 0.55
#         thickness_text = 2

#         (tw1, th1), _ = cv2.getTextSize(line1, font, font_scale, thickness_text)
#         (tw2, th2), _ = cv2.getTextSize(line2, font, font_scale, thickness_text)

#         label_w = max(tw1, tw2) + 10
#         label_h = th1 + th2 + 20

#         # Position label above box, clamp to frame
#         lx = max(0, x1)
#         ly = max(label_h, y1) - label_h

#         # Semi-transparent background
#         overlay = img.copy()
#         cv2.rectangle(overlay, (lx, ly), (lx + label_w, ly + label_h), COLOR_BLACK, -1)
#         cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

#         # Text lines
#         cv2.putText(img, line1, (lx + 5, ly + th1 + 5),
#                     font, font_scale, color, thickness_text)
#         cv2.putText(img, line2, (lx + 5, ly + th1 + th2 + 12),
#                     font, font_scale, COLOR_WHITE if not exceeded else COLOR_RED,
#                     thickness_text)

#         # ── Duration progress bar ────────────────────────────
#         bar_x1 = x1
#         bar_x2 = x2
#         bar_y  = y2 + 6
#         bar_h  = 8

#         if bar_y + bar_h < height:
#             # Background bar
#             cv2.rectangle(img, (bar_x1, bar_y), (bar_x2, bar_y + bar_h),
#                           (80, 80, 80), -1)
#             # Filled portion
#             fill_w = int((bar_x2 - bar_x1) * min(1.0, duration_seconds / THRESHOLD_SECONDS))
#             if fill_w > 0:
#                 cv2.rectangle(img, (bar_x1, bar_y), (bar_x1 + fill_w, bar_y + bar_h),
#                               color, -1)

def draw_tracked_persons(img, tracked_persons):
    height, width = img.shape[:2]

    # Sort by y position so labels don't overlap predictably
    sorted_persons = sorted(
        tracked_persons,
        key=lambda p: p.get("bbox", [0, 0, 0, 0])[1] if p.get("bbox") else 0
    )

    used_label_rects = []  # Track used label positions to avoid overlap

    for person in sorted_persons:
        bbox             = person.get("bbox", [])
        track_id         = person.get("track_id", "?")
        duration_seconds = person.get("duration_seconds", 0)
        exceeded         = person.get("exceeded_threshold", False)

        if not bbox or len(bbox) < 4:
            continue

        # Handle normalized vs pixel coordinates
        if all(0 <= c <= 1 for c in bbox):
            x1 = int(bbox[0] * width)
            y1 = int(bbox[1] * height)
            x2 = int(bbox[2] * width)
            y2 = int(bbox[3] * height)
        else:
            x1, y1, x2, y2 = [int(c) for c in bbox]

        color     = get_duration_color(duration_seconds)
        thickness = 4 if exceeded else 2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # ── Build label text ─────────────────────────────────
        duration_str = format_duration(duration_seconds)
        progress_pct = min(100, int((duration_seconds / THRESHOLD_SECONDS) * 100))
        line1 = f"ID:{track_id}  {duration_str}"
        line2 = f"EXCEEDED!" if exceeded else f"{progress_pct}% of {THRESHOLD_MINUTES}min"

        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thick = 1

        (tw1, th1), _ = cv2.getTextSize(line1, font, font_scale, font_thick)
        (tw2, th2), _ = cv2.getTextSize(line2, font, font_scale, font_thick)

        label_w = max(tw1, tw2) + 10
        label_h = th1 + th2 + 16

        # ── Find non-overlapping position for label ──────────
        # Try above box first, then below, then inside top
        candidates = [
            (x1, y1 - label_h - 4),        # above box
            (x1, y2 + 4),                   # below box
            (x1, y1 + 4),                   # inside top of box
        ]

        lx, ly = x1, max(0, y1 - label_h - 4)  # default

        for cx, cy in candidates:
            # Clamp to frame boundaries
            cx = max(0, min(cx, width - label_w))
            cy = max(0, min(cy, height - label_h))

            # Check overlap with already placed labels
            new_rect = (cx, cy, cx + label_w, cy + label_h)
            overlap = False
            for ur in used_label_rects:
                if not (new_rect[2] < ur[0] or new_rect[0] > ur[2] or
                        new_rect[3] < ur[1] or new_rect[1] > ur[3]):
                    overlap = True
                    break

            if not overlap:
                lx, ly = cx, cy
                break

        used_label_rects.append((lx, ly, lx + label_w, ly + label_h))

        # ── Draw label background ────────────────────────────
        overlay = img.copy()
        cv2.rectangle(overlay, (lx, ly), (lx + label_w, ly + label_h),
                      (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

        # ── Draw label text ──────────────────────────────────
        cv2.putText(img, line1, (lx + 5, ly + th1 + 4),
                    font, font_scale, color, font_thick)
        text2_color = COLOR_RED if exceeded else COLOR_WHITE
        cv2.putText(img, line2, (lx + 5, ly + th1 + th2 + 10),
                    font, font_scale, text2_color, font_thick)

        # ── Progress bar below bbox ──────────────────────────
        bar_y = y2 + 6
        if bar_y + 6 < height:
            cv2.rectangle(img, (x1, bar_y), (x2, bar_y + 6), (60, 60, 60), -1)
            fill_w = int((x2 - x1) * min(1.0, duration_seconds / THRESHOLD_SECONDS))
            if fill_w > 0:
                cv2.rectangle(img, (x1, bar_y), (x1 + fill_w, bar_y + 6), color, -1)



def draw_dashboard(img, data):
    """Draw top HUD with summary statistics."""
    height, width = img.shape[:2]

    tracked_persons  = data.get("tracked_persons", [])  # list of dicts
    num_persons      = data.get("num_persons", 0)
    exceeded_count   = data.get("exceeded_count", 0)
    camera_id        = data.get("camera_id", "?")
    timestamp        = data.get("timestamp", "")

    # If tracked_persons is a count (int) rather than list, handle gracefully
    if isinstance(tracked_persons, int):
        tracked_count = tracked_persons
        tracked_persons = []
    else:
        tracked_count = len(tracked_persons)

    # ── Top bar background ───────────────────────────────────
    cv2.rectangle(img, (0, 0), (width, 90), (20, 20, 20), -1)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Row 1
    cv2.putText(img, f"Camera: {camera_id}",
                (10, 28), font, 0.7, COLOR_WHITE, 2)
    cv2.putText(img, f"Time: {timestamp}",
                (200, 28), font, 0.6, (180, 180, 180), 1)

    # Row 2 - stats
    cv2.putText(img, f"Detected: {num_persons}",
                (10, 62), font, 0.7, COLOR_CYAN, 2)
    cv2.putText(img, f"Tracked: {tracked_count}",
                (200, 62), font, 0.7, COLOR_GREEN, 2)

    # Exceeded count - highlight in red if any
    exceeded_color = COLOR_RED if exceeded_count > 0 else COLOR_WHITE
    cv2.putText(img, f"Exceeded {THRESHOLD_MINUTES}min: {exceeded_count}",
                (380, 62), font, 0.7, exceeded_color, 2)

    # ── Legend (bottom-right) ────────────────────────────────
    legend_items = [
        (COLOR_GREEN,  f"< 50% ({THRESHOLD_MINUTES//2}min)"),
        (COLOR_YELLOW, f"50-75%"),
        (COLOR_ORANGE, f"75-100%"),
        (COLOR_RED,    f"Exceeded!"),
    ]
    lx = width - 180
    ly = height - (len(legend_items) * 25) - 10
    for i, (col, text) in enumerate(legend_items):
        y = ly + i * 25
        cv2.rectangle(img, (lx, y), (lx + 16, y + 16), col, -1)
        cv2.putText(img, text, (lx + 22, y + 13), font, 0.5, COLOR_WHITE, 1)


def visualize():
    print(f"Crew Room Tracker Visualizer | Queue: {QUEUE_NAME}")
    print(f"Threshold: {THRESHOLD_MINUTES} minutes | Press 'q' to quit")

    while True:
        try:
            msg = r.blpop(QUEUE_NAME, timeout=5)
            if msg is None:
                continue

            _, payload = msg
            data = json.loads(payload)

            # ── Get frame ────────────────────────────────────
            frame = None
            if "frame" in data:
                frame = decode_frame(data["frame"])
            elif "frame_url" in data:
                frame = download_frame(
                    data["frame_url"].replace("minio", "localhost")
                )

            if frame is None:
                print("[Warning] No frame available")
                continue

            # ── Draw tracked persons with duration ───────────
            tracked_persons = data.get("tracked_persons", [])
            if isinstance(tracked_persons, list) and len(tracked_persons) > 0:
                draw_tracked_persons(frame, tracked_persons)

            # ── Draw HUD dashboard ───────────────────────────
            draw_dashboard(frame, data)

            # ── Show ─────────────────────────────────────────
            cv2.imshow("Crew Room Tracker", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            try:
                if cv2.getWindowProperty("Crew Room Tracker", cv2.WND_PROP_AUTOSIZE) < 0:
                    break
            except cv2.error:
                break

        except Exception as e:
            print(f"[Visualizer Error] {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    visualize()