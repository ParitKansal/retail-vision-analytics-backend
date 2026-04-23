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
CAMERA_ID  = os.getenv("CAMERA_ID", "1")
QUEUE_NAME = os.getenv(
    "VIS_QUEUE_NAME",
    f"stream:visualization:customer_path:{CAMERA_ID}"
)

# Threshold in seconds (must match service env var)
WAITING_TIME_THRESHOLD = float(os.getenv("WAITING_TIME_THRESHOLD", "300"))

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ─────────────────────────────────────────────────────────────
# Color scheme  (mirrors crew_room visualizer)
# ─────────────────────────────────────────────────────────────
COLOR_GREEN  = (0, 255, 0)      # < 50 % of threshold
COLOR_YELLOW = (0, 255, 255)    # 50–75 %
COLOR_ORANGE = (0, 165, 255)    # 75–100 %
COLOR_RED    = (0, 0, 255)      # exceeded
COLOR_WHITE  = (255, 255, 255)
COLOR_BLACK  = (0, 0, 0)
COLOR_CYAN   = (255, 255, 0)
COLOR_GRAY   = (180, 180, 180)


def get_duration_color(duration_seconds: float):
    """Return bbox color based on how close the person is to the threshold."""
    progress = duration_seconds / WAITING_TIME_THRESHOLD
    if progress >= 1.0:
        return COLOR_RED
    elif progress >= 0.75:
        return COLOR_ORANGE
    elif progress >= 0.5:
        return COLOR_YELLOW
    return COLOR_GREEN


def format_duration(seconds: float) -> str:
    """Format seconds into MM:SS string."""
    minutes = int(seconds // 60)
    secs    = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


# ─────────────────────────────────────────────────────────────
# Frame helpers
# ─────────────────────────────────────────────────────────────

def decode_frame(frame_b64: str) -> np.ndarray:
    img_bytes = base64.b64decode(frame_b64)
    np_arr    = np.frombuffer(img_bytes, np.uint8)
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


# ─────────────────────────────────────────────────────────────
# Draw tracked persons with duration (mirrors crew_room draw_tracked_persons)
# ─────────────────────────────────────────────────────────────

def draw_tracked_persons(img, tracked_persons: list):
    """
    Overlay bounding boxes, duration labels, and progress bars for each
    tracked person.  tracking_details schema (from service.py):
      {
        track_id, bbox, duration_seconds, duration_minutes, exceeded_threshold
      }
    """
    height, width = img.shape[:2]

    # Sort top-to-bottom so label placement is predictable
    sorted_persons = sorted(
        tracked_persons,
        key=lambda p: p.get("bbox", [0, 0, 0, 0])[1] if p.get("bbox") else 0
    )

    used_label_rects = []

    for person in sorted_persons:
        bbox             = person.get("bbox", [])
        track_id         = person.get("track_id", "?")
        duration_seconds = person.get("duration_seconds", 0)
        exceeded         = person.get("exceeded_threshold", False)

        if not bbox or len(bbox) < 4:
            continue

        # Handle normalised vs pixel coordinates
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

        # ── Label lines ──────────────────────────────────────
        duration_str = format_duration(duration_seconds)
        progress_pct = min(100, int((duration_seconds / WAITING_TIME_THRESHOLD) * 100))
        threshold_label = f"{int(WAITING_TIME_THRESHOLD // 60)}m{int(WAITING_TIME_THRESHOLD % 60):02d}s"
        line1 = f"ID:{track_id}  {duration_str}"
        line2 = "EXCEEDED!" if exceeded else f"{progress_pct}% of {threshold_label}"

        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thick = 1

        (tw1, th1), _ = cv2.getTextSize(line1, font, font_scale, font_thick)
        (tw2, th2), _ = cv2.getTextSize(line2, font, font_scale, font_thick)

        label_w = max(tw1, tw2) + 10
        label_h = th1 + th2 + 16

        # ── Find non-overlapping label position ──────────────
        candidates = [
            (x1, y1 - label_h - 4),   # above box
            (x1, y2 + 4),             # below box
            (x1, y1 + 4),             # inside top of box
        ]

        lx, ly = x1, max(0, y1 - label_h - 4)

        for cx, cy in candidates:
            cx = max(0, min(cx, width  - label_w))
            cy = max(0, min(cy, height - label_h))
            new_rect = (cx, cy, cx + label_w, cy + label_h)
            overlap  = any(
                not (new_rect[2] < ur[0] or new_rect[0] > ur[2] or
                     new_rect[3] < ur[1] or new_rect[1] > ur[3])
                for ur in used_label_rects
            )
            if not overlap:
                lx, ly = cx, cy
                break

        used_label_rects.append((lx, ly, lx + label_w, ly + label_h))

        # ── Label background ─────────────────────────────────
        overlay = img.copy()
        cv2.rectangle(overlay, (lx, ly), (lx + label_w, ly + label_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

        cv2.putText(img, line1, (lx + 5, ly + th1 + 4),
                    font, font_scale, color, font_thick)
        text2_color = COLOR_RED if exceeded else COLOR_WHITE
        cv2.putText(img, line2, (lx + 5, ly + th1 + th2 + 10),
                    font, font_scale, text2_color, font_thick)

        # ── Progress bar below bbox ──────────────────────────
        bar_y = y2 + 6
        if bar_y + 6 < height:
            cv2.rectangle(img, (x1, bar_y), (x2, bar_y + 6), (60, 60, 60), -1)
            fill_w = int((x2 - x1) * min(1.0, duration_seconds / WAITING_TIME_THRESHOLD))
            if fill_w > 0:
                cv2.rectangle(img, (x1, bar_y), (x1 + fill_w, bar_y + 6), color, -1)


# ─────────────────────────────────────────────────────────────
# HUD / Dashboard
# ─────────────────────────────────────────────────────────────

def draw_dashboard(img, data: dict):
    """Draw top HUD bar with summary statistics."""
    height, width = img.shape[:2]

    tracking_details = data.get("tracking_details", [])
    num_persons      = data.get("num_persons", 0)
    exceeded_count   = data.get("exceeded_count", 0)
    camera_id        = data.get("camera_id", "?")
    timestamp        = data.get("timestamp", "")

    tracked_count = len(tracking_details) if isinstance(tracking_details, list) else 0

    # Bottom bar background (90px tall, sits at the very bottom)
    bar_top = height - 90
    cv2.rectangle(img, (0, bar_top), (width, height), (20, 20, 20), -1)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Row 1 — camera + time  (bar_top + 28)
    cv2.putText(img, f"Camera: {camera_id}",
                (10, bar_top + 28), font, 0.7, COLOR_WHITE, 2)
    cv2.putText(img, f"Time: {timestamp[:19]}",
                (200, bar_top + 28), font, 0.6, COLOR_GRAY, 1)

    # Row 2 — counts  (bar_top + 62)
    cv2.putText(img, f"Detected: {num_persons}",
                (10, bar_top + 62), font, 0.7, COLOR_CYAN, 2)
    cv2.putText(img, f"Tracked: {tracked_count}",
                (200, bar_top + 62), font, 0.7, COLOR_GREEN, 2)

    exceeded_color = COLOR_RED if exceeded_count > 0 else COLOR_WHITE
    threshold_label = f"{int(WAITING_TIME_THRESHOLD // 60)}m{int(WAITING_TIME_THRESHOLD % 60):02d}s"
    cv2.putText(img, f"Exceeded {threshold_label}: {exceeded_count}",
                (380, bar_top + 62), font, 0.7, exceeded_color, 2)

    # Legend — sits just above the bottom bar (bottom-right corner)
    legend_items = [
        (COLOR_GREEN,  f"< 50% ({int(WAITING_TIME_THRESHOLD * 0.5)}s)"),
        (COLOR_YELLOW, "50–75%"),
        (COLOR_ORANGE, "75–100%"),
        (COLOR_RED,    "Exceeded!"),
    ]
    lx = width - 200
    ly = bar_top - (len(legend_items) * 25) - 10
    for i, (col, text) in enumerate(legend_items):
        y = ly + i * 25
        cv2.rectangle(img, (lx, y), (lx + 16, y + 16), col, -1)
        cv2.putText(img, text, (lx + 22, y + 13), font, 0.5, COLOR_WHITE, 1)


# ─────────────────────────────────────────────────────────────
# Main visualizer loop
# ─────────────────────────────────────────────────────────────

def visualize():
    threshold_label = f"{int(WAITING_TIME_THRESHOLD // 60)}m{int(WAITING_TIME_THRESHOLD % 60):02d}s"
    print(f"Customer Path Tracker Visualizer | Queue: {QUEUE_NAME}")
    print(f"Waiting threshold: {threshold_label} | Press 'q' to quit")

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
                print("[Warning] No frame available in payload")
                continue

            # ── Draw duration overlays on tracked persons ────
            # (path trails are already baked into the frame by the service)
            tracking_details = data.get("tracking_details", [])
            if isinstance(tracking_details, list) and len(tracking_details) > 0:
                draw_tracked_persons(frame, tracking_details)

            # ── Draw HUD dashboard ───────────────────────────
            draw_dashboard(frame, data)

            # ── Show ─────────────────────────────────────────
            cv2.imshow("Customer Path Tracker", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            try:
                if cv2.getWindowProperty("Customer Path Tracker", cv2.WND_PROP_AUTOSIZE) < 0:
                    break
            except cv2.error:
                break

        except json.JSONDecodeError as e:
            print(f"[Error] Failed to decode JSON payload: {e}")
        except Exception as e:
            print(f"[Visualizer Error] {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    visualize()