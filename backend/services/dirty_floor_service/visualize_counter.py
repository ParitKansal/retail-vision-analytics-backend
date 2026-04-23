import redis
import json
import cv2
import numpy as np
import time
import os
import requests
from dotenv import load_dotenv
import base64

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
# Specific to Counter (Cam 1)
INPUT_QUEUES = ["stream:cam:1"]
VIS_QUEUES = [f"stream:visualization:{q.split(':')[-1]}" for q in INPUT_QUEUES if q]

# Layout constants
FRAME_H = 640          # display height for the frame
PANEL_W = 340          # width of the info panel on the right
LINE_H = 36            # line height for text rows
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SM = cv2.FONT_HERSHEY_PLAIN

# Colours (BGR)
BG_DARK   = (30, 30, 30)
GREEN     = (0, 200, 0)
RED       = (0, 0, 220)
WHITE     = (255, 255, 255)
GREY      = (160, 160, 160)
CYAN      = (255, 255, 0)
BAR_BG    = (60, 60, 60)


def decode_frame(frame_b64: str) -> np.ndarray:
    """Decode base64 encoded frame to numpy array (BGR)."""
    img_bytes = base64.b64decode(frame_b64)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def resize_keep_aspect(img, target_h):
    """Resize image to target height, maintaining aspect ratio."""
    h, w = img.shape[:2]
    scale = target_h / h
    new_w = int(w * scale)
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)


def draw_masks_on_frame(frame, masks):
    """
    Draw mask polygons on the frame:
    - Semi-transparent black fill (shows which areas are masked out for the model)
    - Cyan outline so the mask boundaries are clearly visible
    """
    if not masks:
        return
    for mask in masks:
        pts = np.array(mask, dtype=np.int32)
        # Semi-transparent black fill
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 0, 0))
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        # Cyan outline
        cv2.polylines(frame, [pts], isClosed=True, color=CYAN, thickness=2)


def apply_status_tint(frame, status):
    """Apply a semi-transparent colour tint over the whole frame."""
    overlay = frame.copy()
    if status == "Dirty":
        overlay[:] = (0, 0, 200)
    else:
        overlay[:] = (0, 180, 0)
    return cv2.addWeighted(overlay, 0.18, frame, 0.82, 0)


def draw_info_panel(panel_h, panel_w, data):
    """Draw a dark info panel with all the stats."""
    panel = np.full((panel_h, panel_w, 3), BG_DARK, dtype=np.uint8)
    y = 20

    camera_id   = data.get("camera_id", "?")
    timestamp   = data.get("timestamp", "")
    prediction  = data.get("prediction", "—")
    status      = data.get("status", "Clean")
    dirty_count = data.get("dirty_count", 0)
    buffer_len  = data.get("buffer_len", 0)
    dirty_pct   = data.get("dirty_percent", 0.0)

    def put(label, value, val_color=WHITE):
        nonlocal y
        cv2.putText(panel, label, (16, y), FONT, 0.50, GREY, 1, cv2.LINE_AA)
        y += LINE_H - 10
        cv2.putText(panel, str(value), (16, y), FONT, 0.70, val_color, 2, cv2.LINE_AA)
        y += LINE_H + 4

    # ── Title ──
    cv2.putText(panel, "DIRTY FLOOR MONITOR", (16, y), FONT, 0.60, CYAN, 2, cv2.LINE_AA)
    y += LINE_H + 8
    cv2.line(panel, (16, y), (panel_w - 16, y), GREY, 1)
    y += 16

    put("Camera", f"Cam {camera_id}")
    put("Timestamp", str(timestamp))

    # ── Classification ──
    cv2.line(panel, (16, y), (panel_w - 16, y), GREY, 1)
    y += 16
    pred_color = RED if prediction == "Dirty" else GREEN
    put("Current Classification", prediction, pred_color)

    # ── Status ──
    status_color = RED if status == "Dirty" else GREEN
    put("Overall Status", status, status_color)

    # ── Buffer ──
    cv2.line(panel, (16, y), (panel_w - 16, y), GREY, 1)
    y += 16
    cv2.putText(panel, "Buffer", (16, y), FONT, 0.50, GREY, 1, cv2.LINE_AA)
    y += LINE_H - 4

    # Numbers row
    cv2.putText(panel, f"{dirty_count} / {buffer_len} dirty", (16, y), FONT, 0.65, WHITE, 2, cv2.LINE_AA)
    y += LINE_H

    # Percentage
    pct_color = RED if dirty_pct >= 90 else (0, 180, 255) if dirty_pct >= 50 else GREEN
    cv2.putText(panel, f"{dirty_pct:.1f}%", (16, y), FONT, 0.9, pct_color, 2, cv2.LINE_AA)
    y += LINE_H + 4

    # Progress bar
    bar_x, bar_y = 16, y
    bar_w = panel_w - 32
    bar_h = 22
    cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), BAR_BG, -1)
    fill_w = int(bar_w * (dirty_pct / 100.0))
    if fill_w > 0:
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), pct_color, -1)
    cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), GREY, 1)
    y += bar_h + 12

    # Threshold marker
    cv2.putText(panel, "Threshold: 90%", (16, y), FONT_SM, 1.1, GREY, 1, cv2.LINE_AA)
    thresh_x = bar_x + int(bar_w * 0.9)
    cv2.line(panel, (thresh_x, bar_y - 4), (thresh_x, bar_y + bar_h + 4), WHITE, 2)

    return panel


def main():
    print(f"Connecting to Redis at {REDIS_HOST}:{REDIS_PORT}")
    print(f"Listening on queues: {VIS_QUEUES}")

    r = None
    window_name = "Dirty Floor Monitor"
    window_created = False

    while True:
        try:
            # ── Redis connection ──
            if r is None:
                try:
                    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
                    r.ping()
                    print("Connected to Redis.")
                except Exception as e:
                    print(f"Failed to connect to Redis: {e}")
                    time.sleep(2)
                    continue

            # ── Pop a message ──
            try:
                msg = r.blpop(VIS_QUEUES, timeout=1.0)
            except (redis.exceptions.ConnectionError, ConnectionResetError) as e:
                print(f"Connection lost: {e}. Reconnecting...")
                r = None
                time.sleep(1)
                continue

            if msg is None:
                if window_created:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    try:
                        if cv2.getWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE) < 0:
                            break
                    except cv2.error:
                        break
                continue

            # ── Parse payload ──
            _, payload = msg
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                print("Failed to decode JSON payload")
                continue

            # ── Decode frame ──
            frame = None
            if "frame" in data:
                try:
                    frame = decode_frame(data["frame"])
                except Exception as e:
                    print(f"Failed to decode frame: {e}")

            if frame is None:
                continue

            status     = data.get("status", "Clean")
            prediction = data.get("prediction", "—")
            masks      = data.get("masks", [])

            # ── 1. Draw mask polygons on the frame ──
            draw_masks_on_frame(frame, masks)

            # ── 2. Apply status colour tint ──
            frame = apply_status_tint(frame, status)

            # ── 3. Resize to display height ──
            frame = resize_keep_aspect(frame, FRAME_H)
            frame_h, frame_w = frame.shape[:2]

            # ── 4. Build info panel (same height as frame) ──
            info_panel = draw_info_panel(frame_h, PANEL_W, data)

            # ── 5. Combine side-by-side ──
            canvas = np.hstack([frame, info_panel])

            # ── Show ──
            if not window_created:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, canvas.shape[1], canvas.shape[0])
                window_created = True

            cv2.imshow(window_name, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE) < 0:
                    break
            except cv2.error:
                break

        except Exception as e:
            print(f"Error in loop: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
