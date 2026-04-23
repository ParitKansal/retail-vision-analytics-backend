"""
Alert Viewer — reads from the OUTPUT_QUEUE stream and prints alerts to console.
Usage: python view_alerts.py
"""
import redis
import json
import os
import base64
import cv2
import numpy as np
from datetime import datetime

REDIS_HOST   = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT   = int(os.getenv("REDIS_PORT", 6379))
OUTPUT_QUEUE = os.getenv("OUTPUT_QUEUE_NAME_1")   # must match service .env
SHOW_FRAMES  = os.getenv("SHOW_FRAMES", "false").lower() == "true"  # set true to pop up alert images

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ── ANSI colors for terminal ──────────────────────────────────
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def show_frame(frame_b64: str, title: str):
    """Decode base64 frame and show in an OpenCV window."""
    try:
        img_bytes = base64.b64decode(frame_b64)
        np_arr    = np.frombuffer(img_bytes, np.uint8)
        img       = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is not None:
            cv2.imshow(title, img)
            cv2.waitKey(3000)   # show for 3 seconds
            cv2.destroyWindow(title)
    except Exception as e:
        print(f"  [Could not show frame: {e}]")


def print_alert(data: dict, msg_id: str):
    alert_type   = data.get("type", "unknown").upper()
    camera_id    = data.get("camera_id", "?")
    timestamp    = data.get("timestamp", "?")
    description  = data.get("description", "")
    ticket_id    = data.get("ticket_id", "?")
    collection   = data.get("target_collection", "")

    color = RED if alert_type == "RAISED" else GREEN

    print(f"\n{'─' * 60}")
    print(f"{BOLD}{color}[{alert_type}]{RESET}  {BOLD}{description}{RESET}")
    print(f"  Camera    : {CYAN}{camera_id}{RESET}")
    print(f"  Time      : {timestamp}")
    print(f"  Ticket ID : {YELLOW}{ticket_id}{RESET}")
    print(f"  Collection: {collection}")
    print(f"  Stream ID : {msg_id}")

    if SHOW_FRAMES and "frame" in data:
        title = f"{alert_type} — Camera {camera_id} — {ticket_id[:8]}"
        show_frame(data["frame"], title)


def main():
    if not OUTPUT_QUEUE:
        print("ERROR: OUTPUT_QUEUE_NAME_1 env var not set.")
        return

    print(f"{BOLD}Alert Viewer{RESET} | Queue: {CYAN}{OUTPUT_QUEUE}{RESET}")
    print(f"Tip: set SHOW_FRAMES=true to pop up alert images")
    print(f"Listening for alerts... (Ctrl+C to quit)\n")

    # Start reading from the beginning of the stream so you can see
    # historical alerts too. Change '$' to '0' to read from the very start.
    last_id = "0"  # '0' = from beginning, '$' = only new alerts

    raised_count   = 0
    resolved_count = 0

    while True:
        try:
            messages = r.xread({OUTPUT_QUEUE: last_id}, count=10, block=2000)
            if not messages:
                continue

            for stream_name, stream_messages in messages:
                for msg_id, msg_data in stream_messages:
                    last_id = msg_id  # advance cursor

                    raw = msg_data.get("data")
                    if not raw:
                        continue

                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        print(f"[Warning] Could not parse message {msg_id}")
                        continue

                    # Only show alert_collection entries (not regular output payloads)
                    if data.get("target_collection") != "alert_collection":
                        continue

                    print_alert(data, msg_id)

                    if data.get("type") == "raised":
                        raised_count += 1
                    elif data.get("type") == "resolved":
                        resolved_count += 1

                    print(f"  [Total: {RED}{raised_count} raised{RESET} | "
                          f"{GREEN}{resolved_count} resolved{RESET}]")

        except KeyboardInterrupt:
            print(f"\n\nSummary: {raised_count} raised, {resolved_count} resolved")
            break
        except redis.exceptions.ConnectionError as e:
            print(f"[Redis error] {e} — retrying in 3s...")
            import time; time.sleep(3)
        except Exception as e:
            print(f"[Error] {e}")


if __name__ == "__main__":
    main()