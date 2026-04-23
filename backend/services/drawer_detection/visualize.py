import cv2
import redis
import json
import base64
import numpy as np
import os
import time
from dotenv import load_dotenv

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
CAMERA_ID = os.getenv("CAMERA_ID", "1")
VIS_QUEUE = f"stream:visualization:drawer:{CAMERA_ID}"

WINDOW_NAME = f"Drawer Detection - Camera {CAMERA_ID}"


def decode_frame(frame_b64: str) -> np.ndarray:
    """Decode base64 encoded frame to numpy array (BGR)."""
    img_bytes = base64.b64decode(frame_b64)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def run_visualization():
    print(f"Connecting to Redis at {REDIS_HOST}:{REDIS_PORT}...")
    print(f"Listening on queue: {VIS_QUEUE}")

    r = None
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
                msg = r.blpop(VIS_QUEUE, timeout=1)
            except (redis.exceptions.ConnectionError, ConnectionResetError) as e:
                print(f"Connection lost: {e}. Reconnecting...")
                r = None
                time.sleep(1)
                continue

            if msg is None:
                # No data yet — keep the window responsive
                if window_created:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    try:
                        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_AUTOSIZE) < 0:
                            break
                    except cv2.error:
                        break
                continue

            _, payload = msg
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                print("Failed to decode JSON payload")
                continue

            frame_b64 = data.get("frame")
            if not frame_b64:
                print("[Warning] No frame in payload")
                continue

            try:
                frame = decode_frame(frame_b64)
            except Exception as e:
                print(f"Failed to decode frame: {e}")
                continue

            if frame is None:
                continue

            # ── Show the frame ──
            if not window_created:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW_NAME, frame.shape[1], frame.shape[0])
                window_created = True

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_AUTOSIZE) < 0:
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
    run_visualization()
