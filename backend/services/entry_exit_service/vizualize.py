import redis
import json
import base64
import cv2
import numpy as np
import time
import os
from collections import defaultdict

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
CAMERA_ID = os.getenv("CAMERA_ID", "entrance")
QUEUE_NAME = os.getenv("VIS_QUEUE_NAME", f"stream:visualization:{CAMERA_ID}")

# Age/Gender class mapping: 0: child, 1: adult-male, 2: adult-female
AGE_GENDER_DICT = {0: "child", 1: "adult_male", 2: "adult_female"}

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def decode_frame(frame_b64: str) -> np.ndarray:
    """Decode base64 encoded frame to numpy array."""
    img_bytes = base64.b64decode(frame_b64)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def visualize():
    """
    Visualize entry/exit service frames with gender/age demographics.
    
    Note: The frames received already have visualization drawn on them
    (regions, bounding boxes, track IDs, paths) by the service.
    This script simply displays the pre-annotated frames and adds
    additional metadata overlay if needed.
    """
    print(f"Redis frame visualizer started for queue: {QUEUE_NAME}")
    print(f"Entry/Exit Service - Camera ID: {CAMERA_ID}")
    print("Press 'q' to quit")
    
    window_initialized = False
    while True:
        try:
            if window_initialized:
                 if cv2.waitKey(1) & 0xFF == ord("q"):
                     break
                 
                 if cv2.getWindowProperty("Entry/Exit Detection Visualization", cv2.WND_PROP_VISIBLE) < 1:
                    break

            msg = r.blpop(QUEUE_NAME, timeout=0.1)
            if msg is None:
                continue

            _, payload = msg
            data = json.loads(payload)

            # Get frame (already has visualization drawn)
            frame = None
            if "frame" in data:
                frame = decode_frame(data["frame"])
            
            if frame is None:
                print("[Warning] No frame available in message")
                continue

            # Get metadata
            camera_id = data.get("camera_id", "unknown")
            timestamp = data.get("timestamp", "")
            frame_num = data.get("frame_num", 0)
            tracks = data.get("tracks", [])
            num_tracks = data.get("num_tracks", 0)
            
            # Count entries and exits
            entry_count = sum(1 for t in tracks if t.get("entered", False))
            exit_count = sum(1 for t in tracks if t.get("exited", False))
            
            # Count demographics for entered tracks
            demographics = defaultdict(lambda: defaultdict(int))
            for t in tracks:
                if t.get("entered", False):
                    gender = t.get("gender") or "unknown"
                    age_group = t.get("age_group") or "unknown"
                    demographics[gender][age_group] += 1
            
            # Print track demographics to console
            if tracks:
                print(f"\n--- Frame {frame_num} ---")
                for t in tracks:
                    track_id = t.get("track_id", "?")
                    entered = t.get("entered", False)
                    exited = t.get("exited", False)
                    gender = t.get("gender")
                    age_group = t.get("age_group")
                    cls = t.get("cls")
                    
                    # Get human-readable label from class
                    cls_label = AGE_GENDER_DICT.get(cls, str(cls)) if cls is not None else "?"
                    
                    status = ""
                    if entered:
                        status = "[ENTRY]"
                    elif exited:
                        status = "[EXIT]"
                    
                    demo_str = ""
                    if gender or age_group:
                        demo_str = f" | {gender or '?'}/{age_group or '?'}"
                    elif cls is not None:
                        demo_str = f" | {cls_label}"
                    
                    if status or demo_str:
                        print(f"  Track {track_id}: {status}{demo_str}")
            
            # Add additional overlay at bottom of frame
            height, width = frame.shape[:2]
            
            # Draw semi-transparent background for bottom info bar (expanded for demographics)
            overlay = frame.copy()
            bar_height = 110 if demographics else 80
            cv2.rectangle(overlay, (0, height - bar_height), (width, height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # Draw info text
            y_offset = height - bar_height + 25
            cv2.putText(
                frame,
                f"Camera: {camera_id} | Frame: {frame_num} | Time: {timestamp}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            
            y_offset += 25
            cv2.putText(
                frame,
                f"Active Tracks: {num_tracks}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
            
            # Entry count (green)
            cv2.putText(
                frame,
                f"ENTRIES: {entry_count}",
                (width - 280, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            
            # Exit count (red)
            cv2.putText(
                frame,
                f"EXITS: {exit_count}",
                (width - 130, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            
            # Draw demographics breakdown if available
            if demographics:
                y_offset += 25
                demo_parts = []
                
                # Male demographics
                if "male" in demographics:
                    male_parts = [f"{age}:{cnt}" for age, cnt in demographics["male"].items()]
                    demo_parts.append(f"M[{', '.join(male_parts)}]")
                
                # Female demographics
                if "female" in demographics:
                    female_parts = [f"{age}:{cnt}" for age, cnt in demographics["female"].items()]
                    demo_parts.append(f"F[{', '.join(female_parts)}]")
                
                # Unknown demographics
                if "unknown" in demographics:
                    unk_parts = [f"{age}:{cnt}" for age, cnt in demographics["unknown"].items()]
                    demo_parts.append(f"?[{', '.join(unk_parts)}]")
                
                demo_text = "Demographics: " + " | ".join(demo_parts) if demo_parts else ""
                
                if demo_text:
                    cv2.putText(
                        frame,
                        demo_text,
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 200, 100),  # Light orange
                        1,
                    )

            cv2.imshow("Entry/Exit Detection Visualization", frame)
            window_initialized = True

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

                                    # Check if window was closed by user
            if cv2.getWindowProperty("Entry/Exit Detection Visualization", cv2.WND_PROP_VISIBLE) < 1:
                break



        except Exception as e:
            print(f"[Visualizer Error] {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    visualize()
