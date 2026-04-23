# import redis
# import json
# import base64
# import cv2
# import numpy as np
# import time
# import os

# # Configuration
# REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
# REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
# CAMERA_ID = os.getenv("CAMERA_ID", "smilometer")
# QUEUE_NAME = os.getenv("VIS_QUEUE_NAME", f"stream:visualization:{CAMERA_ID}")

# def connect_redis():
#     """Connect to Redis with retry logic."""
#     while True:
#         try:
#             r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
#             r.ping()
#             print(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
#             return r
#         except redis.ConnectionError:
#             print("Failed to connect to Redis. Retrying in 5 seconds...")
#             time.sleep(5)

# def decode_frame(frame_b64: str) -> np.ndarray:
#     """Decode base64 encoded frame to numpy array."""
#     try:
#         img_bytes = base64.b64decode(frame_b64)
#         np_arr = np.frombuffer(img_bytes, np.uint8)
#         return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#     except Exception as e:
#         print(f"Error decoding frame: {e}")
#         return None

# def visualize():
#     """
#     Visualize exit emotion detection service frames.
#     """
#     r = connect_redis()
    
#     print(f"Redis frame visualizer started for queue: {QUEUE_NAME}")
#     print(f"Exit Emotion Service - Camera ID: {CAMERA_ID}")
#     print("Press 'q' to quit")
    
#     window_initialized = False
#     local_frame_count = 0
    
#     while True:
#         try:
#             # Handle Window Events
#             if window_initialized:
#                 key = cv2.waitKey(1) & 0xFF
#                 if key == ord("q"):
#                     break
                
#                 try:
#                     if cv2.getWindowProperty("Exit Emotion Detection Visualization", cv2.WND_PROP_VISIBLE) < 1:
#                         break
#                 except cv2.error:
#                     pass
            
#             # Fetch from Redis
#             msg = r.blpop(QUEUE_NAME, timeout=1.0)
#             if msg is None:
#                 continue

#             _, payload = msg
#             try:
#                 data = json.loads(payload)
#             except json.JSONDecodeError:
#                 print("Failed to decode JSON payload")
#                 continue

#             # Get frame
#             frame = None
#             if "frame" in data:
#                 frame = decode_frame(data["frame"])
            
#             if frame is None:
#                 print("[Warning] No frame available in message or decode failed")
#                 continue

#             # Get metadata
#             camera_id = data.get("camera_id", "unknown")
#             timestamp = data.get("timestamp", "")
            
#             # Use service frame num if available, else local counter
#             service_frame_num = data.get("frame_num", 0)
#             if service_frame_num > 0:
#                 display_frame_num = service_frame_num
#             else:
#                 local_frame_count += 1
#                 display_frame_num = local_frame_count

#             active_tracks = data.get("active_tracks", 0)
#             processed_count = data.get("processed_count", 0)
#             buffer_info = data.get("buffer_info", [])
            
#             height, width = frame.shape[:2]
            
#             # Draw semi-transparent background for bottom info bar
#             overlay = frame.copy()
#             cv2.rectangle(overlay, (0, height - 80), (width, height), (0, 0, 0), -1)
#             cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
#             # Draw info text
#             y_offset = height - 55
#             cv2.putText(
#                 frame,
#                 f"Camera: {camera_id} | Frame: {display_frame_num} | Time: {timestamp}",
#                 (10, y_offset),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.6,
#                 (255, 255, 255),
#                 2,
#             )
            
#             y_offset += 25
#             cv2.putText(
#                 frame,
#                 f"Active Tracks: {active_tracks} | Processed: {processed_count}",
#                 (10, y_offset),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.6,
#                 (0, 255, 255),
#                 2,
#             )
            
#             # Draw emotion legend on the right side
#             # Count emotions in buffer for quick stats
#             emotion_counts = {}
#             for info in buffer_info:
#                 emotion = info.get("latest_emotion", "Unknown")
#                 emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
#             if emotion_counts:
#                 # Legend background
#                 legend_text = " | ".join([f"{e}: {c}" for e, c in emotion_counts.items()])
#                 text_size = cv2.getTextSize(legend_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
#                 legend_x = width - text_size[0] - 20
#                 legend_y = height - 55
                
#                 cv2.putText(
#                     frame,
#                     legend_text,
#                     (legend_x, legend_y),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.5,
#                     (255, 255, 255),
#                     1,
#                 )

#             cv2.imshow("Exit Emotion Detection Visualization", frame)
#             window_initialized = True

#         except redis.ConnectionError:
#             print("Redis connection lost. Reconnecting...")
#             r = connect_redis()
#         except Exception as e:
#             print(f"[Visualizer Error] {e}")
#             import traceback
#             traceback.print_exc()
#             time.sleep(1)

#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     visualize()

import redis
import json
import base64
import cv2
import numpy as np
import time
import os

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
CAMERA_ID = os.getenv("CAMERA_ID", "6")
QUEUE_NAME = os.getenv("VIS_QUEUE_NAME", f"stream:visualization:exit_emotion:{CAMERA_ID}")

WINDOW_NAME = "Exit Emotion Detection Visualization"

def connect_redis():
    """Connect to Redis with retry logic."""
    while True:
        try:
            r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
            r.ping()
            print(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
            return r
        except redis.ConnectionError:
            print("Failed to connect to Redis. Retrying in 5 seconds...")
            time.sleep(5)

def decode_frame(frame_b64: str) -> np.ndarray:
    """Decode base64 encoded frame to numpy array."""
    try:
        img_bytes = base64.b64decode(frame_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error decoding frame: {e}")
        return None

def visualize():
    """
    Visualize exit emotion detection service frames.
    """
    r = connect_redis()

    print(f"Redis frame visualizer started for queue: {QUEUE_NAME}")
    print(f"Exit Emotion Service - Camera ID: {CAMERA_ID}")
    print("Press 'q' to quit")

    # Initialize window upfront so it's ready before first frame
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    window_initialized = True

    local_frame_count = 0

    while True:
        try:
            # Check for quit key or window close
            if window_initialized:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                try:
                    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break

            # Fetch from Redis
            msg = r.blpop(QUEUE_NAME, timeout=1.0)
            if msg is None:
                print("[Waiting] No frame in queue...")
                continue

            _, payload = msg

            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                print("Failed to decode JSON payload")
                continue

            # Get frame
            frame = None
            if "frame" in data:
                frame = decode_frame(data["frame"])

            if frame is None:
                print("[Warning] No frame available in message or decode failed")
                continue

            # Get metadata
            camera_id = data.get("camera_id", "unknown")
            timestamp = data.get("timestamp", "")

            # Use service frame num if available, else local counter
            service_frame_num = data.get("frame_num", 0)
            if service_frame_num > 0:
                display_frame_num = service_frame_num
            else:
                local_frame_count += 1
                display_frame_num = local_frame_count

            # active_tracks = data.get("active_tracks", 0)
            processed_count = data.get("processed_count", 0)
            buffer_info = data.get("buffer_info", [])

            height, width = frame.shape[:2]

            # Draw semi-transparent background for bottom info bar
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, height - 80), (width, height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            # Draw info text
            y_offset = height - 55
            cv2.putText(
                frame,
                f"Camera: {camera_id} | Frame: {display_frame_num} | Time: {timestamp}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            # y_offset += 25
            # cv2.putText(
            #     frame,
            #     # f"Active Tracks: {active_tracks} | Processed: {processed_count}",
            #     (10, y_offset),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.6,
            #     (0, 255, 255),
            #     2,
            # )

            # Count emotions in buffer for quick stats
            emotion_counts = {}
            for info in buffer_info:
                emotion = info.get("latest_emotion", "Unknown")
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

            if emotion_counts:
                legend_text = " | ".join([f"{e}: {c}" for e, c in emotion_counts.items()])
                text_size = cv2.getTextSize(legend_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                legend_x = width - text_size[0] - 20
                legend_y = height - 55

                cv2.putText(
                    frame,
                    legend_text,
                    (legend_x, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

            # Show frame and flush to screen immediately
            cv2.imshow(WINDOW_NAME, frame)
            # waitKey after imshow is critical to actually render the frame
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        except redis.ConnectionError:
            print("Redis connection lost. Reconnecting...")
            r = connect_redis()
        except Exception as e:
            print(f"[Visualizer Error] {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    visualize()