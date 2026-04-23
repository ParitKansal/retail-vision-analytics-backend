
import redis
import json
import base64
import cv2
import numpy as np
import time
import os
from collections import deque

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
CAMERA_ID = os.getenv("CAMERA_ID", "6")
QUEUE_NAME = os.getenv("OUTPUT_QUEUE_NAME", "store_status_output")

def connect_redis():
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        r.ping()
        print("Connected to Redis successfully.")
        return r
    except redis.ConnectionError:
        print(f"Could not connect to Redis at {REDIS_HOST}:{REDIS_PORT}")
        return None

def decode_frame(frame_b64: str) -> np.ndarray:
    """Decode base64 encoded frame to numpy array."""
    try:
        img_bytes = base64.b64decode(frame_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Failed to decode frame: {e}")
        return None

def visualize():
    r = connect_redis()
    if not r:
        return

    # Create consumer group if not exists
    group_name = "viz_group"
    consumer_name = "viz_worker"
    try:
        r.xgroup_create(QUEUE_NAME, group_name, id="0", mkstream=True)
    except redis.ResponseError as e:
        if "BUSYGROUP Consumer Group name already exists" not in str(e):
            print(f"Error creating group: {e}")

    print(f"Visualizer started for queue: {QUEUE_NAME}")
    print("Press 'q' to quit")

    # Load polygon regions if possible to draw them
    # Since we don't have direct access to service memory, we might need to load from file
    # or rely on the service sending region details. The service sends "region_details", 
    # but not the polygon coordinates themselves in the output payload usually.
    # However, for this visualization script to be standalone and run locally,
    # we should try to load the 'lights_region.json' if available.
    
    polygons = []
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(script_dir, "lights_region.json")
        with open(json_path, 'r') as f:
            regions_data = json.load(f)
            for region in regions_data:
                points = region['points']
                pts = np.array([[int(p['x']), int(p['y'])] for p in points], np.int32)
                pts = pts.reshape((-1, 1, 2))
                polygons.append(pts)
        print(f"Loaded {len(polygons)} polygons from file.")
    except Exception as e:
        print(f"Could not load polygons from file: {e}")
        # We will try to download the image from the URL in the message payload if available?
        # The output payload has 'frame_url' only in the *input* message, 
        # but the *output* payload from service usually has analysis results.
        # Wait, the service output payload:
        # {
        #     "camera_id": CAMERA_ID,
        #     "timestamp": timestamp,
        #     "status": current_status,
        #     "stable_status": stable_status,
        #     "store_closing_status": percentage_closed,
        #     "lights_on_percentage": lights_on_percentage,
        #     "region_details": region_details, 
        #     ...
        # }
        # It DOES NOT contain the image frame itself!
        # The alerting payload DOES contain the frame.
        # To visualize continuously, we need the frame.
        # Options:
        # 1. Read from INPUT_QUEUE as well to get the frame (but we need to sync with output).
        # 2. Modify service to include frame in output (High bandwidth).
        # 3. Use the 'frame_url' from the input message if available in some way?
        # 
        # Let's assume for this "visualize.py" intended for debugging the logic/status,
        # we might want to tap into the INPUT stream to get the image, 
        # and match it with the processed result if possible, OR
        # just visualize the Alert queue?
        # 
        # The prompt says: "make a visualize.py file ... for the outside time I want the frame to be shown..."
        # This implies continuous monitoring.
        # 
        # Let's try to read from the INPUT stream (stream:cam:6) to get the image,
        # AND read from the OUTPUT stream to get the status?
        # Or simpler: The user might be okay with us just running a simulation 
        # or acting as a consumer of the input stream that DUPLICATES the logic for visualization?
        # 
        # Actually, if we look at other services, 'visualize.py' often reads from a specific 
        # visualization queue or the input queue directly.
        pass

    # RE-EVALUATION:
    # The 'store_opening_closing_service' reads from 'stream:cam:6'.
    # It publishes results to 'store_status_output'.
    # The 'store_status_output' does NOT have the image.
    # We should probably read from 'stream:cam:6' (the input) to get the image,
    # and maybe overlay the logic LOCALLY for visualization purposes?
    # OR, we can't reliably sync the output from Redis without ID matching.
    # 
    # Let's implement a visualizer that reads from 'stream:cam:6' (Input)
    # AND performs the same region check logic locally to visualize it?
    # This ensures we see what the service sees.
    #
    # Wait, the user asked to "highlight the region that is on".
    # So we need the polygon definitions.
    
    input_queue = os.getenv("INPUT_QUEUE_NAME", "stream:cam:6")
    import requests # Import here to ensure it's available

    # Initialize last_id
    last_id = "$"
    
    window_name = "Store Status Visualization"
    window_initialized = False

    # Try to fetch the very last message to show something immediately
    try:
        initial = r.xrevrange(input_queue, count=1)
        if initial:
            last_id = initial[0][0] # Set to the last ID so we continue from there
            # We want to process this initial message too
            # Synthesize a response format to reuse loop logic or just process it
            # To keep it simple, we'll process it in a special block before the loop or 
            # adjust the loop to handle it. 
            # Let's just process it now.
            _, msg_data = initial[0]
            
            # --- PROCESS MESSAGE (Copy of logic below) ---
            payload = {k: v for k, v in msg_data.items()}
            frame_url = payload.get("frame_url")
            img = None
            if frame_url:
                try:
                    # Fix for local MinIO access with presigned URLs
                    headers = {"Host": "minio:9000"}
                    if "minio:9000" in frame_url:
                        frame_url = frame_url.replace("minio:9000", "localhost:9000")
                    
                    res = requests.get(frame_url, headers=headers, timeout=5)
                    if res.status_code == 200:
                        arr = np.frombuffer(res.content, np.uint8)
                        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    else:
                        print(f"Failed to fetch initial image: {res.status_code}")
                except Exception:
                    pass
            
            if img is not None:
                # Drawing Logic
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                thresholds = [90000, 120000, 70000]
                total_regions = len(polygons)
                closed_regions_count = 0
                from datetime import datetime
                now = datetime.now()
                hour = now.hour
                is_opening = (7 <= hour < 9)
                is_closing = (0 <= hour < 2)
                lights_on_count = 0
                for i, poly in enumerate(polygons):
                    mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.fillPoly(mask, [poly], 255)
                    masked = cv2.bitwise_and(gray, gray, mask=mask)
                    val = np.sum(masked)
                    thresh = thresholds[i] if i < len(thresholds) else 100000
                    is_off = val < thresh
                    color = (0, 255, 0) if not is_off else (0, 0, 255)
                    cv2.polylines(img, [poly], True, color, 2)
                    if not is_off:
                        lights_on_count += 1
                    
                    # Display pixel value and threshold
                    text_pos = tuple(poly[0][0])
                    cv2.putText(img, f"V:{val} T:{thresh}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                percentage_lights_on = (lights_on_count / total_regions) * 100.0 if total_regions > 0 else 0
                border_color = (0, 0, 0) 
                if not (is_opening or is_closing):
                    border_color = (0, 0, 255) # Red
                    status_text = "UNTRACKED"
                else:
                    if percentage_lights_on == 100:
                        border_color = (0, 255, 0) 
                        status_text = "ON (100%)"
                    elif percentage_lights_on == 0:
                        border_color = (255, 0, 0) 
                        status_text = "OFF (0%)"
                    else:
                        border_color = (0, 165, 255)
                        status_text = f"PARTIAL ({percentage_lights_on:.1f}%)"
                
                cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), border_color, 20)
                cv2.putText(img, f"Time: {now.strftime('%H:%M:%S')} - {status_text}", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, border_color, 2)
                
                # Display Time Zones
                cv2.putText(img, "Opening: 07:00 - 09:00", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(img, "Closing: 00:00 - 02:00", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(img, "Untracked: Other times", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow(window_name, img)
                window_initialized = True
                cv2.waitKey(1)
            # --- END PROCESS MESSAGE ---

    except Exception as e:
        print(f"Correction Warning: Could not fetch initial frame: {e}")

    print(f"Listening for new messages on {input_queue} starting from ID: {last_id}")

    while True:
        try:
            # Read new messages starting from last_id
            resp = r.xread({input_queue: last_id}, count=1, block=100)
            if not resp:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            stream_name, messages = resp[0]
            msg_id, message = messages[0]
            last_id = msg_id # Important: update last_id
            
            payload = {k: v for k, v in message.items()}
            frame_url = payload.get("frame_url")
            
            # Download Image
            img = None
            if frame_url:
                try:
                    # Fix for local MinIO access with presigned URLs
                    headers = {"Host": "minio:9000"} 
                    if "minio:9000" in frame_url:
                        frame_url = frame_url.replace("minio:9000", "localhost:9000")
                    
                    res = requests.get(frame_url, headers=headers, timeout=5)
                    if res.status_code == 200:
                        arr = np.frombuffer(res.content, np.uint8)
                        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    else:
                        print(f"Failed to fetch image: {res.status_code}")
                except Exception as e:
                    pass

            if img is None:
                continue

            # Process logic locally for visualization
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresholds = [70000, 120000, 90000]
            
            total_regions = len(polygons)
            closed_regions_count = 0
            
            from datetime import datetime
            now = datetime.now()
            hour = now.hour
            minute = now.minute
            
            is_opening = (7 <= hour < 9)
            is_closing = (0 <= hour < 2)
            
            lights_on_count = 0
            
            for i, poly in enumerate(polygons):
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.fillPoly(mask, [poly], 255)
                masked = cv2.bitwise_and(gray, gray, mask=mask)
                val = np.sum(masked)
                
                thresh = thresholds[i] if i < len(thresholds) else 100000
                is_off = val < thresh
                
                color = (0, 255, 0) if not is_off else (0, 0, 255)
                cv2.polylines(img, [poly], True, color, 2)
                
                
                if not is_off:
                    lights_on_count += 1
                
                # Display pixel value and threshold
                text_pos = tuple(poly[0][0])
                cv2.putText(img, f"V:{val} T:{thresh}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            percentage_lights_on = (lights_on_count / total_regions) * 100.0 if total_regions > 0 else 0
            
            border_color = (0, 0, 0)
            
            if not (is_opening or is_closing):
                border_color = (0, 0, 255) # Red
                status_text = "UNTRACKED"
            else:
                if percentage_lights_on == 100:
                    border_color = (0, 255, 0) 
                    status_text = "ON (100%)"
                elif percentage_lights_on == 0:
                    border_color = (255, 0, 0) 
                    status_text = "OFF (0%)"
                else:
                    border_color = (0, 165, 255) 
                    status_text = f"PARTIAL ({percentage_lights_on:.1f}%)"

            cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), border_color, 20)
            
            cv2.putText(img, f"Time: {now.strftime('%H:%M:%S')} - {status_text}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, border_color, 2)
            
            # Display Time Zones
            cv2.putText(img, "Opening: 07:00 - 09:00", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, "Closing: 00:00 - 02:00", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, "Untracked: Other times", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Check if window was closed by user (before recreating it with imshow)
            if window_initialized and cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed by user.")
                break

            cv2.imshow(window_name, img)
            window_initialized = True
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        except Exception as e:
            print(f"Loop Error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    visualize()
