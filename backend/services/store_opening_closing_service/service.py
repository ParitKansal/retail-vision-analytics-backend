
import json
import numpy as np
import cv2
from dotenv import load_dotenv
import redis
import logging
import time
import requests
import logging_loki
from multiprocessing import Queue
import os
from collections import deque
import base64
import uuid
from datetime import datetime
from shapely.geometry import Polygon

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
LOKI_URL = os.getenv("LOKI_URL", "http://localhost:3100/loki/api/v1/push")

INPUT_QUEUE = os.getenv("INPUT_QUEUE_NAME", "stream:cam:6")
OUTPUT_QUEUE = os.getenv("OUTPUT_QUEUE_NAME", "store_status_output")
ALERT_QUEUE = os.getenv("ALERT_QUEUE_NAME", "database:queue:0")
GROUP_NAME = os.getenv("GROUP_NAME", "store_opening_closing_group")
CAMERA_ID = os.getenv("CAMERA_ID", "6")
TARGET_COLLECTION_NAME = os.getenv("TARGET_COLLECTION_NAME", "store_status_collection")
ALERT_COLLECTION_NAME = os.getenv("ALERT_COLLECTION_NAME", "alert_collection")
ACK_TTL_SECONDS = 3600  # 1 hour

# Configure Logging
handler = logging_loki.LokiQueueHandler(
    Queue(-1),
    url=LOKI_URL,
    tags={"application": "store_opening_closing_service"},
    version="1",
)

logger = logging.getLogger("store_opening_closing_service")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)
class StoreOpeningClosingService:
    def __init__(self):
        self.redis = self.connect_redis(host=REDIS_HOST, port=REDIS_PORT)
        self.polygons = self.load_lights_regions()

        # Buffering for stability
        self.history = deque(maxlen=60)
        self.last_alerted_status = None # "open" or "closed"
        
        # New: Buffer for images during transition (Opening 7-9)
        # Requirement: "during 7-9... keep the latest 5 frames"
        self.frame_buffer = deque(maxlen=5) 
        
        # Storage for transition states: "0%", "partial", "100%"
        self.transition_states = {}

        # Load state from Redis
        self.load_state()

    def mark_acked(self, stream: str, msg_id: str, consumer: str):
        cnt_key = f"ack:cnt:{stream}:{msg_id}"
        seen_key = f"ack:seen:{stream}:{msg_id}"

        pipe = self.redis.pipeline()

        # Ensure each consumer increments only once
        pipe.sadd(seen_key, consumer)
        pipe.expire(seen_key, ACK_TTL_SECONDS)
        results = pipe.execute()
        was_added = results[0]  # 1 if new, 0 if already present

        if was_added:
            pipe = self.redis.pipeline()
            pipe.incr(cnt_key)
            pipe.expire(cnt_key, ACK_TTL_SECONDS)
            pipe.execute()

    def load_state(self):
        try:
            state = self.redis.get(f"store_status:{CAMERA_ID}")
            if state:
                data = json.loads(state)
                self.last_alerted_status = data.get("status")
                self.current_ticket_id = data.get("ticket_id")
                logger.info(f"Loaded persistent state: {self.last_alerted_status}")
            else:
                self.current_ticket_id = None
                logger.info("No persistent state found.")
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            self.current_ticket_id = None

    def save_state(self, status, ticket_id):
        try:
            data = json.dumps({"status": status, "ticket_id": ticket_id})
            self.redis.set(f"store_status:{CAMERA_ID}", data)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def connect_redis(self, host, port, retry_delay=5):
        while True:
            try:
                r = redis.Redis(host=host, port=port, decode_responses=False)
                r.ping()
                logger.info("Connected to Redis successfully.")
                return r
            except Exception as e:
                logger.error(
                    f"Redis connection failed: {e}. Retrying in {retry_delay}s..."
                )
                time.sleep(retry_delay)

    def load_lights_regions(self):
        region_file = "lights_region.json"
        try:
            with open(region_file, 'r') as f:
                regions = json.load(f)
            
            polygons = []
            for region in regions:
                points = region['points']
                pts = np.array([[int(p['x']), int(p['y'])] for p in points], np.int32)
                pts = pts.reshape((-1, 1, 2))
                polygons.append(pts)
            logger.info(f"Loaded {len(polygons)} light regions.")
            return polygons
        except Exception as e:
            logger.error(f"Failed to load light regions: {e}")
            return []

    def process_message(self, message: dict):
        message = {k.decode(): v.decode() for k, v in message.items()}
        frame_url = message.get("frame_url")
        timestamp = message.get("timestamp")

        if not frame_url:
            return

        # 1. Time Check & Context
        # User requirement: Opening 7-9, Closing 12-2
        # Opening: 7:00 - 9:00 (Morning)
        # Closing: 12:00 - 2:00 (Assuming Midnight 00:00 - 02:00) based on typical store hours implied
        now = datetime.now()
        hour = now.hour
        minute = now.minute

        is_opening_time = (7 <= hour < 9)
        is_closing_time = (0 <= hour < 2) # Midnight to 2 AM

        if not (is_opening_time or is_closing_time):
            # Outside of monitoring hours
            return

        # 1. Download Image
        try:
            img_resp = requests.get(frame_url, timeout=10)
            img_resp.raise_for_status()
            img_array = np.frombuffer(img_resp.content, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image")
        except Exception as e:
            logger.error(f"Failed to download/decode frame {frame_url}: {e}")
            return

        # 2. Process Logic (Light Detection)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Define thresholds
        thresholds = [90000, 120000, 90000]
        
        closed_regions_count = 0
        total_regions = len(self.polygons)
        
        if total_regions != len(thresholds):
             logger.warning(f"Region count mismatch. Expected {len(thresholds)}, got {total_regions}. Using defaults.")

        region_details = []

        for i, poly in enumerate(self.polygons):
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [poly], 255)
            
            masked_img = cv2.bitwise_and(gray, gray, mask=mask)
            pixel_sum = np.sum(masked_img)
            
            threshold = thresholds[i] if i < len(thresholds) else 100000
            
            is_closed = bool(pixel_sum < threshold)
            if is_closed:
                closed_regions_count += 1
            
            region_details.append({
                "region_index": i,
                "pixel_sum": float(pixel_sum),
                "threshold": threshold,
                "is_closed": is_closed
            })

        # Calculate store_closing_status percentage (100% = Fully Closed, 0% = Fully Open)
        if total_regions > 0:
            percentage_closed = (closed_regions_count / total_regions) * 100.0
        else:
            percentage_closed = 0.0

        # Invert logic for "Opening" percentage? The prompt uses "0% (0>x>100) and 100%" 
        # referring to "lights on" status presumably.
        # If percentage_closed is 100 (all regions dark/closed), then lights_on_percentage is 0.
        # If percentage_closed is 0 (all regions bright/open), then lights_on_percentage is 100.
        lights_on_percentage = 100.0 - percentage_closed

        # Current Status for buffer
        # Logic: If fully closed (lights 0%), status is "closed".
        # However, for Opening (7-9), user wants alert only if "opens completely".
        
        if is_opening_time:
            # Strict logic for Opening: Only "open" if 100% lights on.
            # Allowing a small epsilon for float precision, though inputs are integers essentially.
            # If percentage_closed is 0.0 -> 100% Open.
            current_status = "open" if percentage_closed == 0.0 else "closed"
        else:
            # Standard/Closing logic: "closed" only if fully dark (100% closed).
            # "open" if any light is on (Partial or Full).
            # This ensures "Store Closed" alert fires only when fully closed.
            current_status = "closed" if (closed_regions_count == total_regions) else "open"
        
        # store previous image for transition capture
        # The buffer stores (timestamp, image_bytes, lights_on_percentage)
        self.frame_buffer.append({
            "timestamp": timestamp,
            "img_bytes": img_resp.content,
            "img": img, # storing decoded image for convenience
            "lights_on": lights_on_percentage
        })

        # Transition Capture Logic (Opening Only: 7-9)
        if is_opening_time:
            self.handle_opening_transition(lights_on_percentage)

        # 3. Buffer Update & Majority Vote
        self.history.append(current_status)
        
        closed_count = sum(1 for s in self.history if s == "closed")
        total_count = len(self.history)
        
        # 80% Buffer Rule
        stable_status = None
        if len(self.history) == self.history.maxlen:
            if (closed_count / total_count) > 0.8:
                stable_status = "closed"
            elif ((total_count - closed_count) / total_count) > 0.8:
                stable_status = "open"
        
        if stable_status:
            # 4. Alerting
            if self.last_alerted_status is None:
                # Initialization
                logger.info(f"Initial stable status: {stable_status}")
                self.last_alerted_status = stable_status
                self.save_state(stable_status, None)

            elif stable_status != self.last_alerted_status:
                logger.info(f"Status changed from {self.last_alerted_status} to {stable_status}")
                
                # Determine Alert Type based on Time and Event
                alert_type = self.determine_alert_type(stable_status, hour, minute)
                
                # If Opening, we might want to attach the transition images?
                # The user said "raise alert of early opening if..."
                # I will include the transition images in the alert payload if available
                
                new_ticket_id = str(uuid.uuid4())
                
                # Construct description
                description = f"Store is now {stable_status.upper()}"
                if alert_type:
                    description = f"{description}: {alert_type}"

                self.send_alert(stable_status, timestamp, img, img_resp.content, new_ticket_id, description, alert_type)
                
                # Update State
                self.last_alerted_status = stable_status
                self.current_ticket_id = new_ticket_id
                self.save_state(stable_status, new_ticket_id)
                
                # Reset transition states after alert
                self.transition_states = {}

        # 5. Publish Result
        output_payload = {
            "camera_id": CAMERA_ID,
            "timestamp": timestamp,
            "status": current_status,
            "stable_status": stable_status,
            "store_closing_status": percentage_closed, # keeping original name for compatibility
            "lights_on_percentage": lights_on_percentage,
            "region_details": region_details, 
            "service": "store_opening_closing_service",
            "target_collection": TARGET_COLLECTION_NAME,
        }
        
        try:
            self.redis.xadd(OUTPUT_QUEUE, {"data": json.dumps(output_payload)})
        except Exception as e:
            logger.error(f"Redis publish failed: {e}")

    def handle_opening_transition(self, current_lights_on):
        """
        Captures images for 0%, partial (0<x<100), and 100% states during opening.
        """
        # "default is lights off (0%)"
        # "if atleast one light turns on, we store the previous image and keep it in a dict"

        # Check for 0%
        if current_lights_on == 0:
            # We don't overwrite if we already have a 0% unless we want the *latest* 0% before opening?
            # "store the previous image" implies the one right before opening.
            # So we can keep updating '0%' until we see > 0.
            # But simpler: constantly store the latest 0% frame in a temporary var or just rely on buffer.
            pass # we handle this when we see > 0

        # Check for transition (0 -> >0)
        # We look at the buffer to find the previous frame
        if len(self.frame_buffer) >= 2:
            prev_frame = self.frame_buffer[-2]
            curr_frame = self.frame_buffer[-1]
            
            prev_lights = prev_frame['lights_on']
            curr_lights = curr_frame['lights_on']
            
            # If we just switched from 0 to > 0
            if prev_lights == 0 and curr_lights > 0:
                # Store the previous image (0%)
                if "0%" not in self.transition_states:
                    self.transition_states["0%"] = prev_frame

            # If current is Partial (0 < x < 100)
            if 0 < curr_lights < 100:
                if "partial" not in self.transition_states:
                    self.transition_states["partial"] = curr_frame
            
            # If current is 100%
            if curr_lights == 100:
                if "100%" not in self.transition_states:
                    self.transition_states["100%"] = curr_frame

    def determine_alert_type(self, status, hour, minute):
        time_min = hour * 60 + minute
        
        if status == "open":
            # Opening logic (7:00 - 9:00)
            # 7:00 = 420 min
            # 7:50 = 470 min
            # 8:15 = 495 min
            # 9:00 = 540 min
            
            if 420 <= time_min < 470:
                return "Early Opening"
            elif 470 <= time_min <= 495:
                return "Perfect Time Opening"
            elif 495 < time_min < 540:
                return "Late Opening"
                
        elif status == "closed":
            # Closing logic (12:00 - 2:00 AM implied)
            # 12:00 AM = 0 min
            # 12:50 AM = 50 min
            # 1:05 AM = 65 min
            # 2:00 AM = 120 min
            
            # Ensure we are in the 0-2 range context
            if 0 <= time_min < 50:
                return "Early Closing"
            elif 50 <= time_min <= 65:
                return "Perfect Time Closing"
            elif 65 < time_min < 120:
                return "Late Closing"
                
        return None

    def send_alert(self, status, timestamp, img, raw_img_bytes, ticket_id, description, alert_type):
        # ticket_id passed as argument
        
        # Encode Frame (Raw)
        frame_without_visualization = base64.b64encode(raw_img_bytes).decode('utf-8')
        
        # Encode Frame (Visualized - draw regions)
        vis_img = img.copy()
        cv2.polylines(vis_img, self.polygons, True, (0, 255, 255), 2)
              
        _, buffer = cv2.imencode('.jpg', vis_img)
        frame_vis = base64.b64encode(buffer).decode('utf-8')

        alert_payload = {
            "type": "store_status",
            "description": description,
            "alert_type": alert_type, # Add specific type
            "ticket_id": ticket_id,
            "camera_id": CAMERA_ID,
            "timestamp": timestamp,
            "frame": frame_vis,
            "frame_without_visualization": frame_without_visualization,
            "target_collection": ALERT_COLLECTION_NAME
        }
        
        # Include transition images if available and relevant (e.g. Opening)
        if self.transition_states:
            transition_data = {}
            for key, frame_data in self.transition_states.items():
                # Encode these images too
                _, buf = cv2.imencode('.jpg', frame_data['img'])
                b64 = base64.b64encode(buf).decode('utf-8')
                transition_data[key] = {
                    "timestamp": frame_data['timestamp'],
                    "image": b64
                }
            alert_payload["transition_images"] = transition_data
        
        try:
            self.redis.xadd(ALERT_QUEUE, {"data": json.dumps(alert_payload)})
            logger.info(f"Alert sent: {description}")
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

    def run(self):
        logger.info("StoreOpeningClosingService started.")
        try:
            self.redis.xgroup_create(INPUT_QUEUE, GROUP_NAME, id="0", mkstream=True)
        except redis.exceptions.ResponseError:
            pass

        while True:
            try:
                resp = self.redis.xreadgroup(
                    groupname=GROUP_NAME,
                    consumername="worker-1",
                    streams={INPUT_QUEUE: ">"},
                    count=1,
                    block=5000,
                )
                if resp:
                    stream_name, messages = resp[0]
                    msg_id, message = messages[0]
                    self.process_message(message)
                    # Increment acknowledgment count for the frame cleanup system
                    self.mark_acked(INPUT_QUEUE, msg_id.decode(), GROUP_NAME)
                    self.redis.xack(INPUT_QUEUE, GROUP_NAME, msg_id)
            except Exception as e:
                logger.error(f"Loop error: {e}")
                time.sleep(1)

if __name__ == "__main__":
    service = StoreOpeningClosingService()
    service.run()
