import json
import numpy as np
import cv2
from dotenv import load_dotenv
from pathlib import Path
import redis
import logging
import time
import requests
import logging_loki
from multiprocessing import Queue
import os
import ast
from collections import deque, defaultdict
from shapely.geometry import Polygon, box, Point
import base64
import uuid
import statistics

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
LOKI_URL = os.getenv("LOKI_URL", "http://localhost:3100/loki/api/v1/push")

INPUT_QUEUE = os.getenv("INPUT_QUEUE_NAME", "kiosk_person_input")
OUTPUT_QUEUE = os.getenv("OUTPUT_QUEUE_NAME", "kiosk_person_output")
YOLO_MODEL_URL = os.getenv("YOLO_MODEL_URL", "http://yolo_service:8000/predict")
GROUP_NAME = os.getenv("GROUP_NAME", "kiosk_person_group")
TARGET_COLLECTION_NAME = os.getenv("TARGET_COLLECTION_NAME", "kiosk_person_collection")
ACK_TTL_SECONDS = 10 * 60  # 10 minutes
ACK_TTL_SECONDS = 10 * 60  # 10 minutes
VISUALIZE = os.getenv("VISUALIZE", "false").lower() in ("true", "1", "yes")
FPS = float(os.getenv("FPS", "1.0"))

# Configure Logging
handler = logging_loki.LokiQueueHandler(
    Queue(-1),
    url=LOKI_URL,
    tags={"application": "kiosk_person_service"},
    version="1",
)


class FixedSizeQueue:
    def __init__(self, redis_client, name, max_size=10):
        self.r = redis_client
        self.name = name
        self.max_size = max_size

    def push(self, item):
        """
        Push item to queue.
        Oldest items are dropped.
        """
        pipe = self.r.pipeline()
        pipe.rpush(self.name, item)
        pipe.ltrim(self.name, -self.max_size, -1)
        pipe.execute()


logger = logging.getLogger("kiosk_person_service")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)


class KioskPersonService:
    def __init__(self):
        self.settings = self.load_settings()
        self.redis = self.connect_redis(host=REDIS_HOST, port=REDIS_PORT)
        self.yolo_endpoint = YOLO_MODEL_URL
        # History: camera_id -> roi_key -> deque
        self.history = defaultdict(lambda: defaultdict(lambda: deque(maxlen=60)))
        # Person Count History: camera_id -> deque (last 5 counts)
        self.person_count_history = defaultdict(lambda: deque(maxlen=5))

        # Kiosk ON/OFF Status History: camera_id -> kiosk_key -> deque (0=OFF, 1=ON, 2=Person Present/Ignore)
        # self.history is now shared for both alerts and status

        # Current Kiosk ON/OFF Status: camera_id -> kiosk_key -> bool (True=ON, False=OFF)
        self.kiosk_status = defaultdict(dict)

        # Coordinate config
        # Assuming we need to load Polygon 3 for each camera or a default
        # For now, we load 'target_polygon' from settings
        self.polygons = {}
        for cam_id, config in self.settings.items():
            pts = config.get("target_polygon", [])
            if pts:
                # Convert list of dicts to list of tuples
                coords = [(p["x"], p["y"]) for p in pts]
                self.polygons[cam_id] = Polygon(coords)

        # Visualization queues
        self.vis_queues = {}

        # Alert State Tracking
        self.alert_state = defaultdict(dict)
        self.active_tickets = defaultdict(dict)

        # Crowd Alert State Tracking
        self.crowd_history = defaultdict(lambda: deque(maxlen=15))
        self.crowd_alert_state = defaultdict(bool)
        self.crowd_active_tickets = defaultdict(str)

        # FPS Throttling
        self.last_processed = defaultdict(float)

    def load_settings(self):
        base_dir = Path(__file__).parent.resolve()
        settings_path = base_dir / "artifacts" / "settings.json"
        with open(settings_path, "r") as f:
            return json.load(f)

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

    def annotate_frame(self, img, camera_id, detections=None):
        # Draw Target Polygon
        cam_config = self.settings.get(str(camera_id), {})
        pts = cam_config.get("target_polygon", [])
        if pts:
            points = np.array([[int(p["x"]), int(p["y"])] for p in pts], np.int32)
            points = points.reshape((-1, 1, 2))
            cv2.polylines(img, [points], True, (255, 255, 0), 2)  # Cyan

        # Draw Person Detections
        if detections:
            for bbox in detections:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 2)  # Orange
                cv2.putText(
                    img, "Person", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2
                )

        # Draw Kiosks
        kiosks_config = cam_config.get("kiosks", {})
        for k_key, rect in kiosks_config.items():
            if not rect:
                continue
            x = int(rect["xMin"])
            y = int(rect["yMin"])
            w = int(rect["width"])
            h = int(rect["height"])
            x2 = x + w
            y2 = y + h

            k_status = self.kiosk_status.get(camera_id, {}).get(k_key, "UNKNOWN")

            if k_status == "ON":
                color = (0, 255, 0)  # Green
            elif k_status == "OFF":
                color = (0, 0, 255)  # Red
            else:
                color = (0, 255, 255)  # Yellow

            cv2.rectangle(img, (x, y), (x2, y2), color, 2)

            label = f"{k_key}: {k_status}"
            cv2.putText(
                img, label, (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

        return img

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

    def _push_alert(self, camera_id, timestamp, type_, desc, img_content, ticket_id=None, detections=None):
        tid = ticket_id or str(uuid.uuid4())

        frame_b64 = ""
        raw_frame_b64 = ""

        if img_content:
            try:
                # Decode
                arr = np.frombuffer(img_content, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                # 1. Encode RAW
                _, raw_buffer = cv2.imencode(".jpg", img)
                raw_frame_b64 = base64.b64encode(raw_buffer).decode("utf-8")

                # 2. Annotate
                img = self.annotate_frame(img.copy(), camera_id, detections=detections)

                # 3. Encode Annotated
                _, buffer = cv2.imencode(".jpg", img)
                frame_b64 = base64.b64encode(buffer).decode("utf-8")
            except Exception as e:
                logger.error(f"Failed to process alert image: {e}")
                # Fallback: send raw as 'frame' if annotation fails
                frame_b64 = base64.b64encode(img_content).decode("utf-8")

        try:
            pl = {
                "camera_id": camera_id,
                "timestamp": timestamp,
                "frame": frame_b64,
                "frame_without_visualization": raw_frame_b64,
                "description": desc,
                "target_collection": "alert_collection",
                "type": type_,
                "ticket_id": tid,
            }
            self.redis.xadd("database:queue:0", {"data": json.dumps(pl)})
            return tid
        except Exception as e:
            logger.error(f"Alert push failed: {e}")
            return None

    def process_message(self, message: dict):
        message = {k.decode(): v.decode() for k, v in message.items()}
        frame_url = message.get("frame_url")
        camera_id = message.get("camera_id")
        timestamp = message.get("timestamp")

        if not frame_url:
            return

        # FPS Control Check
        # Note: Ideally we should use the timestamp from the frame, but for simplicity/robustness with unknown stream sources,
        # we can just throttle processing here if the stream is live.
        # However, since this is an analytics service reading from Redis, likely pushed by a stream handler at a certain rate.
        # If we want to enforce 1FPS processing regardless of input rate:
        # We'll rely on the sleep at the end or just process everything?
        # User requested 60s window. If input is 15FPS, 60 samples is 4s.
        # We MUST ensure we store ~1 sample per second OR increase buffer size.
        # Best approach: Process whatever we get, but if we get data too fast, we might skew stats.
        # But 'kiosk_person_service' usually implies analytics. Let's assume input might be bursty.
        # Actually, standard pattern is to drop frames if we are processing too fast, or just rely on 'sleep' in the consumer loop?
        # The consumer loop uses xreadgroup block.
        # Let's add a `last_processed_time` tracker per camera to throttle to FPS.

        # Initialize throttle tracker
        # Initialize throttle tracker (Moved to __init__, keeping fallback just in case or removing)
        # if not hasattr(self, "last_processed"):
        #     self.last_processed = defaultdict(float)

        current_time = time.time()
        if current_time - self.last_processed[camera_id] < (1.0 / FPS):
            # Skip this frame to maintain FPS rate
            return

        self.last_processed[camera_id] = current_time

        # 1. Download Image (Needed for ROI Stats)
        try:
            img_resp = requests.get(frame_url, timeout=10)
            img_resp.raise_for_status()
            img_array = np.frombuffer(img_resp.content, np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Failed to decode image")
        except Exception as e:
            logger.error(f"Failed to download/decode frame {frame_url}: {e}")
            return

        # 2. Inference
        try:
            payload = {"image_url": frame_url}
            # Note: YOLO service might download it again, optimal to send bytes if supported but keeping URL for now
            resp = requests.post(self.yolo_endpoint, json=payload, timeout=20)
            resp.raise_for_status()
            yolo_result = resp.json()
        except Exception as e:
            logger.error(f"Inference failed for {frame_url}: {e}")
            return

        logger.info(f"Inference done for {camera_id} at {timestamp}")

        # 3. Parse Detections
        detections = []
        try:
            predict_list = yolo_result.get("predict", [])
            if predict_list:
                detections_json = predict_list[0]
                if isinstance(detections_json, str):
                    import json as _json

                    parsed = _json.loads(detections_json)
                else:
                    parsed = detections_json

                for item in parsed:
                    if item.get("name") == "person":
                        box_data = item.get("box", {})
                        bbox = [
                            box_data.get("x1"),
                            box_data.get("y1"),
                            box_data.get("x2"),
                            box_data.get("y2"),
                        ]
                        if all(v is not None for v in bbox):
                            detections.append(bbox)
        except Exception as e:
            logger.error(f"Result parsing failed: {e}")
            return

        # 5. Core Logic
        target_poly = self.polygons.get(camera_id)
        if not target_poly:
            logger.error(
                f"No polygon configuration found for camera {camera_id}. Skipping frame."
            )
            return

        # 5a. Identify Target Persons (Overlap with Polygon 3)
        target_person_indices = []  # Indices of detections that are 'Target Persons'
        person_count = 0

        for idx, bbox in enumerate(detections):
            # Bottom Mid-Point Check
            x1, y1, x2, y2 = bbox
            mid_x = (x1 + x2) / 2
            bottom_y = y2
            point = Point(mid_x, bottom_y)

            if target_poly.contains(point):
                person_count += 1
                target_person_indices.append(idx)

        target_person_detections = [detections[i] for i in target_person_indices]

        # Crowd Alert Logic (More than 3 persons present over 15 frames maxlen buffer)
        self.crowd_history[camera_id].append({
            "count": person_count,
            "image_content": img_resp.content,
            "detections": target_person_detections
        })

        crowd_q = self.crowd_history[camera_id]
        crowded_frames = sum(1 for item in crowd_q if item["count"] > 3)
        not_crowded_frames = len(crowd_q) - crowded_frames

        is_crowd_active = self.crowd_alert_state[camera_id]

        if crowded_frames >= 8 and not is_crowd_active:
            # RAISE
            target_content = img_resp.content
            target_detections = target_person_detections
            for item in reversed(crowd_q):
                if item["count"] > 3:
                    target_content = item["image_content"]
                    target_detections = item.get("detections", [])
                    break

            tid = self._push_alert(
                camera_id,
                timestamp,
                "raised",
                "High customer traffic detected at Kiosk Area (More than 3 customers).",
                target_content,
                detections=target_detections
            )
            if tid:
                self.crowd_alert_state[camera_id] = True
                self.crowd_active_tickets[camera_id] = tid
                logger.info(f"Raised Crowd Alert for {camera_id}")

        elif not_crowded_frames >= 8 and is_crowd_active:
            # RESOLVE
            target_content = img_resp.content
            target_detections = target_person_detections
            for item in reversed(crowd_q):
                if item["count"] <= 3:
                    target_content = item["image_content"]
                    target_detections = item.get("detections", [])
                    break

            tid = self.crowd_active_tickets[camera_id]
            tid = self._push_alert(
                camera_id,
                timestamp,
                "resolved",
                "High customer traffic at Kiosk Area resolved.",
                target_content,
                ticket_id=tid,
                detections=target_detections
            )
            self.crowd_alert_state[camera_id] = False
            self.crowd_active_tickets[camera_id] = None
            logger.info(f"Resolved Crowd Alert for {camera_id}")

        # 5b. ROI Stats & Alerting
        cam_config = self.settings.get(camera_id)
        roi_stats = {}

        # Initialize state for this camera if needed
        # We need persistant state. Ideally this should be in __init__, but keys are dynamic.
        # We'll rely on lazy initialization in self.history/alert_state if possible or handle it here.
        # Check if we need to init queues in self.history

        if cam_config:
            # Helper to check overlap with TARGET PERSONS only
            def check_filtered_overlap(rect, target_indices, detections):
                rx1, ry1 = rect["xMin"], rect["yMin"]
                rx2, ry2 = rx1 + rect["width"], ry1 + rect["height"]
                r_box = box(rx1, ry1, rx2, ry2)

                for idx in target_indices:
                    bbox = detections[idx]
                    bx1, by1, bx2, by2 = bbox
                    b_box = box(bx1, by1, bx2, by2)
                    if r_box.intersects(b_box):
                        return True
                return False

            kiosks_config = cam_config.get("kiosks", {})
            for key, rect in kiosks_config.items():
                if not rect:
                    continue

                # Calc Gray Mean
                stats = None
                try:
                    x = int(rect["xMin"])
                    y = int(rect["yMin"])
                    w = int(rect["width"])
                    h = int(rect["height"])

                    H, W = frame.shape[:2]
                    x2 = min(x + w, W)
                    y2 = min(y + h, H)

                    if x < x2 and y < y2:
                        roi = frame[y:y2, x:x2]
                        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        gray_mean = float(gray.mean())

                        is_overlap = check_filtered_overlap(
                            rect, target_person_indices, detections
                        )

                        # Initialize stats dict
                        stats = {"gray_mean": gray_mean, "is_overlap": is_overlap}
                        roi_stats[key] = stats

                        # --- Unified Logic (Status & Alerts) ---
                        # State Determination:
                        # 0: OFF/Error (Dark & Empty)
                        # 1: ON/Healthy (Bright & Empty)
                        # 2: IGNORED (Occupied)

                        BRIGHTNESS_THRESHOLD = 55
                        current_state = 2  # default: person present (2)

                        if not is_overlap:
                            current_state = (
                                1 if gray_mean >= BRIGHTNESS_THRESHOLD else 0
                            )

                        # Update History Queue (Shared for both Status and Alerts)
                        # history structure: self.history[camera_id][roi_key] -> deque
                        if key not in self.history[camera_id]:
                            self.history[camera_id][key] = deque(maxlen=60)

                        q = self.history[camera_id][key]
                        # Store dict with status and image content to allow representative frame selection
                        q.append(
                            {"status": current_state, "image_content": img_resp.content}
                        )

                        # Check Majority (42/60 = 70%)
                        # 0s = OFF/Error, 1s = ON/Healthy
                        off_count = sum(1 for item in q if item["status"] == 0)
                        on_count = sum(1 for item in q if item["status"] == 1)

                        # --- 1. Update Kiosk ON/OFF Status ---
                        # Initialize if not set
                        if key not in self.kiosk_status[camera_id]:
                            self.kiosk_status[camera_id][key] = True  # default ON

                        if on_count >= 42:
                            self.kiosk_status[camera_id][key] = True
                        elif off_count >= 42:
                            self.kiosk_status[camera_id][key] = False
                        # else: keep previous status

                        # Add kiosk status to stats
                        stats["is_on"] = self.kiosk_status[camera_id][key]

                        # Log detailed info for each region
                        current_state_str = "IGNORED (Occupied)" if current_state == 2 else ("ON" if current_state == 1 else "OFF")
                        final_status_str = "ON" if self.kiosk_status[camera_id][key] else "OFF"
                        logger.info(
                            f"Cam: {camera_id} | Region: {key} | "
                            f"Target People: {person_count} | Overlap: {is_overlap} | "
                            f"Pixel Value (Gray Mean): {gray_mean:.2f} | "
                            f"Current Frame: {current_state_str} | "
                            f"Last 60 Frames: {final_status_str} (ON: {on_count}, OFF: {off_count})"
                        )

                        # --- 2. Alert Logic ---
                        is_error_majority = off_count >= 42
                        is_healthy_majority = on_count >= 42

                        # Make Decision
                        # We need alert state: self.alert_state[camera_id][key] (True/False)
                        # and ticket id: self.active_tickets[camera_id][key]

                        is_active = self.alert_state[camera_id].get(key, False)

                        if is_error_majority and not is_active:
                            # RAISE
                            # Find representative frame (latest one with status 0)
                            target_content = img_resp.content
                            for item in q:
                                if item["status"] == 0:
                                    target_content = item["image_content"]

                            tid = self._push_alert(
                                camera_id,
                                timestamp,
                                "raised",
                                f"Kiosk Error Region {key}: Dark Screen Detected",
                                target_content,
                            )
                            if tid:
                                self.alert_state[camera_id][key] = True
                                self.active_tickets[camera_id][key] = tid
                                logger.info(f"Raised Alert for {camera_id} {key}")

                        elif is_healthy_majority and is_active:
                            # RESOLVE
                            # Find representative resolution frame
                            target_content = img_resp.content
                            for item in q:
                                if item["status"] == 1:
                                    target_content = item["image_content"]

                            tid = self.active_tickets[camera_id].get(key)
                            self._push_alert(
                                camera_id,
                                timestamp,
                                "resolved",
                                f"Kiosk Error Resolved Region {key}",
                                img_content=target_content,
                                ticket_id=tid,
                            )
                            self.alert_state[camera_id][key] = False
                            self.active_tickets[camera_id][key] = None
                            logger.info(f"Resolved Alert for {camera_id} {key}")

                except Exception as e:
                    logger.error(f"Error processing ROI {key}: {e}")

        # Update Person Count History
        self.person_count_history[camera_id].append(person_count)

        # Calculate Stats
        count_q = self.person_count_history[camera_id]
        median_val = statistics.median(count_q) if count_q else 0
        mean_val = statistics.mean(count_q) if count_q else 0
        try:
            mode_val = statistics.mode(count_q) if count_q else 0
        except statistics.StatisticsError:
            # No unique mode - use median as fallback
            mode_val = median_val

        # Prepare kiosk status summary
        kiosk_status_summary = {}
        for kiosk_key, is_on in self.kiosk_status.get(camera_id, {}).items():
            kiosk_status_summary[kiosk_key] = "ON" if is_on else "OFF"

        # Publish Result
        output_payload = {
            "camera_id": camera_id,
            "timestamp": timestamp,
            "target_person_count": person_count,
            "target_person_mean_5s": round(mean_val, 2),
            "target_person_median_5s": median_val,
            "target_person_mode_5s": mode_val,
            "target_person_detections": target_person_detections,
            "roi_stats": roi_stats,
            "kiosk_status": kiosk_status_summary,
            "service": "kiosk_person_service",
            "target_collection": TARGET_COLLECTION_NAME,
        }

        try:
            self.redis.xadd(OUTPUT_QUEUE, {"data": json.dumps(output_payload)})
            logger.info(
                f"Processed {camera_id} at {timestamp}: Count={person_count}, Stats={len(roi_stats)}, Median={median_val}, KioskStatus={kiosk_status_summary}"
            )

            if VISUALIZE:
                try:
                    # Encode frame for visualization
                    frame_b64 = base64.b64encode(img_resp.content).decode("utf-8")

                    vis_payload = {
                        **output_payload,
                        "frame": frame_b64,
                        "all_detections": detections,
                    }

                    vis_queue_name = f"stream:visualization:kiosk_person:{camera_id}"
                    if vis_queue_name not in self.vis_queues:
                        self.vis_queues[vis_queue_name] = FixedSizeQueue(
                            self.redis, vis_queue_name, 10
                        )

                    self.vis_queues[vis_queue_name].push(json.dumps(vis_payload))

                except Exception as e:
                    logger.error(f"Failed to push visualization frame: {e}")

        except Exception as e:
            logger.error(f"Redis publish failed: {e}")

    def run(self):
        logger.info("PersonCounterService started.")
        # Ensure group
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
                    self.redis.xack(INPUT_QUEUE, GROUP_NAME, msg_id)
                    self.mark_acked(INPUT_QUEUE, msg_id.decode(), GROUP_NAME)
            except Exception as e:
                logger.error(f"Loop error: {e}")
                time.sleep(1)


if __name__ == "__main__":
    service = KioskPersonService()
    service.run()
