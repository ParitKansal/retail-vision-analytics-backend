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
import base64
from collections import deque, defaultdict
import uuid
import ast
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import statistics
from insightface.app import FaceAnalysis
from shapely.geometry import box, LineString
from PIL import Image
from transformers import pipeline as hf_pipeline

# from deepface import DeepFace  # Not used - InsightFace handles face detection

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
LOKI_URL = os.getenv("LOKI_URL", "http://localhost:3100/loki/api/v1/push")

INPUT_QUEUE = os.getenv("INPUT_QUEUE_NAME")
OUTPUT_QUEUE_1 = os.getenv("OUTPUT_QUEUE_NAME_1")
YOLO_MODEL_URL = os.getenv("YOLO_MODEL_URL")
STAFF_CUSTOMER_CLASSIFICATION_URL = os.getenv("STAFF_CUSTOMER_CLASSIFICATION_URL")
GROUP_NAME = os.getenv("GROUP_NAME")
CAMERA_ID = os.getenv("CAMERA_ID", INPUT_QUEUE.split(":")[-1])
VISUALIZE = str(os.getenv("VISUALIZE", "False")).lower() == "true"
NMS_THRESHOLD = float(os.getenv("NMS_THRESHOLD", "0.5"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.4"))
ACK_TTL_SECONDS = 10 * 60  # 10 minutes

# Depth estimation config
DEPTH_MODEL_NAME = os.getenv("DEPTH_MODEL_NAME", "depth-anything/Depth-Anything-V2-Small-hf")
DEPTH_DEVICE = os.getenv("DEPTH_DEVICE", "cuda")  # 'cuda' or 'cpu'

# Configure Logging
handler = logging_loki.LokiQueueHandler(
    Queue(-1),
    url=LOKI_URL,
    tags={"application": "counter_staff_detection_service"},
    version="1",
)

logger = logging.getLogger("counter_staff_detection_service")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
# Also log to console
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)


# r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)


class FixedSizeQueue:
    def __init__(self, redis_client, name, max_size):
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

    def pop(self, timeout=0):
        """
        Blocking FIFO pop.
        """
        result = self.r.blpop(self.name, timeout=timeout)
        if result:
            _, value = result
            return value
        return None

    def size(self):
        return self.r.llen(self.name)


class CounterStaffDetector:
    def __init__(self):
        # Load configuration settings
        self.settings = self.load_settings()
        # Redis connection
        self.redis = self.connect_redis(host=REDIS_HOST, port=REDIS_PORT)
        # YOLO endpoint URL is taken from environment variable YOLO_MODEL_URL
        self.yolo_endpoint = YOLO_MODEL_URL
        self.history = defaultdict(lambda: deque(maxlen=15))
        self.last_staff_detected_state = defaultdict(lambda: True)
        self.current_alert_id = defaultdict(lambda: None)

        self.last_unattended_state = defaultdict(lambda: False)
        self.current_unattended_alert_id = defaultdict(lambda: None)

        self.last_high_traffic_state = defaultdict(lambda: False)
        self.current_high_traffic_alert_id = defaultdict(lambda: None)


        # ThreadPool for parallel classification requests
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Setup visualization queue if enabled
        if VISUALIZE:
            self.vis_queue_name = os.getenv("VIS_QUEUE_NAME", f"stream:visualization:cafe_counter:{CAMERA_ID}")
            self.vis_queue = FixedSizeQueue(
                self.redis, self.vis_queue_name, 10
            )
            logger.info(f"Visualization initialized on queue: {self.vis_queue_name}")

        try:
            # Initialize InsightFace (persist models in /workspace/insightface_models)
            self.face_app = FaceAnalysis(
                name="buffalo_l", root="/workspace/insightface_models", allowed_modules=["detection"]
            )
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
            logging.info("InsightFace app prepared.")
        except Exception as e:
            logger.error(f"Error initializing InsightFace: {e}")

        # Load Depth-Anything pipeline once at startup (GPU if available)
        try:
            _depth_device = 0 if DEPTH_DEVICE == "cuda" else -1
            self.depth_pipe = hf_pipeline(
                task="depth-estimation",
                model=DEPTH_MODEL_NAME,
                device=_depth_device,
            )
            logger.info(f"Depth pipeline loaded: {DEPTH_MODEL_NAME} on device={DEPTH_DEVICE}")
        except Exception as e:
            self.depth_pipe = None
            logger.error(f"Failed to load depth pipeline: {e}")

        logger.info(
            f"VISUALIZE status: {VISUALIZE}, Queue: stream:visualization:{CAMERA_ID}"
        )
        self.load_alert_state()
        logger.info("Service ready (depth-based customer detection enabled).")

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

    def load_settings(self):
        """Load detector configuration from setting.json relative to this file"""
        base_dir = Path(__file__).parent.resolve()
        settings_path = base_dir / "setting.json"
        with open(settings_path, "r") as f:
            data = json.load(f)
        camera_id = INPUT_QUEUE.split(":")[-1]
        return data.get(camera_id, None)

    def load_alert_state(self):
        """Load active alert states from Redis on startup."""
        try:
            # Unattended
            ua_key = f"alert_state:unattended:{CAMERA_ID}"
            ua_id = self.redis.get(ua_key)
            if ua_id:
                self.last_unattended_state[CAMERA_ID] = True
                self.current_unattended_alert_id[CAMERA_ID] = ua_id.decode()
                logger.info(f"Restored Unattended Alert state for {CAMERA_ID}: {ua_id.decode()}")

            # High Traffic
            ht_key = f"alert_state:high_traffic:{CAMERA_ID}"
            ht_id = self.redis.get(ht_key)
            if ht_id:
                self.last_high_traffic_state[CAMERA_ID] = True
                self.current_high_traffic_alert_id[CAMERA_ID] = ht_id.decode()
                logger.info(f"Restored High Traffic Alert state for {CAMERA_ID}: {ht_id.decode()}")


        except Exception as e:
            logger.error(f"Failed to load alert state from Redis: {e}")

    # The YOLO model is served remotely; inference is performed via HTTP request.
    # No local model loading is required.

    def classify_object(self, frame_url, bbox):
        """
        Send frame URL and bbox to the classification service.
        bbox: [x1, y1, x2, y2] (pixels)
        Returns: 'staff', 'customer', or None (on error)
        """
        if not STAFF_CUSTOMER_CLASSIFICATION_URL:
            return "customer"

        try:
            payload = {
                "image_url": frame_url,
                "bbox": bbox
            }

            # Send request
            resp = requests.post(
                STAFF_CUSTOMER_CLASSIFICATION_URL, json=payload, timeout=5
            )
            resp.raise_for_status()

            result = resp.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("predicted_label")
            elif isinstance(result, dict):
                return result.get("predicted_label")

            return None
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return "customer"

    def classify_objects_batch(self, frame_url, bboxes):
        """
        Send frame URL and multiple bboxes to the classification service in a single request.
        bboxes: List of [x1, y1, x2, y2] (pixels)
        Returns: List of labels ['staff', 'customer', ...] or None on error
        """
        if not STAFF_CUSTOMER_CLASSIFICATION_URL:
            return ["customer"] * len(bboxes)
        
        if not bboxes:
            return []

        try:
            payload = {
                "image_url": frame_url,
                "bboxes": bboxes
            }

            # Send batch request
            resp = requests.post(
                STAFF_CUSTOMER_CLASSIFICATION_URL, json=payload, timeout=10
            )
            resp.raise_for_status()

            result = resp.json()
            if isinstance(result, list):
                # Extract labels from batch response
                labels = []
                for item in result:
                    if isinstance(item, dict):
                        labels.append(item.get("predicted_label"))
                    else:
                        labels.append(None)
                return labels
            
            return [None] * len(bboxes)
        except Exception as e:
            logger.error(f"Batch classification failed: {e}")
            return [None] * len(bboxes)


    def draw_text_with_bg(self, img, text, position, font_scale=0.5, font_thickness=2, text_color=(0, 255, 0), bg_color=(0, 0, 0), padding=2):
        """
        Draw text with a background rectangle for better legibility.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        
        x, y = position
        # Ensure text is within bounds (rudimentary)
        x = max(padding, x)
        y = max(text_h + padding, y)
        
        # Draw background rectangle
        cv2.rectangle(img, (x - padding, y - text_h - padding), (x + text_w + padding, y + padding), bg_color, -1)
        # Draw text
        cv2.putText(img, text, (x, y), font, font_scale, text_color, font_thickness)
        
        return text_w + 2 * padding

    def annotate_frame(
        self,
        img,
        staff_polygons,
        customer_polygons,
        customer_line_points,
        staff_detections,
        customer_detections,
        all_customer_hits,
        customer_landmarks,
    ):
        """
        Annotate the frame with all visual information.
        """
        height, width = img.shape[:2]

        # Draw Staff Polygons (Green)
        for poly in staff_polygons:
            pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)

        # Draw Customer Polygons (Orange)
        for poly in customer_polygons:
            pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, (0, 165, 255), 2)

        # Draw Customer Line (Blue/Cyan)
        if customer_line_points and len(customer_line_points) >= 2:
            p1 = (int(customer_line_points[0][0]), int(customer_line_points[0][1]))
            p2 = (int(customer_line_points[1][0]), int(customer_line_points[1][1]))
            cv2.line(img, p1, p2, (255, 255, 0), 2)  # Cyan

        # Draw Staff Detections (Green)
        for det in staff_detections:
            bbox = det.get("bbox")
            if bbox:
                x1 = int(bbox[0] * width)
                y1 = int(bbox[1] * height)
                x2 = int(bbox[2] * width)
                y2 = int(bbox[3] * height)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = "Staff"
                depth = det.get("depth")
                if depth is not None:
                    depth_text = f"D:{depth:.1f}"
                    self.draw_text_with_bg(img, depth_text, (x1, max(0, y1 - 25)))
                
                cv2.putText(
                    img,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        # Draw Customer Detections (Orange)
        for det in customer_detections:
            bbox = det.get("bbox")
            if bbox:
                x1 = int(bbox[0] * width)
                y1 = int(bbox[1] * height)
                x2 = int(bbox[2] * width)
                y2 = int(bbox[3] * height)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 2)  # Orange
                label = "Customer"
                depth = det.get("depth")
                if depth is not None:
                    depth_text = f"D:{depth:.1f}"
                    self.draw_text_with_bg(img, depth_text, (x1, max(0, y1 - 25)))

                cv2.putText(
                    img,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 165, 255),
                    2,
                )

        # Draw Facial Landmarks (Customer Only)
        for landmarks in customer_landmarks:
            le = landmarks.get("left_eye")
            re = landmarks.get("right_eye")
            nose = landmarks.get("nose")

            # Eyes: Yellow (0, 255, 255) in BGR
            if le:
                cv2.circle(img, le, 3, (0, 255, 255), -1)
            if re:
                cv2.circle(img, re, 3, (0, 255, 255), -1)

            # Nose: Purple (255, 0, 255) in BGR
            if nose:
                cv2.circle(img, nose, 3, (255, 0, 255), -1)

        # Draw All Overlapping Hits (Yellow thin boxes)
        COLOR_ALL_HITS = (0, 255, 255)
        for idx, det in enumerate(all_customer_hits or []):
            bbox = det.get("bbox")
            if bbox:
                x1 = int(bbox[0] * width)
                y1 = int(bbox[1] * height)
                x2 = int(bbox[2] * width)
                y2 = int(bbox[3] * height)
                cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_ALL_HITS, 1)
                
                depth = det.get("depth")
                if depth is not None:
                    depth_text = f"D:{depth:.1f}"
                    self.draw_text_with_bg(img, depth_text, (x1, min(height, y2 + 35)), font_scale=0.4, font_thickness=1)
                
                cv2.putText(
                    img,
                    f"Overlap {idx+1}",
                    (x1, min(height, y2 + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    COLOR_ALL_HITS,
                    1,
                )

        return img

    def preprocessing(self, frame_url):
        """
        Download and decode the frame.
        """
        try:
            resp = requests.get(frame_url, timeout=10)
            resp.raise_for_status()
            arr = np.frombuffer(resp.content, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            logger.error(f"Failed to download/decode frame from {frame_url}: {e}")
            return None

    def compute_full_depth_map(self, frame_img) -> np.ndarray:
        """
        Run depth estimation on the FULL frame and return the depth map.
        Matches temp.txt reference: depth is estimated globally so all
        per-person slices are comparable within the same scene.
        Returns None on failure.
        """
        if self.depth_pipe is None or frame_img is None:
            return None
        try:
            # BGR → RGB PIL for HuggingFace pipeline
            pil_img = Image.fromarray(cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB))
            depth_output = self.depth_pipe(pil_img)
            return np.array(depth_output["depth"])   # shape (H, W)
        except Exception as e:
            logger.error(f"Full-frame depth estimation failed: {e}")
            return None

    def post_processing(self, results, frame_shape, frame_img=None, frame_url=None):
        height, width = frame_shape[:2]
        if height == 0 and width == 0:
            logger.error(f"height = {height} width = {width}")
            raise ValueError(f"height = {height} width = {width}")

        staff_polygons_config = self.settings.get("staff_polygons", [])
        # content_line_points and customer_polygons are NOT used in this version.

        # Convert staff polygons to pixel coordinates
        staff_pixel_polygons = []
        staff_pixel_polygons_list = []
        for poly in staff_polygons_config:
            normalized_points = poly.get("normalized", [])
            points = []
            for pt in normalized_points:
                x = int(pt["x"] * width)
                y = int(pt["y"] * height)
                points.append([x, y])
            if points:
                staff_pixel_polygons.append(np.array(points, dtype=np.int32))
                staff_pixel_polygons_list.append(points)

        # Convert customer polygons to pixel coordinates
        # Each entry MUST carry its own depth_min/depth_max in setting.json.
        customer_polygons_config = self.settings.get("customer_polygons", [])
        customer_pixel_polygons = []  # list of {"poly": np.array, "depth_min": float, "depth_max": float}
        customer_pixel_polygons_list = []
        for poly_cfg in customer_polygons_config:
            normalized_points = poly_cfg.get("normalized", [])
            points = []
            for pt in normalized_points:
                x = int(pt["x"] * width)
                y = int(pt["y"] * height)
                points.append([x, y])
            if len(points) >= 3:
                # If depth is missing, use invalid range [999, -999] (always fails)
                d_min = float(poly_cfg.get("depth_min", 999.0))
                d_max = float(poly_cfg.get("depth_max", -999.0))
                if "depth_min" not in poly_cfg:
                    logger.warning(f"Polygon in camera {self.camera_id} missing depth_min — will reject all persons in this region.")
                
                customer_pixel_polygons.append({
                    "poly":      np.array(points, dtype=np.int32),
                    "depth_min": d_min,
                    "depth_max": d_max,
                })
                customer_pixel_polygons_list.append(points)

        staff_detections = []
        staff_candidates = []  # Persons inside staff region
        customer_detections = []
        customer_landmarks = []  # kept for vizualize.py compatibility
        customer_detections_candidates = []  # Depth-filtered, pending classification
        all_customer_hits = []  # All persons overlapping a customer region

        # ------------------------------------------------------------------ #
        # Compute depth map ONCE for the whole frame (matches temp.txt)       #
        # ------------------------------------------------------------------ #
        depth_map = self.compute_full_depth_map(frame_img)  # shape (H, W) or None
        if depth_map is None:
            logger.warning("Depth map unavailable — depth-based customer filtering skipped.")

        # ------------------------------------------------------------------ #
        # Per-detection loop                                                  #
        # ------------------------------------------------------------------ #
        for box_det in results:
            if int(box_det.get("class")) != 0:
                continue

            x1, y1, x2, y2 = box_det.get("bbox")
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            center_point = (center_x, center_y)

            # Normalized bbox for output payload
            normalized_bbox = [x1 / width, y1 / height, x2 / width, y2 / height]
            normalized_box = box_det.copy()
            normalized_box["bbox"] = normalized_bbox

            # ----------------------------------------------------------------
            # STAFF LOGIC (UNCHANGED) — center-point polygon test
            # Priority: if inside staff region -> staff, skip customer check
            # ----------------------------------------------------------------
            is_inside = False
            region_id = -1
            for idx, poly in enumerate(staff_pixel_polygons):
                if cv2.pointPolygonTest(poly, center_point, False) >= 0:
                    is_inside = True
                    region_id = idx
                    break

            if is_inside:
                normalized_box["region_id"] = region_id
                staff_candidates.append(normalized_box)
                continue  # Do NOT check customer logic for this person

            # ----------------------------------------------------------------
            # CUSTOMER REGION GATE — bbox must overlap a customer polygon.
            # Returns the matched polygon's depth range for the depth gate.
            # MANDATORY: If no customer polygons configured, NO detection passes.
            # ----------------------------------------------------------------
            matched_depth_min = 999.0
            matched_depth_max = -999.0

            if not customer_pixel_polygons:
                logger.debug("No customer regions configured — person rejected.")
                continue

            matched_entry = None
            for entry in customer_pixel_polygons:
                cpoly = entry["poly"]
                test_points = [
                    (float(x1), float(y1)), (float(x2), float(y1)),
                    (float(x1), float(y2)), (float(x2), float(y2)),
                    (float(center_x), float(center_y)),
                ]
                hit = any(
                    cv2.pointPolygonTest(cpoly, pt, False) >= 0
                    for pt in test_points
                )
                # Also check: any polygon vertex inside bbox
                if not hit:
                    hit = any(
                        x1 <= int(pt[0]) <= x2 and y1 <= int(pt[1]) <= y2
                        for pt in cpoly
                    )
                if hit:
                    matched_entry = entry
                    break

            if matched_entry is None:
                logger.debug(
                    f"Person [{x1},{y1},{x2},{y2}] outside all customer regions — skipped."
                )
                continue

            matched_depth_min = matched_entry["depth_min"]
            matched_depth_max = matched_entry["depth_max"]
            logger.debug(
                f"Person [{x1},{y1},{x2},{y2}] matched customer region "
                f"depth=[{matched_depth_min}, {matched_depth_max}]"
            )

            # (Moved downstream after depth calculation for visualization)

            # ----------------------------------------------------------------
            # DEPTH GATE — use the matched polygon's depth range
            # ----------------------------------------------------------------
            if box_det.get("confidence", 0.0) < CONFIDENCE_THRESHOLD:
                continue

            if depth_map is None:
                continue  # No depth available, skip

            # Apply bbox padding then clip to frame bounds
            pad = 5
            px1, py1 = max(0, x1 + pad), max(0, y1 + pad)
            px2, py2 = min(width, x2 - pad), min(height, y2 - pad)

            depth_crop = depth_map[py1:py2, px1:px2]
            if depth_crop.size == 0:
                continue

            depth_val = float(np.median(depth_crop))
            # Record this overlap hit with its depth for visualization
            box_with_depth = normalized_box.copy()
            box_with_depth["depth"] = depth_val
            all_customer_hits.append(box_with_depth)
            
            logger.debug(
                f"Person depth={depth_val:.2f} vs range=[{matched_depth_min}, {matched_depth_max}]"
            )

            if matched_depth_min <= depth_val <= matched_depth_max:
                # This person is at the counter distance → potential customer
                if STAFF_CUSTOMER_CLASSIFICATION_URL:
                    candidate_data = {
                        "box": normalized_box,
                        "bbox": [x1, y1, x2, y2],
                        "depth": depth_val,
                    }
                    customer_detections_candidates.append(candidate_data)
                else:
                    normalized_box["depth"] = depth_val
                    customer_detections.append(normalized_box)
            else:
                logger.debug(
                    f"Person depth {depth_val:.2f} outside [{matched_depth_min}, {matched_depth_max}] – ignored."
                )

        # ------------------------------------------------------------------ #
        # Batch visual classification for depth-filtered customer candidates  #
        # ------------------------------------------------------------------ #
        if customer_detections_candidates and STAFF_CUSTOMER_CLASSIFICATION_URL:
            customer_bboxes = [c["bbox"] for c in customer_detections_candidates]
            labels = self.classify_objects_batch(frame_url, customer_bboxes)

            for candidate, label in zip(customer_detections_candidates, labels):
                if label == "customer" or label is None:
                    det = candidate["box"]
                    det["depth"] = candidate["depth"]
                    customer_detections.append(det)
                else:
                    logger.info(
                        f"Depth-candidate classified as '{label}' – excluded from customers."
                    )

        # Every person inside the staff region is classified as staff (unchanged rule)
        staff_detections.extend(staff_candidates)

        return (
            staff_detections,
            staff_pixel_polygons_list,
            customer_detections,
            customer_pixel_polygons_list,
            customer_landmarks,
            all_customer_hits,
        )

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) of two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        x_inter_min = max(x1_min, x2_min)
        y_inter_min = max(y1_min, y2_min)
        x_inter_max = min(x1_max, x2_max)
        y_inter_max = min(y1_max, y2_max)

        if x_inter_max < x_inter_min or y_inter_max < y_inter_min:
            return 0.0

        intersection = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def apply_nms(self, detections, iou_threshold=0.5):
        """Apply Non-Maximum Suppression to detections."""
        if not detections:
            return []

        # Sort by confidence descending
        detections = sorted(
            detections, key=lambda x: x.get("confidence", 0), reverse=True
        )

        keep = []
        while detections:
            current = detections.pop(0)
            keep.append(current)

            non_overlapping = []
            for det in detections:
                iou = self.calculate_iou(current["bbox"], det["bbox"])
                if iou < iou_threshold:
                    non_overlapping.append(det)
            detections = non_overlapping

        return keep

    def connect_redis(self, host, port, retry_delay=5):
        while True:
            try:
                r = redis.Redis(host=host, port=port, decode_responses=False)
                # Test connection
                r.ping()
                logging.info("Connected to Redis successfully.")
                return r

            except Exception as e:
                logging.error(
                    f"Redis connection failed: {e}. Retrying in {retry_delay} seconds..."
                )
                time.sleep(retry_delay)

    def process_message(self, message: dict):
        """Process a single Redis message.
        Expected format: {"frame_url": "...", "camera_id": "...", "ts": "..."}
        """

        service_reading_time = time.time()
        # print(message)
        message = {k.decode(): v.decode() for k, v in message.items()}
        frame_url = message["frame_url"]
        camera_id = message["camera_id"]
        timestamp = message["timestamp"]
        frame_shape = ast.literal_eval(message["frame_shape"])
        if not frame_url:
            logger.warning("Message missing 'frame_url'. Skipping.")
            return

        # Send image URL to remote YOLO service
        try:
            payload = {"image_url": frame_url}
            yolo_start_time = time.time()
            resp = requests.post(self.yolo_endpoint, json=payload, timeout=20)
            yolo_end_time = time.time()
            yolo_latency = yolo_end_time - yolo_start_time
            # print(resp)
            resp.raise_for_status()
            yolo_result = resp.json()
        except Exception as e:
            logger.error(f"Failed to get YOLO inference for {frame_url}: {e}")
            return

        # The YOLO service returns a dict with a 'predict' key containing a list with a JSON string.
        # Example: {'predict': ['[{"name":"person","class":0,"confidence":0.9,"box":{"x1":10,"y1":20,"x2":30,"y2":40}}]']}
        # Parse this format to extract bounding boxes.
        detections = []
        try:
            predict_list = yolo_result.get("predict", [])
            if predict_list:
                # The first element is a JSON string of a list of detections.
                detections_json = predict_list[0]
                import json as _json

                parsed = _json.loads(detections_json)
                for item in parsed:
                    box = item.get("box", {})
                    clas_ = item.get("class", -1)
                    bbox = [box.get("x1"), box.get("y1"), box.get("x2"), box.get("y2")]
                    conf = item.get("confidence", 0.0)
                    # Filter for Person class (0) only
                    if clas_ == 0 and all(v is not None for v in bbox) and conf >= CONFIDENCE_THRESHOLD:
                        detections.append(
                            {"class": clas_, "bbox": bbox, "confidence": conf}
                        )
        except Exception as parse_err:
            logger.error(f"Failed to parse YOLO result: {parse_err}")

        # Apply NMS to remove duplicates
        detections = self.apply_nms(detections, iou_threshold=NMS_THRESHOLD)

        # Download frame for processing
        frame_img = self.preprocessing(frame_url)

        (
            staff_detection,
            staff_pixel_polygons,
            customer_detection,
            customer_pixel_polygons,
            customer_landmarks,
            all_customer_hits,
        ) = self.post_processing(detections, frame_shape, frame_img=frame_img, frame_url=frame_url)

        current_data = {
            "camera_id": camera_id,
            "timestamp": timestamp,
            "staff_detection": staff_detection,
            "customer_detection": customer_detection,
            "staff_polygons": staff_pixel_polygons,  # Add polygons here
            "customer_polygons": customer_pixel_polygons,
            "frame_url": frame_url,
        }
        self.history[camera_id].append(current_data)

        # Check if any frame in history has valid detections
        is_staff_detected = any(
            len(d["staff_detection"]) > 0 for d in self.history[camera_id]
        )

        # Customer detected if present in more than half of the history frames (Majority Vote / Mode)
        customer_frames_count = sum(
            1
            for d in self.history[camera_id]
            if len(d.get("customer_detection", [])) > 0
        )
        is_customer_detected = customer_frames_count > (
            len(self.history[camera_id]) / 2
        )

        # Unattended Alert Logic (Staff Absent AND Customer Present)
        is_unattended = (not is_staff_detected) and is_customer_detected

        if is_unattended and not self.last_unattended_state[camera_id]:
            try:
                # Optimized Alert Image Selection:
                # Find the most recent frame in history where customer count == mode_val
                # This ensures we send a representative image, not a glitchy one.
                target_frame_url = frame_url  # Default to current

                # Calculate mode if not already (it might be 0 if history empty, but here we have history)
                current_mode = 0
                if self.history[camera_id]:
                    counts = [
                        len(d.get("customer_detection", []))
                        for d in self.history[camera_id]
                    ]
                    try:
                        current_mode = statistics.mode(counts)
                    except:
                        current_mode = (
                            max(set(counts), key=counts.count) if counts else 0
                        )

                # Search backwards for a frame matching the mode
                for d in reversed(self.history[camera_id]):
                    if len(d.get("customer_detection", [])) == current_mode and d.get(
                        "frame_url"
                    ):
                        target_frame_url = d["frame_url"]
                        logger.info(
                            f"Selected representative frame for alert. Mode: {current_mode}"
                        )
                        break

                # Download and encode the frame
                img_response = requests.get(target_frame_url, timeout=10)
                img_response.raise_for_status()

                # Decode to CV2 image
                img_array = np.frombuffer(img_response.content, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                # Annotate the frame
                img = self.annotate_frame(
                    img,
                    staff_pixel_polygons,
                    customer_pixel_polygons,
                    customer_line_points,
                    staff_detection,
                    customer_detection,
                    all_customer_hits,
                    customer_landmarks,
                )

                # Encode back to jpg then base64
                _, buffer = cv2.imencode(".jpg", img)
                frame_b64 = base64.b64encode(buffer).decode("utf-8")

                alert_id = str(uuid.uuid4())
                self.current_unattended_alert_id[camera_id] = alert_id

                dump_payload = {
                    "camera_id": camera_id,
                    "timestamp": timestamp,
                    "frame": frame_b64,
                    "description": "Customer waiting unattended at counter.",
                    "target_collection": "temp_alert",
                    "type": "raised",
                    "ticket_id": alert_id,
                }
                self.redis.xadd(OUTPUT_QUEUE_1, {"data": json.dumps(dump_payload)})
                # Persist state
                self.redis.set(f"alert_state:unattended:{camera_id}", alert_id)
            except Exception as e:
                logger.error(f"Failed to process/push unattended alert dump: {e}")

        # Resolve alert if is_unattended becomes False (Staff returned OR Customer left)
        elif not is_unattended and self.last_unattended_state[camera_id]:
            try:
                # Smart Resolution Frame Selection: Choose the best proof
                target_frame_url = frame_url  # Default fallback

                if is_staff_detected:
                    # Reason: Staff returned. Find most recent frame WITH staff.
                    for d in reversed(self.history[camera_id]):
                        if len(d.get("staff_detection", [])) > 0 and d.get("frame_url"):
                            target_frame_url = d["frame_url"]
                            logger.info(
                                f"Resolution proof: Selected frame with staff present."
                            )
                            break
                elif not is_customer_detected:
                    # Reason: Customer left. Find most recent frame with NO customers.
                    for d in reversed(self.history[camera_id]):
                        if len(d.get("customer_detection", [])) == 0 and d.get(
                            "frame_url"
                        ):
                            target_frame_url = d["frame_url"]
                            logger.info(
                                f"Resolution proof: Selected frame with 0 customers."
                            )
                            break

                # Download and encode the frame
                img_response = requests.get(target_frame_url, timeout=10)
                img_response.raise_for_status()

                if len(img_response.content) == 0:
                    raise ValueError("Empty image content")

                # Decode to CV2 image
                img_array = np.frombuffer(img_response.content, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                # Annotate the frame
                img = self.annotate_frame(
                    img,
                    staff_pixel_polygons,
                    customer_pixel_polygons,
                    customer_line_points,
                    staff_detection,
                    customer_detection,
                    all_customer_hits,
                    customer_landmarks,
                )

                # Encode back to jpg then base64
                _, buffer = cv2.imencode(".jpg", img)
                frame_b64 = base64.b64encode(buffer).decode("utf-8")

                alert_id = self.current_unattended_alert_id[camera_id]

                resolve_payload = {
                    "camera_id": camera_id,
                    "timestamp": timestamp,
                    "frame": frame_b64,
                    "description": "Unattended customer situation resolved (Staff returned or customer left).",
                    "target_collection": "temp_alert",
                    "type": "resolved",
                    "ticket_id": alert_id,
                }
                self.redis.xadd(OUTPUT_QUEUE_1, {"data": json.dumps(resolve_payload)})
                # Clear persisted state
                self.redis.delete(f"alert_state:unattended:{camera_id}")
            except Exception as e:
                logger.error(f"Failed to process/push unattended resolve dump: {e}")

        # High Customer Traffic Alert Logic (Customers > 3 in majority of history)
        high_traffic_frames_count = sum(
            1
            for d in self.history[camera_id]
            if len(d.get("customer_detection", [])) > 3
        )
        is_high_traffic = high_traffic_frames_count > (
            len(self.history[camera_id]) / 2
        )

        if is_high_traffic and not self.last_high_traffic_state[camera_id]:
            try:
                # Find representative frame (one with > 3 customers)
                target_frame_url = frame_url
                for d in reversed(self.history[camera_id]):
                    if len(d.get("customer_detection", [])) > 3 and d.get("frame_url"):
                        target_frame_url = d["frame_url"]
                        logger.info("Selected representative frame for High Traffic alert.")
                        break

                img_response = requests.get(target_frame_url, timeout=10)
                img_response.raise_for_status()
                if len(img_response.content) == 0:
                    raise ValueError("Empty image content")

                img_array = np.frombuffer(img_response.content, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                # Annotate
                img = self.annotate_frame(
                    img,
                    staff_pixel_polygons,
                    customer_pixel_polygons,
                    customer_line_points,
                    staff_detection,
                    customer_detection,
                    all_customer_hits,
                    customer_landmarks,
                )
                _, buffer = cv2.imencode(".jpg", img)
                frame_b64 = base64.b64encode(buffer).decode("utf-8")

                alert_id = str(uuid.uuid4())
                self.current_high_traffic_alert_id[camera_id] = alert_id

                dump_payload = {
                    "camera_id": camera_id,
                    "timestamp": timestamp,
                    "frame": frame_b64,
                    "description": "High customer traffic detected (More than 3 customers).",
                    "target_collection": "temp_alert",
                    "type": "raised",
                    "ticket_id": alert_id,
                }
                self.redis.xadd(OUTPUT_QUEUE_1, {"data": json.dumps(dump_payload)})
                # Persist state
                self.redis.set(f"alert_state:high_traffic:{camera_id}", alert_id)
            except Exception as e:
                logger.error(f"Failed to process/push High Traffic alert dump: {e}")

        elif not is_high_traffic and self.last_high_traffic_state[camera_id]:
            try:
                # Find representative frame (one with <= 3 customers)
                target_frame_url = frame_url
                for d in reversed(self.history[camera_id]):
                    if len(d.get("customer_detection", [])) <= 3 and d.get("frame_url"):
                        target_frame_url = d["frame_url"]
                        logger.info("Resolution proof: Selected frame with <= 3 customers.")
                        break

                img_response = requests.get(target_frame_url, timeout=10)
                img_response.raise_for_status()
                if len(img_response.content) == 0:
                    raise ValueError("Empty image content")

                img_array = np.frombuffer(img_response.content, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                img = self.annotate_frame(
                    img,
                    staff_pixel_polygons,
                    customer_pixel_polygons,
                    customer_line_points,
                    staff_detection,
                    customer_detection,
                    all_customer_hits,
                    customer_landmarks,
                )
                _, buffer = cv2.imencode(".jpg", img)
                frame_b64 = base64.b64encode(buffer).decode("utf-8")

                alert_id = self.current_high_traffic_alert_id[camera_id]

                resolve_payload = {
                    "camera_id": camera_id,
                    "timestamp": timestamp,
                    "frame": frame_b64,
                    "description": "High customer traffic resolved.",
                    "target_collection": "temp_alert",
                    "type": "resolved",
                    "ticket_id": alert_id,
                }
                self.redis.xadd(OUTPUT_QUEUE_1, {"data": json.dumps(resolve_payload)})
                # Clear persisted state
                self.redis.delete(f"alert_state:high_traffic:{camera_id}")
            except Exception as e:
                logger.error(f"Failed to process/push High Traffic resolve dump: {e}")

        # Calculate Statistics
        history_counts = [
            len(d.get("customer_detection", [])) for d in self.history[camera_id]
        ]
        # mean_val = statistics.mean(history_counts)
        median_val = statistics.median(history_counts)
        try:
            mode_val = statistics.mode(history_counts)
        except:
            # statistics.mode raises error if multi-modal in older python versions or empty
            mode_val = (
                max(set(history_counts), key=history_counts.count)
                if history_counts
                else 0
            )

        mean_val = statistics.mean(history_counts) if history_counts else 0
        std_val = statistics.stdev(history_counts) if len(history_counts) > 1 else 0
        min_val = min(history_counts) if history_counts else 0
        max_val = max(history_counts) if history_counts else 0

        # service_output_time = time.time()
        output_payload = {
            "camera_id": camera_id,
            "timestamp": timestamp,
            "valid_detection": staff_detection,
            "customer_detection": customer_detection,
            "is_staff_detected": is_staff_detected,
            "is_customer_detected": is_customer_detected,
            "customer_count": len(customer_detection),
            "customer_count_median": median_val,
            "customer_count_mode": mode_val,
            "customer_count_mean": mean_val,
            "customer_count_std": std_val,
            "customer_count_min": min_val,
            "customer_count_max": max_val,
            "target_collection": os.getenv(
                "TARGET_COLLECTION_NAME", "counter_staff_collection"
            ),
        }

        # Update the state for the next frame
        # Update the state for the next frame
        self.last_staff_detected_state[camera_id] = is_staff_detected
        self.last_unattended_state[camera_id] = is_unattended
        self.last_high_traffic_state[camera_id] = is_high_traffic


        try:
            self.redis.xadd(OUTPUT_QUEUE_1, {"data": json.dumps(output_payload)})

            # Push visualization data if enabled
            if VISUALIZE and hasattr(self, "vis_queue"):
                # Encode frame to base64 for reliable visualization
                frame_b64 = None
                if frame_img is not None:
                    try:
                        _, buffer = cv2.imencode(".jpg", frame_img)
                        frame_b64 = base64.b64encode(buffer).decode("utf-8")
                    except Exception as e:
                        logger.error(f"Failed to encode frame for visualization: {e}")

                vis_payload = {
                    "camera_id": camera_id,
                    "timestamp": timestamp,
                    "frame": frame_b64,  # Send base64 frame
                    "frame_url": frame_url,
                    "staff_polygons": current_data["staff_polygons"],
                    "customer_polygons": current_data["customer_polygons"],
                    "customer_line": [], # customer_line_points,
                    "staff_detection": staff_detection,
                    "customer_detection": customer_detection,
                    "all_customer_hits": all_customer_hits,
                    "is_staff_detected": is_staff_detected,
                    "is_customer_detected": is_customer_detected,
                }
                self.vis_queue.push(json.dumps(vis_payload))
            #     try:
            #         # Download and decode the frame for visualization
            #         # We utilize the already existing helper (or do it again here if frame_img not accessible)
            #         # Ideally we should use self.preprocessing but let's just do requests to be safe/consistent with this block structure
            #         # Actually, we have frame_img from line 337, but let's re-download to be sure or use if passing frame_img is hard.
            #         # WAIT, we already have frame_img from line 337 in process_message scope!
            #         # But line 337: frame_img = self.preprocessing(frame_url)
            #         # Let's reuse frame_img if possible, but if it was None we wouldn't be here?
            #         # Actually logic continues even if frame_img is None (lines 339 returned empty lists).

            #         if frame_img is not None:
            #             viz_img = frame_img.copy()

            #             # Draw Staff Polygons (Red)
            #             for poly in staff_pixel_polygons:
            #                 pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            #                 cv2.polylines(viz_img, [pts], True, (0, 0, 255), 2)

            #             # Draw Customer Line (Cyan)
            #             if customer_line_points:
            #                 p1 = (
            #                     int(customer_line_points[0][0]),
            #                     int(customer_line_points[0][1]),
            #                 )
            #                 p2 = (
            #                     int(customer_line_points[1][0]),
            #                     int(customer_line_points[1][1]),
            #                 )
            #                 cv2.line(viz_img, p1, p2, (255, 255, 0), 2)

            #             # Draw All Raw Detections (White - "Person")
            #             # These are raw pixel coordinates from YOLO
            #             for det in detections:
            #                 bbox = det.get("bbox")
            #                 if bbox:
            #                     x1, y1, x2, y2 = (
            #                         int(bbox[0]),
            #                         int(bbox[1]),
            #                         int(bbox[2]),
            #                         int(bbox[3]),
            #                     )
            #                     # Draw thinner white box to signify "background" detection
            #                     cv2.rectangle(
            #                         viz_img, (x1, y1), (x2, y2), (255, 255, 255), 1
            #                     )
            #                     # Optional: Label with confidence
            #                     conf = det.get("confidence", 0.0)
            #                     cv2.putText(
            #                         viz_img,
            #                         f"P {conf:.2f}",
            #                         (x1, y1 - 5),
            #                         cv2.FONT_HERSHEY_SIMPLEX,
            #                         0.4,
            #                         (255, 255, 255),
            #                         1,
            #                     )

            #             # Draw Staff Detections (Green)
            #             # staff_detection contains normalized detections. Need to scale.
            #             h, w = viz_img.shape[:2]
            #             for det in staff_detection:
            #                 bbox = det.get("bbox")
            #                 x1 = int(bbox[0] * w)
            #                 y1 = int(bbox[1] * h)
            #                 x2 = int(bbox[2] * w)
            #                 y2 = int(bbox[3] * h)
            #                 # Draw thicker green box
            #                 cv2.rectangle(viz_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #                 cv2.putText(
            #                     viz_img,
            #                     "Staff",
            #                     (x1, y1 - 10),
            #                     cv2.FONT_HERSHEY_SIMPLEX,
            #                     0.5,
            #                     (0, 255, 0),
            #                     2,
            #                 )

            #             # Draw Customer Detections (Blue)
            #             for det in customer_detection:
            #                 bbox = det.get("bbox")
            #                 x1 = int(bbox[0] * w)
            #                 y1 = int(bbox[1] * h)
            #                 x2 = int(bbox[2] * w)
            #                 y2 = int(bbox[3] * h)
            #                 # Draw thicker blue box
            #                 cv2.rectangle(viz_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            #                 cv2.putText(
            #                     viz_img,
            #                     "Customer",
            #                     (x1, y1 - 10),
            #                     cv2.FONT_HERSHEY_SIMPLEX,
            #                     0.5,
            #                     (255, 0, 0),
            #                     2,
            #                 )

            #             # Draw Facial Landmarks (Customer Only)
            #             for landmarks in customer_landmarks:
            #                 le = landmarks.get("left_eye")
            #                 re = landmarks.get("right_eye")
            #                 nose = landmarks.get("nose")

            #                 # Eyes: Yellow (0, 255, 255) in BGR
            #                 if le:
            #                     cv2.circle(viz_img, le, 3, (0, 255, 255), -1)
            #                 if re:
            #                     cv2.circle(viz_img, re, 3, (0, 255, 255), -1)

            #                 # Nose: Purple (255, 0, 255) in BGR
            #                 if nose:
            #                     cv2.circle(viz_img, nose, 3, (255, 0, 255), -1)

            #             # Save Image
            #             # Ensure directory exists (it should be mounted, but good practice)
            #             os.makedirs("/workspace/viz_output", exist_ok=True)
            #             filename = f"/workspace/viz_output/{timestamp}.jpg"
            #             cv2.imwrite(filename, viz_img)
            #             # logger.info(f"Saved visualization frame to {filename}")

            # except Exception as e:
            #     logger.error(f"Failed to process/save visualization frame: {e}")

            msg = (
                f"Published detection result for camera {camera_id} at {timestamp} | "
                f"Staff: {is_staff_detected} (Count: {len(staff_detection)}), Customers: {len(customer_detection)}, "
                f"Unattended: {is_unattended} | Total Persons: {len(detections)}"
            )
            print(msg, flush=True)
            logger.info(msg)
        except Exception as e:
            logger.error(f"Failed to push result to output queue: {e}")

    def run(self):
        logger.info("CounterStaffDetector started, listening on input queue.")
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

            except json.JSONDecodeError:
                logger.error("Failed to decode JSON message from input queue.")
            except Exception as e:
                logger.exception(f"Unexpected error in main loop: {e}")
                time.sleep(1)


if __name__ == "__main__":
    service = CounterStaffDetector()
    service.run()
