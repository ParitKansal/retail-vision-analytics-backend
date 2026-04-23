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
import statistics


load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
LOKI_URL = os.getenv("LOKI_URL", "http://localhost:3100/loki/api/v1/push")

INPUT_QUEUE = os.getenv("INPUT_QUEUE_NAME")
OUTPUT_QUEUE_1 = os.getenv("OUTPUT_QUEUE_NAME_1")
YOLO_MODEL_URL = os.getenv("YOLO_MODEL_URL")
STAFF_CUSTOMER_CLASSIFICATION_URL = os.getenv("STAFF_CUSTOMER_CLASSIFICATION_URL")
GROUP_NAME = os.getenv("GROUP_NAME")
CUSTOMER_OVERLAP_THRESHOLD = float(os.getenv("CUSTOMER_OVERLAP_THRESHOLD", "0.2"))
STAFF_OVERLAP_THRESHOLD = float(os.getenv("STAFF_OVERLAP_THRESHOLD", "0.3"))
NMS_THRESHOLD = float(os.getenv("NMS_THRESHOLD", "0.6"))
CAMERA_ID = os.getenv("CAMERA_ID", INPUT_QUEUE.split(":")[-1])
VISUALIZE = str(os.getenv("VISUALIZE", "False")).lower() == "true"
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.2"))
ACK_TTL_SECONDS = 60 * 60  # 1 hour

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
        # self.last_processed_time = 0
        # Load configuration settings
        self.settings = self.load_settings()
        # Redis connection
        self.redis = self.connect_redis(host=REDIS_HOST, port=REDIS_PORT)
        # YOLO endpoint URL is taken from environment variable YOLO_MODEL_URL
        self.yolo_endpoint = YOLO_MODEL_URL
        self.history = defaultdict(lambda: deque(maxlen=15))
        self.last_staff_detected_state = defaultdict(lambda: True)

        self.last_unattended_state = defaultdict(lambda: False)
        self.current_unattended_alert_id = defaultdict(lambda: None)

        self.last_high_traffic_state = defaultdict(lambda: False)
        self.current_high_traffic_alert_id = defaultdict(lambda: None)

        self.last_customer_in_staff_zone_state = defaultdict(lambda: False)
        self.current_customer_in_staff_alert_id = defaultdict(lambda: None)

        # Rate limiting state
        self.last_processed_time = defaultdict(lambda: 0.0)



        # Setup visualization queue if enabled
        if VISUALIZE:
            self.vis_queue = FixedSizeQueue(
                self.redis, f"stream:visualization:counter_staff:{CAMERA_ID}", 10
            )
        
        # Load persisted alert states
        self.load_alert_state()

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

            # Customer In Staff Zone
            cis_key = f"alert_state:cust_in_staff:{CAMERA_ID}"
            cis_id = self.redis.get(cis_key)
            if cis_id:
                self.last_customer_in_staff_zone_state[CAMERA_ID] = True
                self.current_customer_in_staff_alert_id[CAMERA_ID] = cis_id.decode()
                logger.info(f"Restored CustInStaff Alert state for {CAMERA_ID}: {cis_id.decode()}")

        except Exception as e:
            logger.error(f"Failed to load alert state from Redis: {e}")

    def draw_polygons(self, img, polygons, label, color, thickness=2):
        """
        Draw polygons on the image.
        """
        height, width = img.shape[:2]

        for idx, polygon in enumerate(polygons):
            # Convert normalized coordinates to pixel coordinates if needed
            points = []
            # Check if polygon is a list of points or list of dicts or numpy array
            # In service.py post_processing, we have staff_pixel_polygons (list of np.array)
            # and staff_pixel_polygons_list (list of list of [x,y]).
            # The vizualize.py handles list of dicts (normalized) or list of list (pixels).
            # Here we expect to pass the pixel polygons directly from post_processing.

            # If it's a numpy array, we can use it directly
            if isinstance(polygon, np.ndarray):
                pts = polygon
                points = polygon.tolist()
            elif isinstance(polygon, list):
                # Check format
                if len(polygon) > 0 and isinstance(polygon[0], dict):
                    # Normalized dicts
                    for pt in polygon:
                        x = int(pt["x"] * width)
                        y = int(pt["y"] * height)
                        points.append([x, y])
                    pts = np.array(points, dtype=np.int32)
                else:
                    # Assumed pixel coords
                    points = polygon
                    pts = np.array(points, dtype=np.int32)
            else:
                continue

            # Draw polygon outline
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)

            # Draw semi-transparent fill
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)

            # Add label at the first point
            if len(points) > 0:
                first_pt = points[0]
                label_text = f"{label} {idx + 1}"
                cv2.putText(
                    img,
                    label_text,
                    (first_pt[0], first_pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

    def draw_unified_detections(self, img, detections):
        """
        Draw all person detections with a single color and generic label (used for high-traffic alerts).
        """
        COLOR_PERSON = (0, 165, 255)  # Orange
        height, width = img.shape[:2]

        for idx, detection in enumerate(detections):
            bbox = detection["bbox"]
            if all(0 <= coord <= 1 for coord in bbox):
                x1 = int(bbox[0] * width)
                y1 = int(bbox[1] * height)
                x2 = int(bbox[2] * width)
                y2 = int(bbox[3] * height)
            else:
                x1, y1, x2, y2 = [int(c) for c in bbox]

            cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_PERSON, 2)
            label = f"Person {idx + 1}"
            cv2.putText(
                img,
                label,
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                COLOR_PERSON,
                2,
            )

    def draw_staff_detections(self, img, staff_detections):
        """
        Draw staff detection bounding boxes in Green.
        """
        COLOR_STAFF = (0, 255, 0)  # Green
        height, width = img.shape[:2]

        for idx, detection in enumerate(staff_detections):
            bbox = detection["bbox"]

            # Check if bbox is normalized (values between 0 and 1)
            # Detections from service.py post_processing are normalized.
            if all(0 <= coord <= 1 for coord in bbox):
                # Denormalize
                x1 = int(bbox[0] * width)
                y1 = int(bbox[1] * height)
                x2 = int(bbox[2] * width)
                y2 = int(bbox[3] * height)
            else:
                # Already pixel coordinates
                x1, y1, x2, y2 = [int(c) for c in bbox]

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_STAFF, 2)

            # Get confidence if available
            conf = detection.get("confidence", 0)
            label = f"Staff {idx + 1}"
            if conf > 0:
                label += f" ({conf:.2f})"

            cv2.putText(
                img,
                label,
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                COLOR_STAFF,
                2,
            )

    def draw_customer_detections(self, img, customer_detections):
        """
        Draw customer detection bounding boxes in Orange.
        """
        COLOR_CUSTOMER = (0, 165, 255)  # Orange
        height, width = img.shape[:2]

        for idx, detection in enumerate(customer_detections):
            bbox = detection["bbox"]

            # Check if bbox is normalized
            if all(0 <= coord <= 1 for coord in bbox):
                # Denormalize
                x1 = int(bbox[0] * width)
                y1 = int(bbox[1] * height)
                x2 = int(bbox[2] * width)
                y2 = int(bbox[3] * height)
            else:
                # Already pixel coordinates
                x1, y1, x2, y2 = [int(c) for c in bbox]

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_CUSTOMER, 2)

            # Get confidence if available
            conf = detection.get("confidence", 0)
            label = f"Customer {idx + 1}"
            if conf > 0:
                label += f" ({conf:.2f})"

            cv2.putText(
                img,
                label,
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                COLOR_CUSTOMER,
                2,
            )

    def draw_staff_in_customer_detections(self, img, staff_in_customer_detections):
        """
        Draw detections that were in customer zone but classified as staff in Grey.
        """
        COLOR_GREY = (255, 255, 255)  # White
        height, width = img.shape[:2]

        for idx, detection in enumerate(staff_in_customer_detections):
            bbox = detection["bbox"]
            if all(0 <= coord <= 1 for coord in bbox):
                x1 = int(bbox[0] * width)
                y1 = int(bbox[1] * height)
                x2 = int(bbox[2] * width)
                y2 = int(bbox[3] * height)
            else:
                x1, y1, x2, y2 = [int(c) for c in bbox]

            cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_GREY, 2)
            label = f"Staff (in Cust Zone) {idx + 1}"
            cv2.putText(
                img,
                label,
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                COLOR_GREY,
                2,
            )

    def draw_customer_in_staff_detections(self, img, customer_in_staff_detections):
        """
        Draw customer in staff detection bounding boxes in Red.
        """
        COLOR_RED = (0, 0, 255)  # Red
        height, width = img.shape[:2]

        for idx, detection in enumerate(customer_in_staff_detections):
            bbox = detection["bbox"]
            if all(0 <= coord <= 1 for coord in bbox):
                x1 = int(bbox[0] * width)
                y1 = int(bbox[1] * height)
                x2 = int(bbox[2] * width)
                y2 = int(bbox[3] * height)
            else:
                x1, y1, x2, y2 = [int(c) for c in bbox]

            cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_RED, 2)
            label = f"CUST IN STAFF ZONE {idx + 1}"
            cv2.putText(
                img,
                label,
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                COLOR_RED,
                2,
            )

    def annotate_frame(
        self,
        img,
        staff_polygons,
        customer_polygons,
        staff_detections,
        customer_detections,
        staff_in_customer_detections=None,
        customer_in_staff_detections=None,
        unified=False,
    ):
        """
        Annotate the frame with all visual information.
        When unified=True, all person detections are drawn with the same color and
        label (no staff/customer distinction) — used for high-traffic alerts.
        """
        if staff_in_customer_detections is None:
            staff_in_customer_detections = []
        # Draw staff polygons (Green)
        if staff_polygons:
            self.draw_polygons(img, staff_polygons, "Staff Zone", (0, 255, 0))

        # Draw customer polygons (Orange)
        if customer_polygons:
            self.draw_polygons(img, customer_polygons, "Customer Zone", (0, 165, 255))

        if unified:
            all_persons = list(staff_detections or []) + list(customer_detections or []) + list(staff_in_customer_detections or [])
            if all_persons:
                self.draw_unified_detections(img, all_persons)
        else:
            # Draw staff detections
            if staff_detections:
                self.draw_staff_detections(img, staff_detections)

            # Draw customer detections
            if customer_detections:
                self.draw_customer_detections(img, customer_detections)

            # Draw staff found in customer zone
            if staff_in_customer_detections:
                self.draw_staff_in_customer_detections(img, staff_in_customer_detections)

        # Draw customers found in staff zone (violation)
        # if customer_in_staff_detections:
        #     self.draw_customer_in_staff_detections(img, customer_in_staff_detections)

        return img

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
            return None

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
            # LitServe returns a list of results for each request in the batch.
            # Since we send one request, we need the first (and only) result list.
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
                result = result[0]
            
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

    # The YOLO model is served remotely; inference is performed via HTTP request.
    # No local model loading is required.

    def preprocessing(self, frame):
        """
        Preprocess the frame before detection.
        Currently just returns the frame as is.
        """
        return frame

    def post_processing(self, results, frame_shape, frame_img=None, frame_url=None):
        height, width = frame_shape[:2]
        if height == 0 and width == 0:
            logger.error(f"height = {height} width = {width}")
            raise ValueError(f"height = {height} width = {width}")

        if self.settings is None:
            logger.error("No settings found for this camera. Returning empty detections.")
            return ([], [], [], [], [], [])

        staff_polygons_config = self.settings.get("staff_polygons", [])
        customer_polygons_config = self.settings.get("customer_polygons", [])
        customer_in_staff_polygons_config = self.settings.get("customer_in_staff_polygons", [])

        # Convert polygons to pixel coordinates
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

        # Create a single mask for all staff polygons
        staff_mask = np.zeros((height, width), dtype=np.uint8)
        if staff_pixel_polygons:
            cv2.fillPoly(staff_mask, staff_pixel_polygons, 255)

        # Convert customer polygons to pixel coordinates
        customer_pixel_polygons = []
        customer_pixel_polygons_list = []
        for poly in customer_polygons_config:
            normalized_points = poly.get("normalized", [])
            points = []
            for pt in normalized_points:
                x = int(pt["x"] * width)
                y = int(pt["y"] * height)
                points.append([x, y])
            if points:
                customer_pixel_polygons.append(np.array(points, dtype=np.int32))
                customer_pixel_polygons_list.append(points)

        # Create a single mask for all customer polygons to handle overlaps efficiently
        customer_mask = np.zeros((height, width), dtype=np.uint8)
        if customer_pixel_polygons:
            cv2.fillPoly(customer_mask, customer_pixel_polygons, 255)

        # Convert customer_in_staff_polygons to pixel coordinates
        customer_in_staff_pixel_polygons = []
        for poly in customer_in_staff_polygons_config:
            normalized_points = poly.get("normalized", [])
            points = []
            for pt in normalized_points:
                x = int(pt["x"] * width)
                y = int(pt["y"] * height)
                points.append([x, y])
            if points:
                customer_in_staff_pixel_polygons.append(np.array(points, dtype=np.int32))

        staff_candidates = [] # Potentially staff, need verification if enabled
        staff_detections = []
        customer_candidates = []  # Store potential customers for classification
        customer_detections = []
        staff_in_customer_detections = []  # Staff found in customer region
        customer_in_staff_detections = [] # Customer found in staff region

        # results[0] is the result for the single frame
        for box in results:
            # Check if class is person (class 0 for COCO)
            if int(box.get("class")) != 0:
                continue

            x1, y1, x2, y2 = box.get("bbox")
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            center_point = (center_x, center_y)

            # Create normalized box for output
            normalized_bbox = [x1 / width, y1 / height, x2 / width, y2 / height]
            normalized_box = box.copy()
            normalized_box["bbox"] = normalized_bbox

            # --- Check Customer Condition First (Priority) ---
            is_inside_customer = False
            for poly in customer_pixel_polygons:
                if cv2.pointPolygonTest(poly, center_point, False) >= 0:
                    is_inside_customer = True
                    break

            if not is_inside_customer:
                # Cast coordinates to int for slicing
                ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
                # Clip to frame image limits
                ix1, iy1 = max(0, ix1), max(0, iy1)
                ix2, iy2 = min(width, ix2), min(height, iy2)

                if ix2 > ix1 and iy2 > iy1:
                    box_area = (x2 - x1) * (y2 - y1)
                    if box_area > 0:
                        roi = customer_mask[iy1:iy2, ix1:ix2]
                        intersection_area = cv2.countNonZero(roi)
                        overlap_ratio = intersection_area / box_area

                        if overlap_ratio >= CUSTOMER_OVERLAP_THRESHOLD:
                            is_inside_customer = True

            if is_inside_customer:
                # Apply confidence threshold only for customers
                if box.get("confidence", 0.0) >= CONFIDENCE_THRESHOLD:
                    customer_candidates.append(normalized_box)
                    # If identified as customer (or candidate), DO NOT check for staff.
                    # Priority: Customer > Staff
                    continue
                # If it didn't meet customer threshold, we still allow checking if it's staff
                # because staff detection currently allows "any score".
                # To do that, we don't 'continue' here.
            # --- Check Staff Condition Only If Not Customer ---
            is_inside_staff = False
            for poly in staff_pixel_polygons:
                # measureDist=False returns +1 if inside, -1 if outside, 0 if on edge
                if cv2.pointPolygonTest(poly, center_point, False) >= 0:
                    is_inside_staff = True
                    break

            if not is_inside_staff:
                # Cast coordinates to int for slicing
                ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
                # Clip to frame image limits
                ix1, iy1 = max(0, ix1), max(0, iy1)
                ix2, iy2 = min(width, ix2), min(height, iy2)

                if ix2 > ix1 and iy2 > iy1:
                    box_area = (x2 - x1) * (y2 - y1)
                    if box_area > 0:
                        roi = staff_mask[iy1:iy2, ix1:ix2]
                        intersection_area = cv2.countNonZero(roi)
                        overlap_ratio = intersection_area / box_area

                        if overlap_ratio >= STAFF_OVERLAP_THRESHOLD:
                            is_inside_staff = True

            if is_inside_staff:
                # staff_detections.append(normalized_box)
                staff_candidates.append(normalized_box)

        # --- Integration: Classify Potential Customers AND Staff ---
        # Use batch classification for better performance
        if (
            (customer_candidates or staff_candidates)
            and frame_url is not None
            and STAFF_CUSTOMER_CLASSIFICATION_URL
        ):
            # Collect all bboxes and their metadata
            all_bboxes = []
            bbox_metadata = []  # Track which zone each bbox belongs to
            
            # Add customer candidates
            for candidate in customer_candidates:
                bbox = candidate["bbox"]  # [x1/w, y1/h, x2/w, y2/h]
                x1 = int(bbox[0] * width)
                y1 = int(bbox[1] * height)
                x2 = int(bbox[2] * width)
                y2 = int(bbox[3] * height)
                
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                
                if x2 > x1 and y2 > y1:
                    absolute_bbox = [x1, y1, x2, y2]
                    all_bboxes.append(absolute_bbox)
                    bbox_metadata.append({"candidate": candidate, "zone_type": "customer_zone"})
                else:
                    logger.warning(f"Invalid bbox dims: {x1, y1, x2, y2}")
            
            # Add staff candidates
            if staff_candidates:
                for candidate in staff_candidates:
                    bbox = candidate["bbox"]
                    x1 = int(bbox[0] * width)
                    y1 = int(bbox[1] * height)
                    x2 = int(bbox[2] * width)
                    y2 = int(bbox[3] * height)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    
                    if x2 > x1 and y2 > y1:
                        absolute_bbox = [x1, y1, x2, y2]
                        all_bboxes.append(absolute_bbox)
                        bbox_metadata.append({"candidate": candidate, "zone_type": "staff_zone"})
            
            # Make single batch classification request
            if all_bboxes:
                labels = self.classify_objects_batch(frame_url, all_bboxes)
                
                # Process results
                for idx, (metadata, label) in enumerate(zip(bbox_metadata, labels)):
                    candidate = metadata["candidate"]
                    zone_type = metadata["zone_type"]
                    
                    if zone_type == "customer_zone":
                        # Filter: Only keep if label is 'customer' or None (fallback)
                        if label == "customer" or label is None:
                            candidate["classification"] = label or "customer_fallback"
                            customer_detections.append(candidate)
                        elif label == "staff":
                            candidate["classification"] = label
                            staff_in_customer_detections.append(candidate)
                            logger.info(
                                f"Filtered out candidate classified as staff in customer zone."
                            )
                        else:
                            logger.info(f"Filtered out candidate classified as {label}")
                    
                    elif zone_type == "staff_zone":
                        if label == "staff":
                            candidate["classification"] = label
                            staff_detections.append(candidate)
                        elif label == "customer":
                            # Check if strict polygon is defined for this violation
                            is_violation = True
                            if customer_in_staff_pixel_polygons:
                                cx = int((candidate["bbox"][0] + candidate["bbox"][2]) * width / 2)
                                cy = int((candidate["bbox"][1] + candidate["bbox"][3]) * height / 2)
                                center_pt = (cx, cy)
                                
                                is_inside_strict = False
                                for poly in customer_in_staff_pixel_polygons:
                                    if cv2.pointPolygonTest(poly, center_pt, False) >= 0:
                                        is_inside_strict = True
                                        break
                                
                                if not is_inside_strict:
                                    is_violation = False
                                    logger.info("Customer in Staff Zone violation IGNORED (outside strict polygon).")

                            if is_violation:
                                candidate["classification"] = label
                                customer_in_staff_detections.append(candidate)
                                logger.warning("Customer detected in Staff Zone!")
                        else:
                            # Assume staff for unknown labels
                            staff_detections.append(candidate)
        else:
            # If no frame image or service not configured, pass all candidates through
            customer_detections = customer_candidates
            staff_detections = staff_candidates
            if frame_img is None and (customer_candidates or staff_candidates):
                logger.warning(
                    "Frame image not provided to post_processing, skipping classification."
                )

        return (
            staff_detections,
            staff_pixel_polygons_list,
            customer_detections,
            customer_pixel_polygons_list,
            staff_in_customer_detections,
            customer_in_staff_detections,
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
#        logger.info(f"Incoming Payload: {message}")
        frame_url = message["frame_url"]
        camera_id = message["camera_id"]
        timestamp = message["timestamp"]
        frame_shape = ast.literal_eval(message["frame_shape"])

        # Minimal code to limit FPS to 1
        current_time = time.time()
        if (current_time - self.last_processed_time[camera_id]) < 1.0:
            logger.info(f"Skipping frame for {camera_id} to maintain 1 FPS")
            return
        self.last_processed_time[camera_id] = current_time
        if not frame_url:
            logger.warning("Message missing 'frame_url'. Skipping.")
            return

        # Send image URL to remote YOLO service
        try:
            payload = {"image_url": frame_url}
            yolo_start_time = time.time()
            resp = requests.post(self.yolo_endpoint, json=payload, timeout=20)
            yolo_end_time = time.time()
            yolo_duration = yolo_end_time - yolo_start_time
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
                parsed = json.loads(detections_json)
                for item in parsed:
                    box = item.get("box", {})
                    clas_ = item.get("class", -1)
                    bbox = [box.get("x1"), box.get("y1"), box.get("x2"), box.get("y2")]
                    if clas_ == 0 and all(v is not None for v in bbox):
                        detections.append(
                            {
                                "class": clas_,
                                "bbox": bbox,
                                "confidence": item.get("confidence", 0.0),
                            }
                        )
        except Exception as parse_err:
            logger.error(f"Failed to parse YOLO result: {parse_err}")

        # Apply NMS to remove duplicates
        detections = self.apply_nms(detections, iou_threshold=NMS_THRESHOLD)

        frame_img = None 
        # Image download removed as classification service now handles it via URL.

        post_start_time = time.time()
        (
            staff_detection,
            staff_pixel_polygons,
            customer_detection,
            customer_pixel_polygons_list,
            staff_in_customer_detections,
            customer_in_staff_detections,
        ) = self.post_processing(detections, frame_shape, frame_img=None, frame_url=frame_url)
        post_end_time = time.time()
        post_duration = post_end_time - post_start_time

        current_data = {
            "camera_id": camera_id,
            "timestamp": timestamp,
            "staff_detection": staff_detection,
            "customer_detection": customer_detection,
            "staff_in_customer_detections": staff_in_customer_detections,
            "customer_in_staff_detections": customer_in_staff_detections,
            "frame_url": frame_url,  # Store frame URL for better alert image selection
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
                repr_staff_detection = staff_detection
                repr_customer_detection = customer_detection
                repr_staff_in_customer_detections = staff_in_customer_detections
                repr_customer_in_staff_detections = customer_in_staff_detections

                # Calculate mode of NON-ZERO customer counts only.
                # Using all counts (including zeros) can yield mode=0 even when
                # the majority vote says customers ARE present, because the
                # non-zero counts are split across different values (1,2,3..)
                # while 0 remains the single most frequent value.
                current_mode = 1  # Safe default: at least 1 customer
                if self.history[camera_id]:
                    non_zero_counts = [
                        len(d.get("customer_detection", []))
                        for d in self.history[camera_id]
                        if len(d.get("customer_detection", [])) > 0
                    ]
                    if non_zero_counts:
                        try:
                            current_mode = statistics.mode(non_zero_counts)
                        except Exception:
                            current_mode = (
                                max(set(non_zero_counts), key=non_zero_counts.count)
                            )

                # Search backwards for a frame matching the mode
                for d in reversed(self.history[camera_id]):
                    if len(d.get("customer_detection", [])) == current_mode and d.get(
                        "frame_url"
                    ):
                        target_frame_url = d["frame_url"]
                        repr_staff_detection = d.get("staff_detection", [])
                        repr_customer_detection = d.get("customer_detection", [])
                        repr_staff_in_customer_detections = d.get("staff_in_customer_detections", [])
                        repr_customer_in_staff_detections = d.get("customer_in_staff_detections", [])
                        logger.info(
                            f"Selected representative frame for alert. Mode: {current_mode}"
                        )
                        break

                # Download and encode the frame (resizing to 25% of original pixel area)
                img_response = requests.get(target_frame_url, timeout=10)
                img_response.raise_for_status()
                # Save full resolution image
                if len(img_response.content) == 0:
                    raise ValueError("Empty image content")

                # Decode to CV2 image
                img_array = np.frombuffer(img_response.content, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                # Encode RAW (without visualization)
                _, raw_buffer = cv2.imencode(".jpg", img)
                raw_frame_b64 = base64.b64encode(raw_buffer).decode("utf-8")

                # Annotate the frame
                img = self.annotate_frame(
                    img,
                    staff_pixel_polygons,
                    customer_pixel_polygons_list,
                    repr_staff_detection,
                    repr_customer_detection,
                    repr_staff_in_customer_detections,
                    repr_customer_in_staff_detections,
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
                    "frame_without_visualization": raw_frame_b64,
                    "description": "Customer waiting unattended at counter.",
                    "target_collection": "alert_collection",
                    "type": "raised",
                    "ticket_id": alert_id,
                }
                self.redis.xadd(OUTPUT_QUEUE_1, {"data": json.dumps(dump_payload)})
                logger.warning(f"ALERT RAISED: Unattended customer at counter {camera_id}. Ticket: {alert_id}")
                # Persist state
                self.redis.set(f"alert_state:unattended:{camera_id}", alert_id)
                # Update state only on success
                self.last_unattended_state[camera_id] = True
            except Exception as e:
                logger.error(f"Failed to process/push unattended alert dump: {e}")

        # Resolve alert if is_unattended becomes False (Staff returned OR Customer left)
        elif not is_unattended and self.last_unattended_state[camera_id]:
            try:
                # Smart Resolution Frame Selection: Choose the best proof
                target_frame_url = frame_url  # Default fallback
                repr_staff_detection = staff_detection
                repr_customer_detection = customer_detection
                repr_staff_in_customer_detections = staff_in_customer_detections
                repr_customer_in_staff_detections = customer_in_staff_detections

                if is_staff_detected:
                    # Reason: Staff returned. Find most recent frame WITH staff.
                    for d in reversed(self.history[camera_id]):
                        if len(d.get("staff_detection", [])) > 0 and d.get("frame_url"):
                            target_frame_url = d["frame_url"]
                            repr_staff_detection = d.get("staff_detection", [])
                            repr_customer_detection = d.get("customer_detection", [])
                            repr_staff_in_customer_detections = d.get("staff_in_customer_detections", [])
                            repr_customer_in_staff_detections = d.get("customer_in_staff_detections", [])
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
                            repr_staff_detection = d.get("staff_detection", [])
                            repr_customer_detection = d.get("customer_detection", [])
                            repr_staff_in_customer_detections = d.get("staff_in_customer_detections", [])
                            repr_customer_in_staff_detections = d.get("customer_in_staff_detections", [])
                            logger.info(
                                f"Resolution proof: Selected frame with 0 customers."
                            )
                            break

                # Download and encode the frame (resizing to 25% of original pixel area)
                img_response = requests.get(target_frame_url, timeout=10)
                img_response.raise_for_status()
                # Save full resolution image
                if len(img_response.content) == 0:
                    raise ValueError("Empty image content")

                # Decode to CV2 image
                img_array = np.frombuffer(img_response.content, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                # Encode RAW (without visualization)
                _, raw_buffer = cv2.imencode(".jpg", img)
                raw_frame_b64 = base64.b64encode(raw_buffer).decode("utf-8")

                # Annotate the frame
                img = self.annotate_frame(
                    img,
                    staff_pixel_polygons,
                    customer_pixel_polygons_list,
                    repr_staff_detection,
                    repr_customer_detection,
                    repr_staff_in_customer_detections,
                    repr_customer_in_staff_detections,
                )

                # Encode back to jpg then base64
                _, buffer = cv2.imencode(".jpg", img)
                frame_b64 = base64.b64encode(buffer).decode("utf-8")

                alert_id = self.current_unattended_alert_id[camera_id]

                resolve_payload = {
                    "camera_id": camera_id,
                    "timestamp": timestamp,
                    "frame": frame_b64,
                    "frame_without_visualization": raw_frame_b64,
                    "description": "Unattended customer situation resolved (Staff returned or customer left).",
                    "target_collection": "alert_collection",
                    "type": "resolved",
                    "ticket_id": alert_id,
                }
                self.redis.xadd(OUTPUT_QUEUE_1, {"data": json.dumps(resolve_payload)})
                logger.info(f"ALERT RESOLVED: Unattended customer situation at counter {camera_id} resolved. Ticket: {alert_id}")
                # Clear persisted state
                self.redis.delete(f"alert_state:unattended:{camera_id}")
                # Update state only on success
                self.last_unattended_state[camera_id] = False
            except Exception as e:
                logger.error(f"Failed to process/push unattended resolve dump: {e}")


        # High Customer Traffic Alert Logic (Customers > 3 in majority of latest half of history)
        full_history = list(self.history[camera_id])
        half_index = len(full_history) // 2
        recent_history = full_history[half_index:]  # Latest half of frames

        high_traffic_frames_count = sum(
            1
            for d in recent_history
            if len(d.get("customer_detection", [])) > 3
        )
        is_high_traffic = high_traffic_frames_count > (
            len(recent_history) / 2
        )

        if is_high_traffic and not self.last_high_traffic_state[camera_id]:
            try:
                # Find representative frame from recent history (one with > 3 customers)
                target_frame_url = frame_url
                repr_staff_detection = staff_detection
                repr_customer_detection = customer_detection
                repr_staff_in_customer_detections = staff_in_customer_detections
                repr_customer_in_staff_detections = customer_in_staff_detections

                for d in reversed(recent_history):
                    if len(d.get("customer_detection", [])) > 3 and d.get("frame_url"):
                        target_frame_url = d["frame_url"]
                        repr_staff_detection = d.get("staff_detection", [])
                        repr_customer_detection = d.get("customer_detection", [])
                        repr_staff_in_customer_detections = d.get("staff_in_customer_detections", [])
                        repr_customer_in_staff_detections = d.get("customer_in_staff_detections", [])
                        logger.info("Selected representative frame for High Traffic alert.")
                        break

                img_response = requests.get(target_frame_url, timeout=10)
                img_response.raise_for_status()
                if len(img_response.content) == 0:
                    raise ValueError("Empty image content")

                img_array = np.frombuffer(img_response.content, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                _, raw_buffer = cv2.imencode(".jpg", img)
                raw_frame_b64 = base64.b64encode(raw_buffer).decode("utf-8")

                img = self.annotate_frame(
                    img,
                    staff_pixel_polygons,
                    customer_pixel_polygons_list,
                    repr_staff_detection,
                    repr_customer_detection,
                    repr_staff_in_customer_detections,
                    repr_customer_in_staff_detections,
                    unified=True,
                )
                _, buffer = cv2.imencode(".jpg", img)
                frame_b64 = base64.b64encode(buffer).decode("utf-8")

                alert_id = str(uuid.uuid4())
                self.current_high_traffic_alert_id[camera_id] = alert_id

                dump_payload = {
                    "camera_id": camera_id,
                    "timestamp": timestamp,
                    "frame": frame_b64,
                    "frame_without_visualization": raw_frame_b64,
                    "description": "High customer traffic detected (More than 3 customers).",
                    "target_collection": "alert_collection",
                    "type": "raised",
                    "ticket_id": alert_id,
                }
                self.redis.xadd(OUTPUT_QUEUE_1, {"data": json.dumps(dump_payload)})
                logger.warning(f"ALERT RAISED: High customer traffic at counter {camera_id}. Ticket: {alert_id}")
                # Persist state
                self.redis.set(f"alert_state:high_traffic:{camera_id}", alert_id)
                # Update state only on success
                self.last_high_traffic_state[camera_id] = True
            except Exception as e:
                logger.error(f"Failed to process/push High Traffic alert dump: {e}")

        elif not is_high_traffic and self.last_high_traffic_state[camera_id]:
            try:
                # Find representative frame from recent history (one with <= 3 customers)
                target_frame_url = frame_url
                repr_staff_detection = staff_detection
                repr_customer_detection = customer_detection
                repr_staff_in_customer_detections = staff_in_customer_detections
                repr_customer_in_staff_detections = customer_in_staff_detections

                for d in reversed(recent_history):
                    if len(d.get("customer_detection", [])) <= 3 and d.get("frame_url"):
                        target_frame_url = d["frame_url"]
                        repr_staff_detection = d.get("staff_detection", [])
                        repr_customer_detection = d.get("customer_detection", [])
                        repr_staff_in_customer_detections = d.get("staff_in_customer_detections", [])
                        repr_customer_in_staff_detections = d.get("customer_in_staff_detections", [])
                        logger.info("Resolution proof: Selected frame with <= 3 customers.")
                        break

                img_response = requests.get(target_frame_url, timeout=10)
                img_response.raise_for_status()
                if len(img_response.content) == 0:
                    raise ValueError("Empty image content")

                img_array = np.frombuffer(img_response.content, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                _, raw_buffer = cv2.imencode(".jpg", img)
                raw_frame_b64 = base64.b64encode(raw_buffer).decode("utf-8")

                img = self.annotate_frame(
                    img,
                    staff_pixel_polygons,
                    customer_pixel_polygons_list,
                    repr_staff_detection,
                    repr_customer_detection,
                    repr_staff_in_customer_detections,
                    repr_customer_in_staff_detections,
                    unified=True,
                )
                _, buffer = cv2.imencode(".jpg", img)
                frame_b64 = base64.b64encode(buffer).decode("utf-8")

                alert_id = self.current_high_traffic_alert_id[camera_id]

                resolve_payload = {
                    "camera_id": camera_id,
                    "timestamp": timestamp,
                    "frame": frame_b64,
                    "frame_without_visualization": raw_frame_b64,
                    "description": "High customer traffic resolved.",
                    "target_collection": "alert_collection",
                    "type": "resolved",
                    "ticket_id": alert_id,
                }
                self.redis.xadd(OUTPUT_QUEUE_1, {"data": json.dumps(resolve_payload)})
                logger.info(f"ALERT RESOLVED: High customer traffic at counter {camera_id} resolved. Ticket: {alert_id}")
                # Clear persisted state
                self.redis.delete(f"alert_state:high_traffic:{camera_id}")
                # Update state only on success
                self.last_high_traffic_state[camera_id] = False
            except Exception as e:
                logger.error(f"Failed to process/push High Traffic resolve dump: {e}")

        # Customer In Staff Zone Alert Logic
        # Condition: Customer detected in staff zone in majority of history (similar to other alerts)
        cust_in_staff_frames_count = sum(
            1
            for d in self.history[camera_id]
            if len(d.get("customer_in_staff_detections", [])) > 0
        )
        is_cust_in_staff = cust_in_staff_frames_count > (
            len(self.history[camera_id]) / 2
        )

        if is_cust_in_staff and not self.last_customer_in_staff_zone_state[camera_id]:
            try:
                # Find representative frame and its detections
                target_frame_url = frame_url
                repr_staff_detection = staff_detection
                repr_customer_detection = customer_detection
                repr_staff_in_customer_detections = staff_in_customer_detections
                repr_customer_in_staff_detections = customer_in_staff_detections
                for d in reversed(self.history[camera_id]):
                    if len(d.get("customer_in_staff_detections", [])) > 0 and d.get("frame_url"):
                        target_frame_url = d["frame_url"]
                        repr_staff_detection = d.get("staff_detection", [])
                        repr_customer_detection = d.get("customer_detection", [])
                        repr_staff_in_customer_detections = d.get("staff_in_customer_detections", [])
                        repr_customer_in_staff_detections = d.get("customer_in_staff_detections", [])
                        logger.info("Selected representative frame for CustInStaff alert.")
                        break

                img_response = requests.get(target_frame_url, timeout=10)
                img_response.raise_for_status()
                if len(img_response.content) == 0:
                    raise ValueError("Empty image content")
                img_array = np.frombuffer(img_response.content, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                _, raw_buffer = cv2.imencode(".jpg", img)
                raw_frame_b64 = base64.b64encode(raw_buffer).decode("utf-8")

                img = self.annotate_frame(
                    img,
                    staff_pixel_polygons,
                    customer_pixel_polygons_list,
                    repr_staff_detection,
                    repr_customer_detection,
                    repr_staff_in_customer_detections,
                    repr_customer_in_staff_detections,
                )

                _, buffer = cv2.imencode(".jpg", img)
                frame_b64 = base64.b64encode(buffer).decode("utf-8")

                alert_id = str(uuid.uuid4())
                self.current_customer_in_staff_alert_id[camera_id] = alert_id

                dump_payload = {
                    "camera_id": camera_id,
                    "timestamp": timestamp,
                    "frame": frame_b64,
                    "frame_without_visualization": raw_frame_b64,
                    "description": "Violation: Customer detected in Staff Zone.",
                    "target_collection": "Customer_in_Staff",
                    "type": "raised",
                    "ticket_id": alert_id,
                }
                self.redis.xadd(OUTPUT_QUEUE_1, {"data": json.dumps(dump_payload)})
                logger.warning(f"ALERT RAISED: Customer in staff zone at counter {camera_id}. Ticket: {alert_id}")
                # Persist state
                self.redis.set(f"alert_state:cust_in_staff:{camera_id}", alert_id)
                # Update state only on success
                self.last_customer_in_staff_zone_state[camera_id] = True
            except Exception as e:
                logger.error(f"Failed to process/push CustInStaff alert: {e}")

        elif not is_cust_in_staff and self.last_customer_in_staff_zone_state[camera_id]:
            try:
                # Find representative frame (one with NO customers in staff zone)
                target_frame_url = frame_url
                repr_staff_detection = staff_detection
                repr_customer_detection = customer_detection
                repr_staff_in_customer_detections = staff_in_customer_detections

                for d in reversed(self.history[camera_id]):
                    if len(d.get("customer_in_staff_detections", [])) == 0 and d.get("frame_url"):
                        target_frame_url = d["frame_url"]
                        repr_staff_detection = d.get("staff_detection", [])
                        repr_customer_detection = d.get("customer_detection", [])
                        repr_staff_in_customer_detections = d.get("staff_in_customer_detections", [])
                        repr_customer_in_staff_detections = d.get("customer_in_staff_detections", [])
                        logger.info("Resolution proof: Selected frame with 0 customers in staff zone.")
                        break

                img_response = requests.get(target_frame_url, timeout=10)
                img_response.raise_for_status()
                if len(img_response.content) == 0:
                     raise ValueError("Empty image content")

                img_array = np.frombuffer(img_response.content, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                # Encode RAW (without visualization)
                _, raw_buffer = cv2.imencode(".jpg", img)
                raw_frame_b64 = base64.b64encode(raw_buffer).decode("utf-8")

                img = self.annotate_frame(
                    img,
                    staff_pixel_polygons,
                    customer_pixel_polygons_list,
                    repr_staff_detection,
                    repr_customer_detection,
                    repr_staff_in_customer_detections,
                    repr_customer_in_staff_detections,
                )
                _, buffer = cv2.imencode(".jpg", img)
                frame_b64 = base64.b64encode(buffer).decode("utf-8")

                alert_id = self.current_customer_in_staff_alert_id[camera_id]
                dump_payload = {
                    "camera_id": camera_id,
                    "timestamp": timestamp,
                    "frame": frame_b64,
                    "frame_without_visualization": raw_frame_b64,
                    "description": "Customer in Staff Zone violation resolved.",
                    "target_collection": "Customer_in_Staff",
                    "type": "resolved",
                    "ticket_id": alert_id,
                }
                self.redis.xadd(OUTPUT_QUEUE_1, {"data": json.dumps(dump_payload)})
                logger.info(f"ALERT RESOLVED: Customer in staff zone at counter {camera_id} resolved. Ticket: {alert_id}")
                # Clear persisted state
                self.redis.delete(f"alert_state:cust_in_staff:{camera_id}")
                # Update state only on success
                self.last_customer_in_staff_zone_state[camera_id] = False
            except Exception as e:
                logger.error(f"Failed to process/push CustInStaff resolve: {e}")

        # Calculate Statistics
        history_counts = [
            len(d.get("customer_detection", [])) for d in self.history[camera_id]
        ]

        mean_val = 0
        median_val = 0
        mode_val = 0

        if history_counts:
            try:
                mean_val = statistics.mean(history_counts)
                median_val = statistics.median(history_counts)
                mode_val = statistics.mode(history_counts)
            except Exception as e:
                logger.error(f"Error calculating stats: {e}")

        # service_output_time = time.time()
        output_payload = {
            "camera_id": camera_id,
            "timestamp": timestamp,
            "valid_detection": staff_detection,
            "customer_detection": customer_detection,
            "is_staff_detected": is_staff_detected,
            "is_customer_detected": is_customer_detected,
            # "service_reading_time": service_reading_time,
            # "service_output_time": service_output_time,
            # "processing_latency": service_output_time - service_reading_time,
            # "yolo_processing_latency": yolo_latency,
            # "internal_processing_latency": (service_output_time - service_reading_time) - yolo_latency,
            "customer_count": len(customer_detection),
            "customer_count_mean": mean_val,
            "customer_count_median": median_val,
            "customer_count_mode": mode_val,
            # "customer_count_std": std_val,
            # "customer_count_min": min_val,
            # "customer_count_max": max_val,
            "target_collection": os.getenv(
                "TARGET_COLLECTION_NAME", "counter_staff_collection"
            ),
        }

        # Update non-alert states unconditionally
        self.last_staff_detected_state[camera_id] = is_staff_detected
        # NOTE: Alert states (unattended, high_traffic, cust_in_staff) are updated
        # inside their respective try blocks above, so that if the publish fails,
        # the state remains unchanged and the resolution will be retried next frame.


        # logger.info(f"Output Payload: {json.dumps(output_payload)}")
        try:
            self.redis.xadd(OUTPUT_QUEUE_1, {"data": json.dumps(output_payload)})

            # Push visualization data if enabled
            vis_duration = 0
            if VISUALIZE:
                vis_start = time.time()
                try:
                    # Download and encode the frame for visualization
                    img_response = requests.get(frame_url, timeout=10)
                    img_response.raise_for_status()
                    if len(img_response.content) == 0:
                        raise ValueError("Empty image content")

                    frame_b64 = base64.b64encode(img_response.content).decode("utf-8")

                    vis_payload = {
                        "camera_id": camera_id,
                        "timestamp": timestamp,
                        "frame": frame_b64,
                        "staff_detection": staff_detection,
                        "customer_detection": customer_detection,
                        "staff_in_customer_detections": staff_in_customer_detections,
                        "customer_in_staff_detections": customer_in_staff_detections,
                        "staff_polygons": staff_pixel_polygons,
                        "customer_polygons": customer_pixel_polygons_list,
                        "is_staff_detected": is_staff_detected,
                        "is_customer_detected": is_customer_detected,
                        "customer_count": len(customer_detection),
                        "stats": {
                            "mean": mean_val,
                            "median": median_val,
                            "mode": mode_val,
                        },
                    }
                    self.vis_queue.push(json.dumps(vis_payload))
                except Exception as e:
                    logger.error(f"Failed to push visualization frame: {e}")
                vis_duration = time.time() - vis_start

            total_duration = time.time() - service_reading_time
            logger.info(
                f"Published detection result for camera {camera_id} at {timestamp}. "
                f"Detections - Persons: {len(detections)}, Staff: {len(staff_detection)}, "
                f"Customers: {len(customer_detection)}, Staff-in-Cust: {len(staff_in_customer_detections)}, "
                f"Cust-in-Staff: {len(customer_in_staff_detections)}. "
                f"Total Time: {total_duration:.3f}s (YOLO: {yolo_duration:.3f}s, Post: {post_duration:.3f}s, Vis: {vis_duration:.3f}s)"
            )
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
