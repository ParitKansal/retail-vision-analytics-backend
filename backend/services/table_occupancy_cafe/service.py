import requests
import numpy as np
import os
import sys
import json
from collections import defaultdict, deque
from enum import Enum
from uuid import uuid4
import base64
import redis
import logging_loki
import logging
from multiprocessing import Queue
from dotenv import load_dotenv
import ast
import time

# Add current directory to path for imports when running as script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import total occupancy patch functions
from total_occupancy_patch import count_persons_in_region, load_region_config


load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
LOKI_URL = os.getenv("LOKI_URL", "http://localhost:3100/loki/api/v1/push")
INPUT_QUEUE = os.getenv("INPUT_QUEUE_NAME")
OUTPUT_QUEUE_1 = os.getenv("OUTPUT_QUEUE_NAME_1")
YOLO_MODEL_URL = os.getenv("YOLO_MODEL_URL")
# GROUNDING_DINO_URL = os.getenv("GROUNDING_DINO_URL")
GROUP_NAME = os.getenv("GROUP_NAME")
FPS = os.getenv("FPS", 1)
CAMERA_ID = os.getenv("CAMERA_ID", INPUT_QUEUE.split(":")[-1])
MASK_PATH = os.getenv("MASK_PATH", None)
ROTATION = os.getenv("ROTATION", None)
VISUALIZE = os.getenv("VISUALIZE", "false").lower() in ("true", "1", "yes")

# -- New Configuration --
BUFFER_SIZE = int(os.getenv("BUFFER_SIZE", 60))
PERCENTAGE_DIRTY_IN_BUFFER = int(os.getenv("PERCENTAGE_DIRTY_IN_BUFFER", 34))
PERCENTAGE_OCCUPIED_IN_BUFFER = int(os.getenv("PERCENTAGE_OCCUPIED_IN_BUFFER", 34))
CONSECUTIVE_CLEAN_FOR_RESOLUTION = int(
    os.getenv("CONSECUTIVE_CLEAN_FOR_RESOLUTION", 60)
)
CONSECUTIVE_UNOCCUPIED_FOR_RESOLUTION = int(
    os.getenv("CONSECUTIVE_UNOCCUPIED_FOR_RESOLUTION", 60)
)
ACK_TTL_SECONDS = 60 * 60  # 10 minutes
# Configure Logging
handler = logging_loki.LokiQueueHandler(
    Queue(-1),
    url=LOKI_URL,
    tags={"application": f"table_cleaning_service_{CAMERA_ID}"},
    version="1",
)

logger = logging.getLogger(f"table_cleaning_service_{CAMERA_ID}")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
# Also log to console
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)


class TableStatus(Enum):
    EMPTY: int = 0
    OCCUPIED: int = 1
    DIRTY: int = 2


MASK = None
if MASK_PATH:
    with open(MASK_PATH, "rb") as mask_file:
        MASK = json.load(mask_file)


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


class TableFrameProcessor:
    def __init__(
        self,
        # grounding_dino_url,
        yolo_model_url,
        fps,
        mask,
        rotation,
    ):

        self.DETECTION_THRESHOLD = 0.25
        self.NMS_IOU = 0.5
        self.PERSON_TABLE_IOU_THRESHOLD = 0.30
        self.PLATE_TABLE_IOU_THRESHOLD = 0.05
        self.PLATE_OVERLAP_THRESHOLD = 0.5
        self.PLATE_ONLY_FRAME_THRESHOLD = 30
        self.PLATE_ONLY_FRAME_THRESHOLD = 30

        # NMS Control Flags
        self.APPLY_NMS_ON_PERSON = (
            False  # Set to False to skip NMS on person detections
        )
        self.APPLY_NMS_ON_PLATE = True  # Set to False to skip NMS on plate detections

        # New Resolution Vars
        self.CONSECUTIVE_CLEAN_FOR_RESOLUTION = CONSECUTIVE_CLEAN_FOR_RESOLUTION
        self.CONSECUTIVE_UNOCCUPIED_FOR_RESOLUTION = (
            CONSECUTIVE_UNOCCUPIED_FOR_RESOLUTION
        )

        # Detection Thresholds
        self.PERSON_CONF_THRESHOLD = float(os.getenv("PERSON_CONF_THRESHOLD", 0.25))
        self.PLATE_CONF_THRESHOLD = float(os.getenv("PLATE_CONF_THRESHOLD", 0.15))
        self.TABLE_CONF_THRESHOLD = float(os.getenv("TABLE_CONF_THRESHOLD", 0.25))

        # Calculated Thresholds (Count)
        self.OCCUPIED_TRIGGER_THRESHOLD = int(
            BUFFER_SIZE * (PERCENTAGE_OCCUPIED_IN_BUFFER / 100.0)
        )
        self.DIRTY_TRIGGER_THRESHOLD = int(
            BUFFER_SIZE * (PERCENTAGE_DIRTY_IN_BUFFER / 100.0)
        )

        self.mask = mask
        self.rotation = rotation
        # self.API_URL = grounding_dino_url
        self.yolo_endpoint = yolo_model_url

        # Load table configs from JSON
        self.config_path = os.path.join(
            os.path.dirname(__file__), "artifacts", "table_configs.json"
        )
        self.checks_path = os.path.join(
            os.path.dirname(__file__), "artifacts", "checks.json"
        )


        
        # Track modification times for auto-reload
        self.last_config_mtime = os.path.getmtime(self.config_path)
        self.last_checks_mtime = os.path.getmtime(self.checks_path)
        self.last_fs_check_time = 0
        self.FS_CHECK_INTERVAL = 30.0 # Check file system every 2 seconds

        # Load region config for total occupancy counting
        self.occupancy_regions = load_region_config(region_key="cafe")
        logger.info(
            f"Loaded {len(self.occupancy_regions)} occupancy region(s) for counting"
        )

        # Use BUFFER_SIZE directly for deque maxlen
        self.table_history = defaultdict(lambda: deque(maxlen=BUFFER_SIZE))
        self.open_tickets = defaultdict(lambda: {"occupied": None, "dirty": None})
        self.table_status = dict()
        self.table_status_raw = dict()
        self.current_table_state = defaultdict(
            lambda: TableStatus.EMPTY.name
        )  # Track the *latched* state
        self.many_dirty_ticket = None
        
        self.redis = self.connect_redis(host=REDIS_HOST, port=REDIS_PORT)
        
        self._load_configs()
        if VISUALIZE:
            self.vis_queue = FixedSizeQueue(
                self.redis, f"stream:visualization:{CAMERA_ID}", 10
            )

        # Timestamp for last table config update
        self.last_config_update_time = 0
        self.CONFIG_UPDATE_INTERVAL = int(os.getenv("CONFIG_UPDATE_INTERVAL", 1800))  # Default 30 minutes
        self.CONFIG_UPDATE_IOU_THRESHOLD = float(os.getenv("CONFIG_UPDATE_IOU_THRESHOLD", 0.8))  # new table 80% overlap


    class utility_functions:

        @staticmethod
        def calculate_iou(box1, box2):
            x1_min, y1_min, x1_max, y1_max = box1
            x2_min, y2_min, x2_max, y2_max = box2

            inter_xmin = max(x1_min, x2_min)
            inter_ymin = max(y1_min, y2_min)
            inter_xmax = min(x1_max, x2_max)
            inter_ymax = min(y1_max, y2_max)

            inter_w = max(0.0, inter_xmax - inter_xmin)
            inter_h = max(0.0, inter_ymax - inter_ymin)
            inter_area = inter_w * inter_h

            box1_area = max(0.0, (x1_max - x1_min)) * max(0.0, (y1_max - y1_min))
            box2_area = max(0.0, (x2_max - x2_min)) * max(0.0, (y2_max - y2_min))

            union = box1_area + box2_area - inter_area
            return inter_area / union if union > 0 else 0.0

        @staticmethod
        def left_side_overlap(person_box, table_box, min_vertical_iou=0.1):
            """
            Check if the table's LEFT boundary line is inside the person box
            and calculate the vertical overlap ratio.
            """
            px1, py1, px2, py2 = person_box
            tx1, ty1, tx2, ty2 = table_box

            # 1. Check if the line (x = tx1) is within the person's x-range
            # strict containment: px1 <= tx1 <= px2
            line_in_bounds = px1 <= tx1 <= px2

            if not line_in_bounds:
                return False

            # 2. Calculate vertical overlap of the line segment
            # Line segment is from ty1 to ty2
            vert_inter = max(0, min(py2, ty2) - max(py1, ty1))
            vert_union = ty2 - ty1

            ratio = vert_inter / vert_union if vert_union > 0 else 0.0

            return ratio >= min_vertical_iou

        @staticmethod
        def right_side_overlap(person_box, table_box, min_vertical_iou=0.1):
            """
            Check if the table's RIGHT boundary line is inside the person box
            and calculate the vertical overlap ratio.
            """
            px1, py1, px2, py2 = person_box
            tx1, ty1, tx2, ty2 = table_box

            # 1. Check if the line (x = tx2) is within the person's x-range
            line_in_bounds = px1 <= tx2 <= px2

            if not line_in_bounds:
                return False

            # 2. Calculate vertical overlap
            vert_inter = max(0, min(py2, ty2) - max(py1, ty1))
            vert_union = ty2 - ty1

            ratio = vert_inter / vert_union if vert_union > 0 else 0.0

            return ratio >= min_vertical_iou

        @staticmethod
        def lower_side_overlap(person_box, table_box, min_horizontal_iou=0.1):
            """
            Check if the table's BOTTOM boundary line is inside the person box
            and calculate the horizontal overlap ratio.
            """
            px1, py1, px2, py2 = person_box
            tx1, ty1, tx2, ty2 = table_box

            # 1. Check if the line (y = ty2) is within the person's y-range
            line_in_bounds = py1 <= ty2 <= py2

            if not line_in_bounds:
                return False

            # 2. Calculate horizontal overlap
            horiz_inter = max(0, min(px2, tx2) - max(px1, tx1))
            horiz_union = tx2 - tx1

            ratio = horiz_inter / horiz_union if horiz_union > 0 else 0.0

            return ratio >= min_horizontal_iou

        @staticmethod
        def upper_side_overlap(person_box, table_box, min_horizontal_iou=0.05):
            """
            Check if the table's TOP boundary line is inside the person box
            and calculate the horizontal overlap ratio.
            """
            px1, py1, px2, py2 = person_box
            tx1, ty1, tx2, ty2 = table_box

            # 1. Check if the line (y = ty1) is within the person's y-range
            line_in_bounds = py1 <= ty1 <= py2

            if not line_in_bounds:
                return False

            # 2. Calculate horizontal overlap
            horiz_inter = max(0, min(px2, tx2) - max(px1, tx1))
            horiz_union = tx2 - tx1

            ratio = horiz_inter / horiz_union if horiz_union > 0 else 0.0

            return ratio >= min_horizontal_iou

        @staticmethod
        def non_max_suppression(boxes, scores, iou_threshold=0.5):
            if len(boxes) == 0:
                return []
            dets = [
                {"bbox": boxes[i].tolist(), "score": float(scores[i])}
                for i in range(len(boxes))
            ]
            dets.sort(key=lambda x: x["score"], reverse=True)

            keep = []
            suppressed = [False] * len(dets)  # Track suppressed detections

            for i in range(len(dets)):
                if suppressed[i]:
                    continue

                keep.append(dets[i])

                # Mark overlapping boxes as suppressed
                for j in range(i + 1, len(dets)):
                    if not suppressed[j]:
                        iou = TableFrameProcessor.utility_functions.calculate_iou(
                            dets[i]["bbox"], dets[j]["bbox"]
                        )
                        if iou >= iou_threshold:
                            suppressed[j] = True

            return keep

        @staticmethod
        def detect_with_api(
            payload,
            prompt_text,
            score_thresh,
            api_url,
            nms_iou=0.5,
            timeout=10,
            apply_nms=True,
        ):

            payload["text"] = prompt_text

            try:
                r = requests.post(api_url, files=None, json=payload, timeout=timeout)
                r.raise_for_status()
                resp = r.json()
            except Exception as e:
                logging.error(f"API error:{e}")
                return []

            dets = resp.get("detections", {}) if isinstance(resp, dict) else {}
            boxes = dets.get("boxes", [])
            scores = dets.get("scores", [])

            if len(boxes) == 0:
                return []

            boxes_np = np.array(boxes, dtype=np.float32)
            scores_np = np.array(scores, dtype=np.float32)

            if apply_nms:
                filtered = TableFrameProcessor.utility_functions.non_max_suppression(
                    boxes_np, scores_np, iou_threshold=nms_iou
                )
            else:
                # Skip NMS, just convert to expected format
                filtered = [
                    {
                        "bbox": (
                            boxes[i].tolist()
                            if hasattr(boxes[i], "tolist")
                            else boxes[i]
                        ),
                        "score": float(scores[i]),
                    }
                    for i in range(len(boxes))
                ]

            return [
                {"bbox": det["bbox"], "score": det["score"]}
                for det in filtered
                if det["score"] >= score_thresh
            ]

        @staticmethod
        def calculate_object_overlap_ratio(container_box, object_box):
            """
            Calculate what percentage of the object is inside the container.
            This is more appropriate for small objects (like plates) vs large containers (like tables).

            Returns: ratio of (intersection_area / object_area)
            Range: 0.0 (no overlap) to 1.0 (object completely inside container)
            """
            c_x1, c_y1, c_x2, c_y2 = container_box
            o_x1, o_y1, o_x2, o_y2 = object_box

            # Calculate intersection
            inter_xmin = max(c_x1, o_x1)
            inter_ymin = max(c_y1, o_y1)
            inter_xmax = min(c_x2, o_x2)
            inter_ymax = min(c_y2, o_y2)

            inter_w = max(0.0, inter_xmax - inter_xmin)
            inter_h = max(0.0, inter_ymax - inter_ymin)
            inter_area = inter_w * inter_h

            # Calculate object area
            object_area = max(0.0, (o_x2 - o_x1)) * max(0.0, (o_y2 - o_y1))

            return inter_area / object_area if object_area > 0 else 0.0

        @staticmethod
        def get_overlap_stats(person_box, table_box):
            """
            Calculates raw overlap stats for debugging.
            Returns dict with left, right, up, down, table_overlap ratios.
            """
            px1, py1, px2, py2 = person_box
            tx1, ty1, tx2, ty2 = table_box

            # 1. Table Overlap (IOU)
            table_overlap = TableFrameProcessor.utility_functions.calculate_iou(
                person_box, table_box
            )

            # 2. Left Overlap (Check if x=tx1 is inside [px1, px2] and measure vertical overlap)
            left_in_bounds = px1 <= tx1 <= px2
            if left_in_bounds:
                l_vert_inter = max(0, min(py2, ty2) - max(py1, ty1))
                l_vert_union = ty2 - ty1
                left_ratio = l_vert_inter / l_vert_union if l_vert_union > 0 else 0.0
            else:
                left_ratio = 0.0

            # 3. Right Overlap (Check if x=tx2 is inside [px1, px2] and measure vertical overlap)
            right_in_bounds = px1 <= tx2 <= px2
            if right_in_bounds:
                r_vert_inter = max(0, min(py2, ty2) - max(py1, ty1))
                r_vert_union = ty2 - ty1
                right_ratio = r_vert_inter / r_vert_union if r_vert_union > 0 else 0.0
            else:
                right_ratio = 0.0

            # 4. Up/Upper Overlap (Check if y=ty1 is inside [py1, py2] and measure horizontal overlap)
            up_in_bounds = py1 <= ty1 <= py2
            if up_in_bounds:
                u_horiz_inter = max(0, min(px2, tx2) - max(px1, tx1))
                u_horiz_union = tx2 - tx1
                up_ratio = u_horiz_inter / u_horiz_union if u_horiz_union > 0 else 0.0
            else:
                up_ratio = 0.0

            # 5. Down/Lower Overlap (Check if y=ty2 is inside [py1, py2] and measure horizontal overlap)
            down_in_bounds = py1 <= ty2 <= py2
            if down_in_bounds:
                d_horiz_inter = max(0, min(px2, tx2) - max(px1, tx1))
                d_horiz_union = tx2 - tx1
                down_ratio = d_horiz_inter / d_horiz_union if d_horiz_union > 0 else 0.0
            else:
                down_ratio = 0.0

            return {
                "table_overlap": round(table_overlap, 2),
                "left": f"{left_ratio:.2f} ({left_in_bounds})",
                "right": f"{right_ratio:.2f} ({right_in_bounds})",
                "up": f"{up_ratio:.2f} ({up_in_bounds})",
                "down": f"{down_ratio:.2f} ({down_in_bounds})",
            }
    
    def _load_configs(self):
        """
        Helper to load configuration files from disk.
        """
        try:
            with open(self.config_path, "r") as f:
                full_configs = json.load(f)["cafe"]
                self.table_configs = full_configs
                
            with open(self.checks_path, "r") as f:
                self.checks_config = json.load(f)
            
            # Rebuild mutable table structure
            self.tables = []
            for t_id, t_bbox in self.table_configs.items():
                coords = [int(x) for x in t_bbox]
                self.tables.append({"bbox": coords, "id": t_id})
                
                # Ensure history exists for new tables
                if t_id not in self.table_history:
                     self.table_history[t_id].append({"person": False, "tray": False})
                     
                if t_id not in self.open_tickets:
                    self.open_tickets[t_id] = {"occupied": None, "dirty": None}
                    
                if t_id not in self.current_table_state:
                    self.current_table_state[t_id] = TableStatus.EMPTY.name
            
            logger.info(f"Configuration loaded. Tables: {list(self.table_configs.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to load configs: {e}")

    def _reload_configs_if_changed(self):
        """
        Checks file modification times and reloads if changed.
        """
        try:
            now = time.time()
            if now - self.last_fs_check_time < self.FS_CHECK_INTERVAL:
                return

            self.last_fs_check_time = now
            
            config_mtime = os.path.getmtime(self.config_path)
            checks_mtime = os.path.getmtime(self.checks_path)
            
            if config_mtime > self.last_config_mtime or checks_mtime > self.last_checks_mtime:
                logger.info("Configuration change detected. Reloading...")
                self._load_configs()
                self.last_config_mtime = config_mtime
                self.last_checks_mtime = checks_mtime
                
        except Exception as e:
            logger.error(f"Error checking config reload: {e}")


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

    def draw_visualizations(self, img, table_bboxes, table_status, persons, plates):
        """
        Draw bounding boxes and status for alerts.
        """
        try:
            import cv2
        except ImportError:
            logger.error("opencv-python not installed, skipping visualization")
            return img

        # Colors (BGR)
        COLOR_EMPTY = (0, 255, 0)  # Green
        COLOR_OCCUPIED = (0, 0, 255)  # Red
        COLOR_DIRTY = (0, 255, 255)  # Yellow
        COLOR_PERSON = (255, 0, 0)  # Blue
        COLOR_PLATE = (255, 255, 0)  # Cyan

        # Draw Tables
        for table in table_bboxes:
            bbox = table["bbox"]
            table_id = table.get("id", "?")
            x1, y1, x2, y2 = [int(c) for c in bbox]

            status = table_status.get(
                str(table_id), table_status.get(table_id, "EMPTY")
            )

            if status == "OCCUPIED":
                color = COLOR_OCCUPIED
            elif status == "DIRTY":
                color = COLOR_DIRTY
            else:
                color = COLOR_EMPTY

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            label = f"Table {table_id}: {status}"
            cv2.putText(
                img,
                label,
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        # Draw Persons
        for idx, person in enumerate(persons):
            bbox = person["bbox"]
            x1, y1, x2, y2 = [int(c) for c in bbox]
            cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_PERSON, 2)

        # Draw Plates
        for idx, plate in enumerate(plates):
            bbox = plate["bbox"]
            x1, y1, x2, y2 = [int(c) for c in bbox]
            cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_PLATE, 2)

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

    def process_single_frame(self, message):
        """
        Processes ONE SINGLE IMAGE FRAME.
        frame : numpy BGR image
        """

        start_time = time.time()
        frame_url = message.get("frame_url")
        frame_shape = message.get("frame_shape")
        timestamp = message.get("timestamp")
        camera_id = message.get("camera_id")
        # -----------------------------------------
        # Run external detector API
        # -----------------------------------------
        height, width = frame_shape[:2]
        tables = []

        if self.last_fs_check_time == 0:
             # First run initialization of self.tables is handled in _load_configs called from __init__
             pass
        
        # Check for config updates from disk
        self._reload_configs_if_changed()

        # Update initial frame data for any new tables
        # Update initial frame data for any new tables
        for t_id in self.table_configs:
             self.table_history[t_id].append({"person": False, "tray": False})

        # No need to rebuild self.tables here as _reload_configs_if_changed handles it
        # But we need to ensure self.debug_table_stats is reset
        self.debug_table_stats = {}  # Initialize per-frame stats container
        payload = {"image_url": frame_url}
        if ROTATION:
            payload["rotation"] = self.rotation
        if MASK_PATH and self.mask.get("mask", None):
            payload["polygon_mask"] = self.mask.get("mask")
            
        # Request lower confidence from Yolo to allow our flexible filtering
        payload["confidence"] = 0.1

        persons = []
        plates = []
        try:
            response = requests.post(
                self.yolo_endpoint,
                json=payload,
            )
            logger.info(response.json())
            response = response.json()
            response = response.get("predict", None)
            response = json.loads(response[0])

            # Extract person bboxes and scores for NMS
            person_boxes = []
            person_scores = []
            detected_tables = []  # Capture table detections

            for det in response:
                name = det.get("name", "").lower()

                # Filter for plates/trays/dishes
                if name in ["plate", "tray", "dish"]:
                    # Filter plate/tray by confidence
                    if det.get("confidence", 0) < self.PLATE_CONF_THRESHOLD:
                        continue
                    box = det.get("box")
                    plates.append(
                        {
                            "bbox": [
                                box.get("x1"),
                                box.get("y1"),
                                box.get("x2"),
                                box.get("y2"),
                            ],
                            "score": det.get("confidence", 1.0),
                        }
                    )
                    continue # Skip to next detection if it's a plate/tray/dish

                # Check for dining table detection for config update
                if name in ["dining table", "table"]:
                    # Filter table by confidence
                    if det.get("confidence", 0) < self.TABLE_CONF_THRESHOLD:
                        continue
                        
                    box = det.get("box")
                    detected_tables.append(
                        [
                            box.get("x1"),
                            box.get("y1"),
                            box.get("x2"),
                            box.get("y2"),
                        ]
                    )

                if name != "person":
                    continue
                
                # Filter person by confidence
                if det.get("confidence", 0) < self.PERSON_CONF_THRESHOLD:
                    continue

                box = det.get("box")
                person_boxes.append(
                    [
                        box.get("x1"),
                        box.get("y1"),
                        box.get("x2"),
                        box.get("y2"),
                    ]
                )
                person_scores.append(det.get("confidence", 1.0))

            # --- Auto-Update Table Config Logic ---
            # ONLY enabled for tables "1" and "2"
            try:
                current_time = time.time()
                if (
                    current_time - self.last_config_update_time
                ) >= self.CONFIG_UPDATE_INTERVAL:
                    logger.info("Running table config update check...")
                    config_changed = False

                    # For each configured table, find best matching detected table
                    # ONLY update tables "1" and "2"
                    for t_id, config_bbox in self.table_configs.items():
                        # Skip tables other than "1" and "2"
                        if t_id not in ["1", "2"]:
                            continue
                            
                        best_iou = 0.0
                        best_match_bbox = None

                        # Find best match from detections
                        for det_bbox in detected_tables:
                            # Check if it overlaps at all (loose check first)
                            iou = self.utility_functions.calculate_iou(
                                config_bbox, det_bbox
                            )
                            if iou > best_iou:
                                best_iou = iou
                                best_match_bbox = det_bbox

                        # Logic:
                        # If we found a match (best_iou > 0), check if it meets the strictly 80% threshold.
                        # If best_iou < 0.80, it means the table drifted or config is inaccurate.
                        # We update config to the new detected box.
                        if (
                            best_match_bbox is not None and best_iou > 0.1
                        ):  # Ensure it's roughly the same table (min 10% overlap safety)
                            if best_iou < self.CONFIG_UPDATE_IOU_THRESHOLD:
                                logger.info(
                                    f"Table {t_id} overlap {best_iou:.2f} < 0.80. Updating config."
                                )
                                self.table_configs[t_id] = best_match_bbox
                                config_changed = True
                            else:
                                logger.info(
                                    f"Table {t_id} overlap {best_iou:.2f} >= 0.80. Keeping existing config."
                                )
                        else:
                            # No matching detection found for this table (e.g. occlusion). Do not update.
                            logger.info(
                                f"Table {t_id} has no matching detection. Skipping update."
                            )

                    if config_changed:
                        # Save to file
                        try:
                            config_path = os.path.join(
                                os.path.dirname(__file__), "artifacts", "table_configs.json"
                            )
                            # Create the full structure for saving
                            
                            # Re-read to ensure we don't overwrite other keys if they exist in the file but not in memory
                            with open(config_path, "r") as f:
                                full_config = json.load(f)

                            full_config["cafe"] = self.table_configs

                            with open(config_path, "w") as f:
                                json.dump(full_config, f, indent=4)
                            logger.info("Table configuration updated and saved to file.")

                        except Exception as e:
                            logger.error(f"Failed to save updated table config: {e}")

                    self.last_config_update_time = current_time

            except Exception as e:
                logger.error(f"Error in table config auto-update: {e}")
            # --------------------------------------

            # Apply NMS to persons (controlled by APPLY_NMS_ON_PERSON flag)
            if person_boxes:
                if self.APPLY_NMS_ON_PERSON:
                    boxes_np = np.array(person_boxes, dtype=np.float32)
                    scores_np = np.array(person_scores, dtype=np.float32)
                    nms_persons = self.utility_functions.non_max_suppression(
                        boxes_np, scores_np, iou_threshold=self.NMS_IOU
                    )
                    persons = [
                        {"bbox": p["bbox"], "score": p["score"]} for p in nms_persons
                    ]
                    logger.info(
                        f"Persons after NMS: {len(persons)} (from {len(person_boxes)} detections)"
                    )
                else:
                    persons = [
                        {"bbox": person_boxes[i], "score": person_scores[i]}
                        for i in range(len(person_boxes))
                    ]
                    logger.info(f"Persons without NMS: {len(persons)} detections")
            end_time = time.time()
            logger.info(
                f"Person detection and NMS took {end_time - start_time:.2f} seconds"
            )
        except Exception as e:
            logging.error(e)

        # --- GROUNDING DINO PLATE DETECTION DISABLED ---
        # (Commented out due to slow inference time)
        # dino_start_time = time.time()
        # plates = self.utility_functions.detect_with_api(
        #     payload,
        #     "plate . tray .",
        #     self.DETECTION_THRESHOLD,
        #     self.API_URL,
        #     nms_iou=self.NMS_IOU,
        #     apply_nms=self.APPLY_NMS_ON_PLATE,
        # )
        # dino_end_time = time.time()
        # logger.info(f"Plate detection and NMS took {dino_end_time - dino_start_time:.2f} seconds")
        # plates = []  # Empty plates list since detection is disabled

        # --------------------------------------------
        # Compute status: Empty / Occupied / Plates
        # --------------------------------------------
        # --------------------------------------------
        # 1. Determine occupied tables based on priority
        # --------------------------------------------
        occupied_table_ids = set()
        person_assignments = defaultdict(list)
        for p in persons:
            matched_tables = []
            for table_data in self.tables:
                t_id = table_data["id"]
                t_bbox = table_data["bbox"]
                if self.check_person_occupancy(p["bbox"], t_id, t_bbox):
                    # Default priority is 0
                    prio = self.checks_config.get(t_id, {}).get("priority", 0)
                    matched_tables.append((prio, t_id))

            if matched_tables:
                # Sort by priority descending.
                # If a person overlaps multiple tables, they belong to the one with highest priority.
                matched_tables.sort(key=lambda x: x[0], reverse=True)
                assigned_table = matched_tables[0][1]
                occupied_table_ids.add(assigned_table)
                person_assignments[assigned_table].append(p)

        plate_assignments = defaultdict(list)
        for table_data in self.tables:
            index = table_data["id"]
            table_bbox = table_data["bbox"]

            table_plates = []
            for p in plates:
                if self.utility_functions.calculate_object_overlap_ratio(table_bbox, p["bbox"]) >= self.PLATE_OVERLAP_THRESHOLD:
                    table_plates.append(p)
            plate_assignments[index] = table_plates

            # Use priority-based assignment calculated above
            # This ensures we respect the "One Person -> One Table" rule based on priority
            person_overlap = index in occupied_table_ids

            is_person_present = False
            if person_overlap:
                is_person_present = True

            is_tray_present = False
            if len(table_plates) > 0:
                is_tray_present = True

            # Update current frame observation in history (which was initialized at start of frame)
            self.table_history[index][-1] = {
                "person": is_person_present,
                "tray": is_tray_present,
            }

            # --- State Machine Logic ---
            history = list(self.table_history[index])
            person_count_in_history = sum(1 for frame in history if frame["person"])
            tray_count_in_history = sum(1 for frame in history if frame["tray"])

            current_state = self.current_table_state[
                index
            ]  # Can be OCCUPIED, DIRTY, EMPTY

            # 1. Check for OCCUPIED
            if current_state != TableStatus.OCCUPIED.name:
                # Trigger Rule: X% times in buffer
                if person_count_in_history >= self.OCCUPIED_TRIGGER_THRESHOLD:
                    self.current_table_state[index] = TableStatus.OCCUPIED.name

            elif current_state == TableStatus.OCCUPIED.name:
                # Resolve Rule: X frames consecutively seen empty (UNOCCUPIED)
                resolve_window = history[
                    -min(len(history), self.CONSECUTIVE_UNOCCUPIED_FOR_RESOLUTION) :
                ]
                resolve_window_has_person = any(
                    frame["person"] for frame in resolve_window
                )

                if (
                    not resolve_window_has_person
                    and len(resolve_window)
                    >= self.CONSECUTIVE_UNOCCUPIED_FOR_RESOLUTION
                ):
                    self.current_table_state[index] = TableStatus.EMPTY.name

            # 2. Check for DIRTY (Only if currently EMPTY or DIRTY - i.e. NOT OCCUPIED)
            current_state = self.current_table_state[index]

            if current_state != TableStatus.OCCUPIED.name:
                if current_state != TableStatus.DIRTY.name:
                    # Trigger Rule: X% frames with tray
                    if tray_count_in_history >= self.DIRTY_TRIGGER_THRESHOLD:
                        self.current_table_state[index] = TableStatus.DIRTY.name

                elif current_state == TableStatus.DIRTY.name:
                    # Resolve Rule: X frames consecutively to have no trays (CLEAN)
                    resolve_window = history[
                        -min(len(history), self.CONSECUTIVE_CLEAN_FOR_RESOLUTION) :
                    ]
                    resolve_window_has_tray = any(
                        frame["tray"] for frame in resolve_window
                    )

                    if (
                        not resolve_window_has_tray
                        and len(resolve_window) >= self.CONSECUTIVE_CLEAN_FOR_RESOLUTION
                    ):
                        self.current_table_state[index] = TableStatus.EMPTY.name

            # Final State update for output
            self.table_status[index] = self.current_table_state[index]

            # Raw State update (No latching)
            if is_person_present:
                self.table_status_raw[index] = TableStatus.OCCUPIED.name
            elif is_tray_present:
                self.table_status_raw[index] = TableStatus.DIRTY.name
            else:
                self.table_status_raw[index] = TableStatus.EMPTY.name

            # --- Alert Logic ---
            if index not in ["1", "2"]:
                continue

            tickets = self.open_tickets[index]



            # --- DIRTY Alert Logic ---
            dirty_ticket = tickets["dirty"]

            # Raise Dirty Ticket if State became DIRTY and no ticket exists
            if (
                self.current_table_state[index] == TableStatus.DIRTY.name
                and not dirty_ticket
            ):
                try:
                    img_response = requests.get(frame_url, timeout=10)
                    img_response.raise_for_status()
                    if len(img_response.content) == 0:
                        raise ValueError("Empty image content")

                    # Store raw frame
                    raw_frame_b64 = base64.b64encode(img_response.content).decode(
                        "utf-8"
                    )

                    # Try to visualize
                    try:
                        import cv2

                        np_arr = np.frombuffer(img_response.content, np.uint8)
                        frame_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        if frame_cv is not None:
                            target_tables = [t for t in self.tables if str(t["id"]) == str(index)]
                            target_persons = person_assignments.get(index, [])
                            target_plates = plate_assignments.get(index, [])
                            frame_cv = self.draw_visualizations(
                                frame_cv,
                                target_tables,
                                self.table_status,
                                target_persons,
                                target_plates,
                            )
                            _, encoded_img = cv2.imencode(".jpg", frame_cv)
                            frame_b64 = base64.b64encode(encoded_img).decode("utf-8")
                        else:
                            frame_b64 = base64.b64encode(img_response.content).decode(
                                "utf-8"
                            )
                    except Exception as e:
                        logger.error(f"Visualization failed: {e}")
                        frame_b64 = base64.b64encode(img_response.content).decode(
                            "utf-8"
                        )
                    logger.info(f"dirty ticket raised for table {index}: {camera_id}")

                    new_ticket_id = str(uuid4())

                    ticket = {
                        "camera_id": camera_id,
                        "timestamp": timestamp,
                        "description": f"Table {index} detected as dirty in cafe region",
                        "type": "raised",
                        "ticket_id": new_ticket_id,
                        "target_collection": "alert_collection",
                        "frame": frame_b64,
                        "raw_frame": raw_frame_b64,
                        "table_status_raw": self.table_status_raw,
                        "table_bboxes": [t for t in self.tables if str(t["id"]) == str(index)],
                        "table_status": self.table_status,
                    }

                    self.open_tickets[index]["dirty"] = new_ticket_id

                    self.redis.xadd(OUTPUT_QUEUE_1, {"data": json.dumps(ticket)})
                    logger.info(
                        f"Published dirty alert for camera {camera_id} at {timestamp}"
                    )
                except Exception as e:
                    logger.error(f"Failed to raise dirty ticket: {e}")

            # Resolve Dirty Ticket if State became EMPTY (from DIRTY)
            elif (
                dirty_ticket
                and self.current_table_state[index] == TableStatus.EMPTY.name
            ):
                try:
                    img_response = requests.get(frame_url, timeout=10)
                    img_response.raise_for_status()
                    if len(img_response.content) == 0:
                        raise ValueError("Empty image content")

                    # Store raw frame
                    raw_frame_b64 = base64.b64encode(img_response.content).decode(
                        "utf-8"
                    )

                    # Try to visualize
                    try:
                        import cv2

                        np_arr = np.frombuffer(img_response.content, np.uint8)
                        frame_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        if frame_cv is not None:
                            target_tables = [t for t in self.tables if str(t["id"]) == str(index)]
                            target_persons = person_assignments.get(index, [])
                            target_plates = plate_assignments.get(index, [])
                            frame_cv = self.draw_visualizations(
                                frame_cv,
                                target_tables,
                                self.table_status,
                                target_persons,
                                target_plates,
                            )
                            _, encoded_img = cv2.imencode(".jpg", frame_cv)
                            frame_b64 = base64.b64encode(encoded_img).decode("utf-8")
                        else:
                            frame_b64 = base64.b64encode(img_response.content).decode(
                                "utf-8"
                            )
                    except Exception as e:
                        logger.error(f"Visualization failed: {e}")
                        frame_b64 = base64.b64encode(img_response.content).decode(
                            "utf-8"
                        )

                    logger.info(f"dirty ticket resolved for table {index}: {camera_id}")

                    existing_ticket_id = self.open_tickets[index]["dirty"]

                    ticket = {
                        "camera_id": camera_id,
                        "timestamp": timestamp,
                        "description": f"Table {index} marked as clean in cafe region",
                        "type": "resolved",
                        "ticket_id": existing_ticket_id,
                        "target_collection": "alert_collection",
                        "frame": frame_b64,
                        "raw_frame": raw_frame_b64,
                        "table_status_raw": self.table_status_raw,
                        "table_bboxes": [t for t in self.tables if str(t["id"]) == str(index)],
                        "table_status": self.table_status,
                    }
                    self.open_tickets[index]["dirty"] = None
                    self.redis.xadd(OUTPUT_QUEUE_1, {"data": json.dumps(ticket)})
                    logger.info(
                        f"Published dirty resolution for camera {camera_id} at {timestamp}"
                    )
                except Exception as e:
                    logger.error(f"Failed to resolve dirty ticket: {e}")

        # --- Aggregate >4 Dirty Tables Alert Logic ---
        dirty_table_count = sum(1 for status in self.current_table_state.values() if status == TableStatus.DIRTY.name)

        if dirty_table_count > 4 and not self.many_dirty_ticket:
            try:
                img_response = requests.get(frame_url, timeout=10)
                img_response.raise_for_status()
                
                raw_frame_b64 = base64.b64encode(img_response.content).decode("utf-8")
                try:
                    import cv2
                    np_arr = np.frombuffer(img_response.content, np.uint8)
                    frame_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if frame_cv is not None:
                        frame_cv = self.draw_visualizations(
                            frame_cv,
                            self.tables,
                            self.table_status,
                            persons,
                            plates,
                        )
                        _, encoded_img = cv2.imencode(".jpg", frame_cv)
                        frame_b64 = base64.b64encode(encoded_img).decode("utf-8")
                    else:
                        frame_b64 = raw_frame_b64
                except Exception as e:
                    logger.error(f"Visualization failed for aggregate dirty ticket: {e}")
                    frame_b64 = raw_frame_b64

                new_ticket_id = str(uuid4())
                ticket = {
                    "camera_id": camera_id,
                    "timestamp": timestamp,
                    "description": f"High number of dirty tables detected in cafe region (More than 4 tables)",
                    "type": "raised",
                    "ticket_id": new_ticket_id,
                    "target_collection": "alert_collection",
                    "frame": frame_b64,
                    "raw_frame": raw_frame_b64,
                }
                self.many_dirty_ticket = new_ticket_id
                self.redis.xadd(OUTPUT_QUEUE_1, {"data": json.dumps(ticket)})
                logger.info(f"Published aggregate dirty alert for camera {camera_id} at {timestamp}")
            except Exception as e:
                logger.error(f"Failed to raise aggregate dirty ticket: {e}")

        elif dirty_table_count <= 4 and self.many_dirty_ticket:
            try:
                img_response = requests.get(frame_url, timeout=10)
                img_response.raise_for_status()
                
                raw_frame_b64 = base64.b64encode(img_response.content).decode("utf-8")
                try:
                    import cv2
                    np_arr = np.frombuffer(img_response.content, np.uint8)
                    frame_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if frame_cv is not None:
                        frame_cv = self.draw_visualizations(
                            frame_cv,
                            self.tables,
                            self.table_status,
                            persons,
                            plates,
                        )
                        _, encoded_img = cv2.imencode(".jpg", frame_cv)
                        frame_b64 = base64.b64encode(encoded_img).decode("utf-8")
                    else:
                        frame_b64 = raw_frame_b64
                except Exception as e:
                    logger.error(f"Visualization failed for aggregate dirty ticket: {e}")
                    frame_b64 = raw_frame_b64

                existing_ticket_id = self.many_dirty_ticket
                ticket = {
                    "camera_id": camera_id,
                    "timestamp": timestamp,
                    "description": f"High number of dirty tables resolved in cafe region",
                    "type": "resolved",
                    "ticket_id": existing_ticket_id,
                    "target_collection": "alert_collection",
                    "frame": frame_b64,
                    "raw_frame": raw_frame_b64,
                }
                self.many_dirty_ticket = None
                self.redis.xadd(OUTPUT_QUEUE_1, {"data": json.dumps(ticket)})
                logger.info(f"Published aggregate dirty resolution for camera {camera_id} at {timestamp}")
            except Exception as e:
                logger.error(f"Failed to resolve aggregate dirty ticket: {e}")

        # --------------------------------------------
        # Calculate total occupancy count in region
        # --------------------------------------------
        occupancy_result = count_persons_in_region(
            persons, regions=self.occupancy_regions, overlap_threshold=0.6
        )
        total_persons_in_region = occupancy_result["total_count"]
        logger.info(
            f"Total persons in region: {total_persons_in_region}/{len(persons)}"
        )

        # --------------------------------------------
        # Build output
        # --------------------------------------------
        output = {
            "camera_id": camera_id,
            "timestamp": timestamp,
            "table_bboxes": self.tables,
            "table_status": self.table_status,
            "table_status_raw": self.table_status_raw,
            "total_persons_in_region": total_persons_in_region,
            "total_persons_detected": len(persons),
            "target_collection": "table_status_collection",
        }
        try:
            self.redis.xadd(OUTPUT_QUEUE_1, {"data": json.dumps(output)})
            if VISUALIZE:
                try:
                    img_response = requests.get(frame_url, timeout=10)
                    img_response.raise_for_status()
                    frame_b64 = base64.b64encode(img_response.content).decode("utf-8")

                    # Calculate debug stats for all person-table pairs
                    debug_stats = {}
                    for i, p in enumerate(persons):
                        p_stats = {}
                        for t in self.tables:
                            t_id = t["id"]
                            metrics = self.utility_functions.get_overlap_stats(
                                p["bbox"], t["bbox"]
                            )
                            metrics["priority"] = self.checks_config.get(t_id, {}).get(
                                "priority", 0
                            )
                            metrics["is_occupying"] = (
                                t_id in self.table_status
                                and self.table_status[t_id] == TableStatus.OCCUPIED.name
                                and self.check_person_occupancy(
                                    p["bbox"], t_id, t["bbox"]
                                )
                            )
                            p_stats[t_id] = metrics
                        debug_stats[i] = p_stats

                    vis_payload = {
                        **output,
                        "frame": frame_b64,
                        "persons": persons,
                        "plates": plates,
                        "debug_stats": debug_stats,
                        "occupancy_result": occupancy_result,
                        "occupancy_regions": self.occupancy_regions,
                    }
                    self.vis_queue.push(json.dumps(vis_payload))
                except Exception as e:
                    logger.error(f"Failed to push visualization frame: {e}")
            logger.info(
                f"Published detection result for camera {camera_id} at {timestamp}"
            )
            end_time = time.time()
            logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Failed to push result to output queue: {e}")

    def evaluate_check_block(self, person_box, table_box, check_block):
        """
        Evaluates a SINGLE check block (AND logic).
        Returns True if ALL conditions in the block are met.
        """
        # table_overlap (IOU)
        if "table_overlap" in check_block:
            thresh = check_block["table_overlap"]
            # Some configs might be 0-100, others 0-1. Heuristic: if > 1, divide by 100 ?
            if check_block["table_overlap"] is not None:
                iou = self.utility_functions.calculate_iou(table_box, person_box)
                if iou < float(check_block["table_overlap"]):
                    return False

        # Side overlaps
        # Map JSON key -> (Utility Function, Threshold Normalization)
        side_checks = {
            "left": self.utility_functions.left_side_overlap,
            "right": self.utility_functions.right_side_overlap,
            "up": self.utility_functions.upper_side_overlap,
            "down": self.utility_functions.lower_side_overlap,
        }

        for side, func in side_checks.items():
            if side in check_block and check_block[side] is not None:
                val = float(check_block[side])
                if val <= 0:
                    # Skip if threshold is 0 or negative
                    continue

                # Convert to float 0.0-1.0 if it looks like percentage (> 1.0)
                threshold = val / 100.0 if val > 1.0 else val

                # Use correct keyword argument based on function signature
                # left/right: min_vertical_iou
                # up/down: min_horizontal_iou
                kwargs = {}
                if side in ["left", "right"]:
                    kwargs["min_vertical_iou"] = threshold
                else:
                    kwargs["min_horizontal_iou"] = threshold

                if not func(person_box, table_box, **kwargs):
                    return False

        return True

    def check_person_occupancy(self, person_box, table_id, table_box):
        """
        Determines if a person occupies a table based on configured checks (OR logic).
        """
        config = self.checks_config.get(table_id)
        if not config:
            # Default fallback if no config: simple IOU
            return self.utility_functions.calculate_iou(table_box, person_box) > 0.3

        checks = config.get("checks", [])

        # Handle legacy flat format (if "checks" list missing but flat keys exist)
        if not checks:
            # Check if it's a flat dict containing criteria
            # If 'left', 'right', 'table_overlap' etc in config top level
            potential_keys = [
                "left",
                "right",
                "up",
                "down",
                "table_overlap",
                "axis_overlap",
            ]
            if any(k in config for k in potential_keys):
                checks = [config]  # Wrap as single block

        if not checks:
            return self.utility_functions.calculate_iou(table_box, person_box) > 0.3

        # OR logic across blocks
        for block in checks:
            if self.evaluate_check_block(person_box, table_box, block):
                return True

        return False

    def run(self):
        logger.info("TableDetector started, listening on input queue.")
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
                    message = {k.decode(): v.decode() for k, v in message.items()}
                    frame_shape = ast.literal_eval(message["frame_shape"])
                    message["frame_shape"] = frame_shape
                    self.process_single_frame(message)
                    self.redis.xack(INPUT_QUEUE, GROUP_NAME, msg_id)
                    self.mark_acked(INPUT_QUEUE, msg_id.decode(), GROUP_NAME)
            except json.JSONDecodeError:
                logger.error("Failed to decode JSON message from input queue.")
            except Exception as e:
                logger.exception(f"Unexpected error in main loop: {e}")
                time.sleep(1)


if __name__ == "__main__":
    table_processor = TableFrameProcessor(
        # grounding_dino_url=GROUNDING_DINO_URL,
        yolo_model_url=YOLO_MODEL_URL,
        fps=FPS,
        mask=MASK,
        rotation=ROTATION,
    )
    table_processor.run()
