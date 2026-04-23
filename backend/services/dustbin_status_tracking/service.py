import json
from dotenv import load_dotenv
import redis
import logging
import time
import requests
import logging_loki
from multiprocessing import Queue
import os
import cv2
import numpy as np
from typing import List, Dict, Optional
from collections import deque
import io
from PIL import Image

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
LOKI_URL = os.getenv("LOKI_URL", "http://localhost:3100/loki/api/v1/push")

INPUT_QUEUE = os.getenv("INPUT_QUEUE_NAME")
OUTPUT_QUEUE_1 = os.getenv("OUTPUT_QUEUE_NAME_1")
GROUP_NAME = os.getenv("GROUP_NAME")

# Detection configuration from environment
REGION_POLYGON = os.getenv(
    "REGION_POLYGON",
    '[{"x": 70.67, "y": 220.94}, {"x": 159.67, "y": 171.21}, {"x": 162.29, "y": 207.85}, {"x": 78.53, "y": 257.59}]',
)
LOWER_THRESHOLD = int(os.getenv("LOWER_THRESHOLD", "600000"))
UPPER_THRESHOLD = int(os.getenv("UPPER_THRESHOLD", "750000"))
DARK_PATCH_SENSITIVITY = int(os.getenv("DARK_PATCH_SENSITIVITY", "30"))
LOCAL_DEVIATION_THRESHOLD = int(os.getenv("LOCAL_DEVIATION_THRESHOLD", "20"))
LOCAL_STRIP_THRESHOLD_PERCENT = float(
    os.getenv("LOCAL_STRIP_THRESHOLD_PERCENT", "10.0")
)
MIN_CONTOUR_AREA = int(os.getenv("MIN_CONTOUR_AREA", "300"))
EDGE_MARGIN = int(os.getenv("EDGE_MARGIN", "5"))
MORPH_KERNEL_SIZE = int(os.getenv("MORPH_KERNEL_SIZE", "3"))
ACK_TTL_SECONDS = 10 * 60  # 10 minutes

# Configure Logging
handler = logging_loki.LokiQueueHandler(
    Queue(-1),
    url=LOKI_URL,
    tags={"application": "dustbin_status_tracking"},
    version="1",
)

logger = logging.getLogger("dustbin_status_tracking")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
# Also log to console
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)


class DustbinStatusDetector:
    def __init__(self):
        # Redis connection
        self.redis = self.connect_redis(host=REDIS_HOST, port=REDIS_PORT)

        # Parse region polygon
        self.region_data = json.loads(REGION_POLYGON)
        self.region_poly = self._to_polygon(self.region_data)

        # Pre-compute ROI bounding box for faster processing
        self.roi_bbox = cv2.boundingRect(self.region_poly)
        self.roi_x, self.roi_y, self.roi_w, self.roi_h = self.roi_bbox

        # Pre-compute ROI mask (only once, reuse for all frames)
        self.roi_mask_full = np.zeros((self.roi_h, self.roi_w), dtype=np.uint8)
        region_poly_local = self.region_poly.copy().astype(np.float32)
        region_poly_local[:, 0, 0] -= self.roi_x
        region_poly_local[:, 0, 1] -= self.roi_y
        region_poly_local = region_poly_local.astype(np.int32)
        cv2.fillPoly(self.roi_mask_full, [region_poly_local], 255)

        # Status history for smoothing
        self.status_history = deque(maxlen=5)

        logger.info(f"DustbinStatusDetector initialized with ROI: {self.roi_bbox}")

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

    def _to_polygon(self, region_data: List[Dict[str, float]]) -> np.ndarray:
        """Convert region data to numpy polygon array."""
        pts = [[int(p["x"]), int(p["y"])] for p in region_data]
        return np.array(pts, np.int32).reshape((-1, 1, 2))

    def connect_redis(self, host, port, retry_delay=5):
        while True:
            try:
                r = redis.Redis(host=host, port=port, decode_responses=False)
                # Test connection
                r.ping()
                logger.info("Connected to Redis successfully.")
                return r

            except Exception as e:
                logger.error(
                    f"Redis connection failed: {e}. Retrying in {retry_delay} seconds..."
                )
                time.sleep(retry_delay)

    def calculate_region_sum(self, frame: np.ndarray) -> int:
        """Calculate sum of all RGB values within the dustbin region."""
        roi_frame = frame[
            self.roi_y : self.roi_y + self.roi_h, self.roi_x : self.roi_x + self.roi_w
        ]
        selected_pixels = roi_frame[self.roi_mask_full == 255]
        return int(np.sum(selected_pixels))

    def check_vertical_connectivity(self, frame: np.ndarray) -> bool:
        """Check if a connected dark section spans vertically across the region."""
        cropped = frame[
            self.roi_y : self.roi_y + self.roi_h, self.roi_x : self.roi_x + self.roi_w
        ]

        c_mask = self.roi_mask_full

        if not np.any(c_mask == 255):
            return False

        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        avg_intensity = np.mean(gray[c_mask == 255])
        thresh_val = max(0, min(255, avg_intensity - DARK_PATCH_SENSITIVITY))

        _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
        binary = cv2.bitwise_and(binary, binary, mask=c_mask)

        kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        center_y = self.roi_h / 2

        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
                continue

            ys = cnt[:, 0, 1]
            touches_top = np.any(ys <= EDGE_MARGIN)
            touches_bottom = np.any(ys >= self.roi_h - EDGE_MARGIN)
            crosses_center = np.any(ys < center_y) and np.any(ys > center_y)

            if (touches_top and touches_bottom) or crosses_center:
                return True

        return False

    def check_local_strip_threshold(self, frame: np.ndarray) -> bool:
        """Check if bottom-half strip has significant deviation from average."""
        cropped_frame = frame[
            self.roi_y : self.roi_y + self.roi_h, self.roi_x : self.roi_x + self.roi_w
        ]
        cropped_mask = self.roi_mask_full

        # Define strip region (bottom half)
        strip_h_start = int(self.roi_h * 0.5)
        strip_h_end = int(self.roi_h * 0.7)
        strip_w_start = int(self.roi_w * 0.2)
        strip_w_end = int(self.roi_w * 0.8)

        strip_frame = cropped_frame[
            strip_h_start:strip_h_end, strip_w_start:strip_w_end
        ]
        strip_mask = cropped_mask[strip_h_start:strip_h_end, strip_w_start:strip_w_end]

        strip_gray = cv2.cvtColor(strip_frame, cv2.COLOR_BGR2GRAY)
        valid_pixels = strip_gray[strip_mask == 255]

        if len(valid_pixels) == 0:
            return False

        avg_intensity = np.mean(valid_pixels)
        total_valid_pixels = len(valid_pixels)

        dark_patch = np.sum(valid_pixels < (avg_intensity - LOCAL_DEVIATION_THRESHOLD))
        dark_ratio = (dark_patch / total_valid_pixels) * 100

        light_patch = np.sum(valid_pixels > (avg_intensity + LOCAL_DEVIATION_THRESHOLD))
        light_ratio = (light_patch / total_valid_pixels) * 100

        return (
            dark_ratio >= LOCAL_STRIP_THRESHOLD_PERCENT
            or light_ratio >= LOCAL_STRIP_THRESHOLD_PERCENT
        )

    def detect_dustbin_status(self, frame: np.ndarray) -> str:
        """
        Detect dustbin status: 'operational', 'undetectable', or 'full'

        Returns:
            str: The detected status
        """
        # Calculate pixel sum
        pixel_sum = self.calculate_region_sum(frame)

        # Check if sum is within threshold
        sum_in_threshold = LOWER_THRESHOLD <= pixel_sum <= UPPER_THRESHOLD

        if sum_in_threshold:
            return "operational"

        # Outside threshold - check for person
        connectivity_check = self.check_vertical_connectivity(frame)

        if connectivity_check:
            return "undetectable"

        # Check local strip
        local_strip_check = self.check_local_strip_threshold(frame)

        if local_strip_check:
            return "undetectable"

        return "full"

    def download_frame(self, frame_url: str) -> Optional[np.ndarray]:
        """Download frame from URL and convert to numpy array."""
        try:
            response = requests.get(frame_url, timeout=10)
            response.raise_for_status()

            # Convert to PIL Image then to numpy array
            image = Image.open(io.BytesIO(response.content))
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            return frame

        except Exception as e:
            logger.error(f"Failed to download frame from {frame_url}: {e}")
            return None

    def process_message(self, message: dict):
        """Process a single Redis message.
        Expected format: {"frame_url": "...", "camera_id": "...", "timestamp": "..."}
        """
        message = {k.decode(): v.decode() for k, v in message.items()}
        frame_url = message.get("frame_url")
        camera_id = message.get("camera_id")
        timestamp = message.get("timestamp")

        if not frame_url:
            logger.warning("Message missing 'frame_url'. Skipping.")
            return

        # Download frame
        frame = self.download_frame(frame_url)
        if frame is None:
            logger.error(f"Failed to download frame from {frame_url}")
            return

        # Detect dustbin status
        try:
            status = self.detect_dustbin_status(frame)

            # Add to history for smoothing
            self.status_history.append(status)

            # Use majority vote from history
            if len(self.status_history) >= 3:
                final_status = max(
                    set(self.status_history), key=self.status_history.count
                )
            else:
                final_status = status

            # Prepare output data
            current_data = {
                "camera_id": camera_id,
                "timestamp": timestamp,
                "status": final_status,
                "target_collection": "dustbin_status_collection",
            }

            # Publish to output queue
            try:
                self.redis.xadd(OUTPUT_QUEUE_1, {"data": json.dumps(current_data)})
                logger.info(
                    f"Published dustbin status for camera {camera_id} at {timestamp}: {final_status}"
                )
            except Exception as e:
                logger.error(f"Failed to push result to output queue: {e}")

        except Exception as e:
            logger.error(f"Failed to detect dustbin status: {e}")

    def run(self):
        logger.info("DustbinStatusDetector started, listening on input queue.")
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
    service = DustbinStatusDetector()
    service.run()
