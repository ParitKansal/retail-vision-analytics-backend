"""
Entry/Exit Counting Service
A Redis-based service that processes video frames and tracks customer entry/exit events.
Uses Remote YOLO Service + Local ByteTrack + Shapely for detection and logic.
"""

import json
import cv2
import yaml
from dotenv import load_dotenv
from pathlib import Path
import redis
import logging
import time
import argparse
import os
import numpy as np
import requests
import base64
from typing import Dict, Optional, List, Tuple
import logging_loki
from multiprocessing import Queue
from collections import defaultdict
from datetime import datetime
import psutil
import threading
import queue
import signal
from statistics import mode

# Try to import BYTETracker if available, else we'll need a fallback or error
try:
    from ultralytics.trackers import BYTETracker
except ImportError:
    print("WARNING: ultralytics ByteTracker not found. Please install ultralytics.")
    BYTETracker = None

from shapely.geometry import Point, Polygon

load_dotenv()


class MockBoxes:
    """
    Mock class to mimic ultralytics Boxes object.
    BYTETracker expects results.boxes with .xyxy, .conf, .cls, .xywh attributes.
    """

    def __init__(self, detections: np.ndarray):
        """
        Args:
            detections: numpy array of shape (N, 6) with [x1, y1, x2, y2, conf, cls]
        """
        self._detections = detections
        if len(detections) == 0:
            self.xyxy = np.empty((0, 4))
            self.conf = np.empty((0,))
            self.cls = np.empty((0,))
            self.xywh = np.empty((0, 4))
        else:
            self.xyxy = detections[:, :4]  # [x1, y1, x2, y2]
            self.conf = detections[:, 4]  # confidence scores
            self.cls = detections[:, 5]  # class IDs
            # Convert xyxy to xywh (center_x, center_y, width, height)
            self.xywh = self._xyxy_to_xywh(self.xyxy)

    @staticmethod
    def _xyxy_to_xywh(xyxy: np.ndarray) -> np.ndarray:
        """Convert [x1, y1, x2, y2] to [center_x, center_y, width, height]."""
        if len(xyxy) == 0:
            return np.empty((0, 4))
        xywh = np.zeros_like(xyxy)
        xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2  # center_x
        xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2  # center_y
        xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]  # width
        xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]  # height
        return xywh

    def __len__(self):
        return len(self._detections)

    def __getitem__(self, idx):
        """Support indexing/slicing for BYTETracker filtering operations."""
        if isinstance(idx, (int, slice)):
            sliced_dets = self._detections[idx]
        else:
            # Handle numpy array or list indices
            sliced_dets = self._detections[idx]

        # Ensure 2D array even for single item
        if sliced_dets.ndim == 1:
            sliced_dets = sliced_dets.reshape(1, -1)

        return MockBoxes(sliced_dets)


class MockResults:
    """
    Mock class to mimic ultralytics Results object for BYTETracker compatibility.
    The ultralytics BYTETracker.update() expects a Results object with:
    - results.boxes.xyxy  (N, 4) bounding boxes
    - results.boxes.conf  (N,) confidence scores
    - results.boxes.cls   (N,) class IDs
    - results.boxes.xywh  (N, 4) bounding boxes in xywh format
    - results.xywh        (N, 4) for init_track method
    - results.orig_shape  (height, width) of original image
    - Support for indexing/slicing: results[indices]
    """

    def __init__(self, detections: np.ndarray, orig_shape: tuple):
        """
        Args:
            detections: numpy array of shape (N, 6) with [x1, y1, x2, y2, conf, cls]
            orig_shape: tuple of (height, width) of the original frame
        """
        self._detections = detections
        self.boxes = MockBoxes(detections)
        self.orig_shape = orig_shape

        # Additional attributes that BYTETracker might access
        if len(detections) == 0:
            self.conf = np.empty((0,))
            self.cls = np.empty((0,))
            self.xyxy = np.empty((0, 4))
            self.xywh = np.empty((0, 4))
        else:
            self.conf = detections[:, 4]
            self.cls = detections[:, 5]
            self.xyxy = detections[:, :4]
            self.xywh = self.boxes.xywh  # Use the xywh computed by MockBoxes

    def __len__(self):
        return len(self._detections)

    def __getitem__(self, idx):
        """Support indexing/slicing for BYTETracker filtering operations."""
        if isinstance(idx, (int, slice)):
            sliced_dets = self._detections[idx]
        else:
            # Handle numpy array or list indices
            sliced_dets = self._detections[idx]

        # Ensure 2D array even for single item
        if sliced_dets.ndim == 1:
            sliced_dets = sliced_dets.reshape(1, -1)

        return MockResults(sliced_dets, self.orig_shape)


LOKI_URL = os.getenv("LOKI_URL", "http://localhost:3100/loki/api/v1/push")

INPUT_RTSP = os.getenv("INPUT_RTSP")
# print this
print(INPUT_RTSP)
OUTPUT_QUEUE_1 = os.getenv("OUTPUT_QUEUE_NAME_1")
CONFIG_PATH = os.getenv("CONFIG_PATH", "config.json")
YOLO_SERVICE_URL = os.getenv("YOLO_SERVICE_URL", "http://localhost:8083/predict")
FRAME_QUEUE_SIZE = int(os.getenv("FRAME_QUEUE_SIZE", "30"))
CAMERA_ID = os.getenv("CAMERA_ID", "entrance")
VISUALIZE = os.getenv("VISUALIZE", True)

# Configure Logging
handler = logging_loki.LokiQueueHandler(
    Queue(-1),
    url=LOKI_URL,
    tags={"application": "entry_exit_service2"},
    version="1",
)

logger = logging.getLogger("entry_exit_service2")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
# Also log to console
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

logger.addHandler(console_handler)


class FrameReader(threading.Thread):
    def __init__(self, source, queue_obj, drop_if_full=True, reconnect_interval=5):
        super().__init__()
        self.source = source
        self.queue = queue_obj
        self.drop_if_full = drop_if_full
        self.reconnect_interval = (
            reconnect_interval  # seconds between reconnect attempts
        )
        self.running = True
        self.daemon = True

        # Connection state tracking
        self.is_connected = False
        self.total_reconnects = 0
        self.total_downtime_seconds = 0
        self.last_disconnect_time = None
        self.consecutive_failures = 0
        self.frames_read = 0
        self.corrupt_frames = 0
        self.dropped_frames = 0  # Track frames dropped due to full queue

        # Initial connection
        self.cap = None
        self._connect()

    def _get_system_stats(self):
        """Get CPU usage statistics."""
        stats = {}

        # CPU info
        try:
            process = psutil.Process()
            stats["cpu_affinity"] = (
                list(process.cpu_affinity())
                if hasattr(process, "cpu_affinity")
                else "N/A"
            )
            stats["cpu_count"] = psutil.cpu_count()
            stats["cpu_percent"] = psutil.cpu_percent(interval=0.1)
        except Exception:
            stats["cpu_affinity"] = "N/A"
            stats["cpu_count"] = "N/A"
            stats["cpu_percent"] = "N/A"

        return stats

    def _connect(self):
        """Attempt to connect to the stream with logging."""
        attempt_time = datetime.now().isoformat()

        # Get system stats
        stats = self._get_system_stats()

        logger.info(
            f"STREAM_CONNECT_ATTEMPT | timestamp={attempt_time} | source={self.source} | "
            f"cpu_affinity={stats.get('cpu_affinity')} | cpu_count={stats.get('cpu_count')}"
        )

        self.cap = cv2.VideoCapture(self.source)

        if self.cap.isOpened():
            # Try to read a test frame to verify connection
            ret, _ = self.cap.read()
            if ret:
                self.is_connected = True
                self.consecutive_failures = 0
                connect_time = datetime.now().isoformat()

                # Get stream properties
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                logger.info(
                    f"STREAM_CONNECTED | timestamp={connect_time} | source={self.source} | "
                    f"resolution={width}x{height} | fps={fps} | total_reconnects={self.total_reconnects} | "
                    f"cpu_affinity={stats.get('cpu_affinity')}"
                )
                return True

        self.is_connected = False
        logger.warning(
            f"STREAM_CONNECT_FAILED | timestamp={attempt_time} | source={self.source} | "
            f"cpu_affinity={stats.get('cpu_affinity')}"
        )
        return False

    def run(self):
        logger.info(f"FrameReader thread started for source: {self.source}")

        while self.running:
            # Handle disconnected state
            if not self.is_connected or not self.cap or not self.cap.isOpened():
                if self.last_disconnect_time is None:
                    self.last_disconnect_time = time.time()
                    disconnect_iso = datetime.now().isoformat()

                    # Get system stats on disconnect
                    stats = self._get_system_stats()

                    logger.warning(
                        f"STREAM_DISCONNECT | timestamp={disconnect_iso} | source={self.source} | "
                        f"frames_read={self.frames_read} | corrupt_frames={self.corrupt_frames} | "
                        f"cpu_affinity={stats.get('cpu_affinity')}"
                    )

                # Attempt reconnection
                self.consecutive_failures += 1
                reconnect_attempt_time = datetime.now().isoformat()
                logger.info(
                    f"STREAM_RECONNECT_ATTEMPT | timestamp={reconnect_attempt_time} | "
                    f"attempt={self.consecutive_failures} | interval={self.reconnect_interval}s | source={self.source}"
                )

                if self._try_reconnect():
                    # Calculate downtime
                    if self.last_disconnect_time:
                        downtime = time.time() - self.last_disconnect_time
                        self.total_downtime_seconds += downtime
                        self.total_reconnects += 1

                        reconnect_iso = datetime.now().isoformat()
                        logger.info(
                            f"STREAM_RECONNECTED | timestamp={reconnect_iso} | "
                            f"downtime_seconds={round(downtime, 2)} | attempts_used={self.consecutive_failures} | "
                            f"total_reconnects={self.total_reconnects} | total_downtime={round(self.total_downtime_seconds, 2)}s"
                        )

                    self.last_disconnect_time = None
                    self.consecutive_failures = 0
                else:
                    time.sleep(self.reconnect_interval)
                continue

            # Read frame
            ret, frame = self.cap.read()

            if not ret:
                self.corrupt_frames += 1

                # Log every 10th corrupt frame to avoid spam
                if self.corrupt_frames % 10 == 1:
                    corrupt_time = datetime.now().isoformat()
                    logger.warning(
                        f"FRAME_READ_FAILURE | timestamp={corrupt_time} | "
                        f"corrupt_count={self.corrupt_frames} | frames_read={self.frames_read}"
                    )

                # Check if too many consecutive failures
                self.consecutive_failures += 1
                if (
                    self.consecutive_failures >= 30
                ):  # 30 consecutive failures = stream issue
                    self.is_connected = False
                    logger.error(
                        f"STREAM_UNSTABLE | timestamp={datetime.now().isoformat()} | "
                        f"consecutive_failures={self.consecutive_failures} | marking disconnected"
                    )

                time.sleep(0.1)
                continue

            # Successful frame read
            self.frames_read += 1
            self.consecutive_failures = 0  # Reset on successful read

            try:
                # If drop_if_full is True, drop oldest to make space
                if self.drop_if_full and self.queue.full():
                    try:
                        _ = self.queue.get_nowait()
                        self.dropped_frames += 1

                        # Log every 10th dropped frame to avoid spam
                        if self.dropped_frames % 10 == 1:
                            logger.warning(
                                f"QUEUE_FULL | timestamp={datetime.now().isoformat()} | "
                                f"queue_size={self.queue.qsize()}/{self.queue.maxsize} | "
                                f"dropped_frames={self.dropped_frames} | frames_read={self.frames_read}"
                            )
                    except queue.Empty:
                        pass

                self.queue.put(frame, block=False)
            except queue.Full:
                self.dropped_frames += 1
                logger.warning(
                    f"QUEUE_FULL_DROP | timestamp={datetime.now().isoformat()} | "
                    f"dropped_frames={self.dropped_frames}"
                )

    def _try_reconnect(self):
        """Attempt to reconnect to the stream."""
        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(self.source)

        if self.cap.isOpened():
            ret, _ = self.cap.read()
            if ret:
                self.is_connected = True
                return True

        self.is_connected = False
        return False

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

        # Log final statistics
        stop_time = datetime.now().isoformat()
        logger.info(
            f"STREAM_READER_STOPPED | timestamp={stop_time} | "
            f"total_frames={self.frames_read} | corrupt_frames={self.corrupt_frames} | "
            f"total_reconnects={self.total_reconnects} | total_downtime={round(self.total_downtime_seconds, 2)}s"
        )

    def get_stats(self):
        """Return current statistics for monitoring."""
        return {
            "is_connected": self.is_connected,
            "frames_read": self.frames_read,
            "corrupt_frames": self.corrupt_frames,
            "total_reconnects": self.total_reconnects,
            "total_downtime_seconds": round(self.total_downtime_seconds, 2),
            "queue_size": self.queue.qsize(),
            "queue_max": self.queue.maxsize,
        }


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


class EntryExitCountingService:
    """
    Service for counting customer entry/exit events using Remote YOLO + Local ByteTrack.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the entry/exit counting service
        """
        self.service_dir = Path(__file__).parent.resolve()
        self.project_root = self.service_dir.parent.parent.resolve()
        self.config_path = None
        self.service_url = YOLO_SERVICE_URL

        # Load configuration
        self.settings = self.load_settings(config_path)

        # Redis Config
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", "6379"))
        self.output_queue = os.getenv("OUTPUT_QUEUE_NAME_1", "entry_exit_output_queue")
        self.redis_retries = 3

        self.redis = None
        self.redis_connected = False
        self._try_connect_redis()

        # Database Config
        self.config_db_path = os.path.join(self.project_root, "sql_db", "configs.db")
        self.camera_config_name = "test_entry_exit"
        self.camera_id_key = "entrance"

        # Tracking state
        self.tracker = None
        self.track_history = defaultdict(list)
        self.track_classes = defaultdict(list)
        self.processed_states = defaultdict(lambda: {"entered": False, "exited": False})
        self.track_last_seen = {}  # Track ID -> last frame number seen
        self.age_gender_dict={0:"child",1:"adult_male",2:"adult_female"}

        # Load Resources
        self.init_tracker()
        self.load_camera_config()

        # Define Logic Sequences
        self.ENTRY_SEQUENCES = {
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 3),
            (2, 4),
            (1, 2, 3),
            (1, 2, 4),
            (1, 3, 4),
            (2, 3, 4),
            (1, 2, 3, 4),
            (3, 2, 3, 4),
        }
        self.EXIT_SEQUENCES = {
            (2, 1),
            (3, 1),
            (4, 1),
            (3, 2),
            (4, 2),
            (3, 2, 1),
            (4, 2, 1),
            (4, 3, 1),
            (4, 3, 2),
            (4, 3, 2, 1),
            (2, 3, 2, 1),
            (1, 3, 2, 1),
        }

        logger.info(
            f"EntryExit Service initialized. Using Remote YOLO: {self.service_url}"
        )

        # Setup visualization queue if enabled
        if VISUALIZE and self.redis_connected:
            self.vis_queue = FixedSizeQueue(
                self.redis, f"stream:visualization:{CAMERA_ID}", 10
            )
            logger.info(
                f"Visualization queue initialized: stream:visualization:{CAMERA_ID}"
            )

    def _try_connect_redis(self):
        try:
            self.redis = redis.Redis(
                host=self.redis_host, port=self.redis_port, decode_responses=False
            )
            self.redis.ping()
            self.redis_connected = True
            logger.info(f"Connected to Redis at {self.redis_host}:{self.redis_port}")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_connected = False

    def load_settings(self, config_path: Optional[str] = None) -> Dict:
        if config_path is None:
            config_path = CONFIG_PATH
        return {}  # Placeholder if needed

    def init_tracker(self):
        """Initialize local ByteTracker with configuration from YAML file."""
        if BYTETracker:
            # Load tracker configuration from YAML
            tracker_config_path = (
                self.service_dir / "artifacts" / "custom_bytetrack.yaml"
            )

            # Default values (fallback if YAML loading fails)
            tracker_config = {
                "track_high_thresh": 0.25,
                "track_low_thresh": 0.1,
                "new_track_thresh": 0.25,
                "track_buffer": 30,
                "match_thresh": 0.8,
                "fuse_score": True,
                "frame_rate": 30,
            }

            if tracker_config_path.exists():
                try:
                    with open(tracker_config_path, "r") as f:
                        yaml_config = yaml.safe_load(f)
                        tracker_config.update(yaml_config)
                    logger.info(f"Loaded tracker config from: {tracker_config_path}")
                except Exception as e:
                    logger.warning(
                        f"Failed to load tracker config, using defaults: {e}"
                    )
            else:
                logger.warning(
                    f"Tracker config not found at {tracker_config_path}, using defaults"
                )

            # Create arguments object for BYTETracker
            class TrackerArgs:
                def __init__(self, config):
                    self.track_high_thresh = config.get("track_high_thresh", 0.25)
                    self.track_low_thresh = config.get("track_low_thresh", 0.1)
                    self.new_track_thresh = config.get("new_track_thresh", 0.25)
                    self.track_buffer = config.get("track_buffer", 30)
                    self.match_thresh = config.get("match_thresh", 0.8)
                    self.fuse_score = config.get("fuse_score", True)
                    self.mot20 = config.get("mot20", False)
                    self.frame_rate = config.get("frame_rate", 30)

            self.tracker = BYTETracker(TrackerArgs(tracker_config))
            logger.info(
                f"Local ByteTracker initialized with config: track_thresh={tracker_config.get('track_high_thresh')}, "
                f"track_buffer={tracker_config.get('track_buffer')}, match_thresh={tracker_config.get('match_thresh')}"
            )
        else:
            logger.error("ByteTracker could not be initialized!")

    def load_camera_config(self):
        """Load polygon regions from config.json file."""
        # Initialize polygons as empty list (fallback if config not found)
        self.polygons = []

        # Load from config.json in the service directory
        config_file_path = self.service_dir / "config.json"

        if not config_file_path.exists():
            logger.error(f"Config file not found at {config_file_path}")
            return

        try:
            with open(config_file_path, "r") as f:
                data = json.load(f)

            # Get regions for the camera_id_key (e.g., "entrance")
            regions = data.get(self.camera_id_key, [])

            for reg in regions:
                pts = [(p["x"], p["y"]) for p in reg["points"]]
                self.polygons.append(Polygon(pts))

            logger.info(f"Loaded {len(self.polygons)} regions from {config_file_path}")
        except Exception as e:
            logger.error(f"Error loading camera config: {e}")

    def get_region_index(self, bbox_bottom_center):
        point = Point(bbox_bottom_center)
        for i, poly in enumerate(self.polygons):
            if poly.contains(point):
                return i
        return -1

    def simplify_history(self, history: List[int]) -> Tuple[int]:
        if not history:
            return ()
        converted = [h + 1 for h in history]
        simplified = []
        if converted:
            simplified.append(converted[0])
            for x in converted[1:]:
                if x != simplified[-1]:
                    simplified.append(x)
        return tuple(simplified)

    def _get_detections_from_service(self, frame) -> np.ndarray:
        """
        Send frame to remote YOLO service and return detections in format expected by tracker.
        Returns: np.ndarray of shape (N, 6) -> [x1, y1, x2, y2, conf, cls]
        """
        try:
            # Compress frame to JPEG to send
            _, img_encoded = cv2.imencode(".jpg", frame)
            files = {"image": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")}

            resp = requests.post(self.service_url, files=files, timeout=5)
            resp.raise_for_status()

            # Parse response
            # Expected format: {'predict': ['[{"box": {"x1":...}, "class": 0, "confidence": 0.9}]']}
            result = resp.json()
            predict_list = result.get("predict", [])

            detections = []
            if predict_list:
                data = json.loads(predict_list[0])
                for item in data:
                    cls_ = int(item.get("class", -1))
                    # if cls_ != 0:
                    #     continue  # Only persons

                    box = item.get("box", {})
                    conf = item.get("confidence", 0.0)
                    x1, y1, x2, y2 = (
                        box.get("x1"),
                        box.get("y1"),
                        box.get("x2"),
                        box.get("y2"),
                    )

                    if x1 is not None:
                        detections.append([x1, y1, x2, y2, conf, cls_])

            if not detections:
                return np.empty((0, 6))

            return np.array(detections)

        except Exception as e:
            logger.error(f"Error calling YOLO service: {e}")
            return np.empty((0, 6))

    def _publish_event(
        self,
        event_type: str,
        track_id: int,
        frame_num: int,
        timestamp: float,
        history: List[int],
        cls=None,
    ):
        gender=None
        age_group=None
        if cls is not None:
            if cls==0:
                age_group="child"
            elif cls==1:
                age_group="adult"
                gender="male"
            elif cls==2:
                age_group="adult"
                gender="female"

        event_data = {
            "event_type": event_type,
            "timestamp": timestamp,
            "timestamp_iso": datetime.fromtimestamp(timestamp).isoformat(),
            "frame_num": frame_num,
            "track_id": track_id,
            "gender":gender,
            "age_group":age_group,
            "path": [h + 1 for h in history],
        }

        output_payload = {
            "camera_id": self.camera_id_key,
            "timestamp": timestamp,
            "event": event_data,
            "target_collection": os.getenv(
                "TARGET_COLLECTION_NAME", "entry_exit_collection"
            ),
        }

        if self.redis_connected and self.redis:
            self.redis.xadd(self.output_queue, {"data": json.dumps(output_payload)})
            demo_str = f" |{gender} | {age_group}"
            logger.info(
                f"✓ Pushed {event_type} event: ID {track_id}{demo_str} Path {event_data['path']}"
            )

    def process_video(
        self,
        video_path: Optional[str] = None,
        camera_url: Optional[str] = None,
        show_visualization: bool = True,
        save_debug_frames: bool = False,
    ):
        if not camera_url:
            camera_url = INPUT_RTSP
        source = video_path if video_path else camera_url
        if not source:
            raise ValueError("No video source provided.")

        # Setup Queue
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)

        # Signal Handler for Queue Monitoring
        def handle_sigusr1(signum, frame):
            size = self.frame_queue.qsize()
            max_size = self.frame_queue.maxsize
            print(f"\\n[MONITOR] Current Queue Size: {size} / {max_size}")
            logger.info(f"[MONITOR] Current Queue Size: {size} / {max_size}")

        # Register signal
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        logger.info(
            f"Registered SIGUSR1 handler. Run 'kill -SIGUSR1 {os.getpid()}' to check queue size."
        )

        # Start Frame Reader Thread
        # Note: Original code had complex reconnection logic in main loop.
        # Moving reading to thread simplifies main loop to just consuming info.
        # However, to maintain the logic of "reconnect attempts", FrameReader needs to handle it
        # or we just rely on FrameReader's internal simple retry.
        # For this refactor, we will instantiate FrameReader.

        frame_reader = FrameReader(source, self.frame_queue)
        frame_reader.start()

        logger.info(f"Starting processing on: {source}")
        # cap = cv2.VideoCapture(source) # Removed, handled by thread

        if show_visualization:
            cv2.namedWindow("EntryExit Service", cv2.WINDOW_NORMAL)

        # Setup debug frame saving
        debug_frames_dir = None
        save_every_n_frames = 10  # Save every 10th frame in debug mode
        if save_debug_frames:
            debug_frames_dir = self.service_dir / "debug_frames"
            debug_frames_dir.mkdir(exist_ok=True)
            logger.info(f"Debug mode: Saving frames to {debug_frames_dir}")

        frame_num = 0
        # Reconnection stats now harder to track exactly matching previous logic without shared state.
        # We'll rely on the fact that if queue is empty for too long, maybe stream is dead.

        while True:
            try:
                # Wait for frame with timeout
                frame = self.frame_queue.get(timeout=5)
                ret = True
            except queue.Empty:
                ret = False
                logger.warning("Queue empty (starved) or stream disconnected.")
                # Could implement restart logic here if queue stays empty for too long
                continue

            if not ret:
                continue  # Should be handled by queue get

            # frame_num += 1 logic...

            # --- OLD LOGIC BLOCK START (Removed/Modified) ---
            # ret, frame = cap.read()
            # ... entire reconnection block ...
            # replaced by queue get
            # --- OLD LOGIC BLOCK END ---

            frame_num += 1
            timestamp = time.time()

            # 1. Remote Detection
            det_array = self._get_detections_from_service(frame)

            current_tracks = []

            # 2. Local Tracking
            if self.tracker is not None:
                # Get frame dimensions for MockResults
                frame_height, frame_width = frame.shape[:2]
                orig_shape = (frame_height, frame_width)

                # Wrap numpy detections in MockResults for BYTETracker compatibility
                mock_results = MockResults(det_array, orig_shape)

                # BYTETracker.update() expects a Results-like object
                tracks = self.tracker.update(mock_results, frame)

                # tracks is likely (N, 7) array: [x1, y1, x2, y2, id, conf, cls] or similar

                for track in tracks:
                    # Depending on return format of ByteTracker.update
                    # Usually it returns numpy array x1, y1, x2, y2, id, score, cls
                    x1, y1, x2, y2 = map(int, track[:4])
                    track_id = int(track[4])
                    cls = int(track[-2])

                    bottom_center = ((x1 + x2) / 2, y2)
                    reg_idx = self.get_region_index(bottom_center)

                    history = self.track_history[track_id]
                    if not history:
                        if reg_idx != -1:
                            history.append(reg_idx)
                    else:
                        if reg_idx != -1 and history[-1] != reg_idx:
                            history.append(reg_idx)

                    current_tracks.append((track_id, (x1, y1, x2, y2), cls, reg_idx))
                    self.track_classes[track_id].append(cls)
                    self.track_last_seen[track_id] = frame_num  # Record last seen frame

                    simplified_seq = self.simplify_history(history)

                    if simplified_seq in self.ENTRY_SEQUENCES:
                        if not self.processed_states[track_id]["entered"]:
                            cls=mode(self.track_classes[track_id])
                            self.track_classes.pop(track_id)
                            self._publish_event(
                                "ENTRY",
                                track_id,
                                frame_num,
                                timestamp,
                                history,
                                cls=cls,
                            )
                            self.processed_states[track_id]["entered"] = True

                    if simplified_seq in self.EXIT_SEQUENCES:
                        if not self.processed_states[track_id]["exited"]:
                            cls=mode(self.track_classes[track_id])
                            self.track_classes.pop(track_id)
                            self._publish_event(
                                "EXIT",
                                track_id,
                                frame_num,
                                timestamp,
                                history,
                                cls=cls,
                            )
                            self.processed_states[track_id]["exited"] = True

            # Visualization (for display or debug saving)
            if show_visualization or save_debug_frames or VISUALIZE:
                self.draw_visualization(frame, current_tracks, frame_num)

                if show_visualization:
                    cv2.imshow("EntryExit Service", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                # Save debug frames
                if (
                    save_debug_frames
                    and debug_frames_dir
                    and (frame_num % save_every_n_frames == 0)
                ):
                    frame_filename = f"frame_{frame_num:06d}_{int(timestamp)}.jpg"
                    frame_path = debug_frames_dir / frame_filename
                    cv2.imwrite(str(frame_path), frame)
                    logger.debug(f"Saved debug frame: {frame_filename}")

                # Push visualization to Redis queue
                if VISUALIZE and self.redis_connected and hasattr(self, "vis_queue"):
                    try:
                        _, img_encoded = cv2.imencode(".jpg", frame)
                        frame_b64 = base64.b64encode(img_encoded.tobytes()).decode(
                            "utf-8"
                        )

                        # Prepare track data for visualization
                        track_data = []
                        for tid, box,cls, rid in current_tracks:
                            history = self.track_history.get(tid, [])
                            states = self.processed_states.get(tid, {})
                            
                            # Decode class to gender and age_group
                            # 0: child, 1: adult-male, 2: adult-female
                            gender = None
                            age_group = None
                            if cls == 0:
                                age_group = "child"
                            elif cls == 1:
                                gender = "male"
                                age_group = "adult"
                            elif cls == 2:
                                gender = "female"
                                age_group = "adult"
                            
                            track_data.append(
                                {
                                    "track_id": tid,
                                    "bbox": list(box),
                                    "region": rid,
                                    "path": [h + 1 for h in history],
                                    "entered": states.get("entered", False),
                                    "exited": states.get("exited", False),
                                    "cls": cls,
                                    "gender": gender,
                                    "age_group": age_group,
                                }
                            )

                        vis_payload = {
                            "camera_id": CAMERA_ID,
                            "timestamp": datetime.fromtimestamp(timestamp).isoformat(),
                            "frame": frame_b64,
                            "frame_num": frame_num,
                            "tracks": track_data,
                            "num_tracks": len(current_tracks),
                        }
                        self.vis_queue.push(json.dumps(vis_payload))
                    except Exception as e:
                        logger.error(f"Failed to push visualization frame: {e}")

        frame_reader.stop()
        if show_visualization:
            cv2.destroyAllWindows()
        if save_debug_frames:
            logger.info(f"Debug frames saved to: {debug_frames_dir}")

    def draw_visualization(self, frame, tracks, frame_num):
        """Draw regions, bounding boxes, track IDs, and paths on the frame."""
        # Define colors for each region
        region_colors = [
            (255, 255, 0),  # Cyan - Region 1
            (255, 0, 255),  # Magenta - Region 2
            (0, 255, 255),  # Yellow - Region 3
            (0, 255, 0),  # Green - Region 4
        ]

        # Draw polygon regions with labels
        for i, poly in enumerate(self.polygons):
            if poly.is_empty:
                continue
            pts = np.array(poly.exterior.coords, np.int32).reshape((-1, 1, 2))
            color = region_colors[i % len(region_colors)]
            cv2.polylines(frame, [pts], True, color, 2)

            # Add region label at centroid
            centroid = poly.centroid
            cv2.putText(
                frame,
                f"R{i+1}",
                (int(centroid.x), int(centroid.y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

        # Draw tracked objects with IDs, boxes, and paths
        for tid, box, cls, rid in tracks:
            x1, y1, x2, y2 = box

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Get track history/path for this ID
            history = self.track_history.get(tid, [])
            path_str = "->".join([str(h + 1) for h in history]) if history else "N/A"

            # Decode class to human-readable label
            # 0: child, 1: adult-male, 2: adult-female
            cls_label = self.age_gender_dict.get(cls, str(cls))

            # Draw track ID and current region
            region_text = f"R{rid+1}" if rid != -1 else "?"
            label = f"ID:{tid} [{region_text}] | {cls_label}"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            # Draw path below the box
            cv2.putText(
                frame,
                f"Path:{path_str}",
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

        # Draw frame number
        cv2.putText(
            frame,
            f"Frame: {frame_num}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Draw summary panel with all track crossing sequences (top-right corner)
        frame_height, frame_width = frame.shape[:2]
        panel_x = frame_width - 380
        panel_y = 10
        line_height = 20
        stale_threshold = (
            60  # Frames before a track is considered old and removed from panel
        )

        # Get current track IDs
        current_track_ids = set(tid for tid, _,_, _ in tracks)

        # Collect all tracks: active + recently stale
        all_display_tracks = []

        # Add active tracks first
        for tid, _,_, _ in tracks:
            history = self.track_history.get(tid, [])
            states = self.processed_states.get(tid, {})
            all_display_tracks.append((tid, history, states, True))  # True = active

        # Add stale tracks (recently seen but not currently visible)
        for tid, last_frame in self.track_last_seen.items():
            if tid not in current_track_ids:
                frames_ago = frame_num - last_frame
                if frames_ago <= stale_threshold:
                    history = self.track_history.get(tid, [])
                    states = self.processed_states.get(tid, {})
                    all_display_tracks.append(
                        (tid, history, states, False)
                    )  # False = stale

        if all_display_tracks:
            # Sort: active first, then stale by recency
            all_display_tracks.sort(
                key=lambda x: (not x[3], -self.track_last_seen.get(x[0], 0))
            )

            panel_height = 50 + len(all_display_tracks) * line_height
            max_panel_height = frame_height - 20
            panel_height = min(panel_height, max_panel_height)

            cv2.rectangle(
                frame,
                (panel_x - 10, panel_y - 5),
                (frame_width - 10, panel_y + panel_height),
                (0, 0, 0),
                -1,
            )
            cv2.rectangle(
                frame,
                (panel_x - 10, panel_y - 5),
                (frame_width - 10, panel_y + panel_height),
                (255, 255, 255),
                1,
            )

            # Panel title with counts
            active_count = sum(
                1 for _, _, _, is_active in all_display_tracks if is_active
            )
            stale_count = len(all_display_tracks) - active_count
            title = f"TRACK CROSSINGS (Active:{active_count} Stale:{stale_count})"
            cv2.putText(
                frame,
                title,
                (panel_x, panel_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
            )

            # Draw each track's crossing sequence
            max_rows = (panel_height - 40) // line_height
            for i, (tid, history, states, is_active) in enumerate(
                all_display_tracks[:max_rows]
            ):
                y_pos = panel_y + 35 + i * line_height
                path_str = (
                    "->".join([str(h + 1) for h in history]) if history else "..."
                )

                # Determine status
                entered = states.get("entered", False)
                exited = states.get("exited", False)
                if entered:
                    status = "[ENTRY]"
                    status_color = (0, 255, 0)  # Green
                elif exited:
                    status = "[EXIT]"
                    status_color = (0, 0, 255)  # Red
                else:
                    status = ""
                    status_color = (255, 255, 255)

                # Active vs stale styling
                if is_active:
                    text_color = (255, 255, 255)  # White for active
                    prefix = "●"
                else:
                    text_color = (128, 128, 128)  # Gray for stale
                    prefix = "○"

                # Draw track info
                gender = states.get("gender")
                age = states.get("age")
                demo_str = f"({gender}/{age})" if gender or age else ""

                cv2.putText(
                    frame,
                    f"{prefix} ID {tid}: {path_str} {demo_str}",
                    (panel_x, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    text_color,
                    1,
                )
                if status:
                    cv2.putText(
                        frame,
                        status,
                        (panel_x + 230, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        status_color,
                        1,
                    )


def main():
    parser = argparse.ArgumentParser(description="Entry/Exit Counting Service")
    parser.add_argument(
        "--visualize", action="store_true", help="Show live visualization window"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save processed frames with visualization to debug_frames folder",
    )
    args = parser.parse_args()

    service = EntryExitCountingService()
    service.process_video(show_visualization=args.visualize, save_debug_frames=args.debug)


if __name__ == "__main__":
    main()
