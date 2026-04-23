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
from collections import deque, Counter
import concurrent.futures
import uuid
from concurrent.futures import ThreadPoolExecutor

from detector import DrawerDetector
from config import Config

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
LOKI_URL = os.getenv("LOKI_URL", "http://localhost:3100/loki/api/v1/push")

INPUT_QUEUE = os.getenv("INPUT_QUEUE_NAME")
OUTPUT_QUEUE = os.getenv("OUTPUT_QUEUE_NAME")
GROUP_NAME = os.getenv("GROUP_NAME")
CAMERA_ID = os.getenv("CAMERA_ID", INPUT_QUEUE.split(":")[-1] if INPUT_QUEUE else "unknown")
VISUALIZE = str(os.getenv("VISUALIZE", "False")).lower() == "true"
ACK_TTL_SECONDS = 10 * 60  # 10 minutes

# Per-drawer detection timeout (seconds). If a single drawer detection
# takes longer than this, the prediction defaults to CLOSED (PASS).
DRAWER_DETECT_TIMEOUT = float(os.getenv("DRAWER_DETECT_TIMEOUT", "5.0"))

# Consecutive duration (seconds) required before a drawer's confirmed
# state actually changes.  E.g. 10 → the new raw state must persist for
# 10 continuous seconds before `confirmed_state` flips.
STATE_CHANGE_DURATION = float(os.getenv("STATE_CHANGE_DURATION", "10.0"))

# YOLO and Alert Configuration
YOLO_MODEL_URL = os.getenv("YOLO_MODEL_URL")
ALERT_QUEUE = os.getenv("ALERT_QUEUE_NAME", "database:queue:0")
OVERLAP_THRESHOLD = float(os.getenv("OVERLAP_THRESHOLD", "0.3"))

# Configure Logging
handler = logging_loki.LokiQueueHandler(
    Queue(-1),
    url=LOKI_URL,
    tags={"application": "drawer_detection_service"},
    version="1",
)

logger = logging.getLogger("drawer_detection_service")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
# Also log to console
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)


# ── helper: per-drawer state machine ────────────────────────────────
class DrawerState:
    """Tracks the confirmed state of a single drawer with hysteresis.

    * `confirmed_state` — the "official" state used for alerts / output.
      It starts as CLOSED and only changes after the raw detected state
      has been different for `STATE_CHANGE_DURATION` consecutive seconds.
    * `consecutive_state` / `consecutive_start` — tracks the tentative
      new state and when it started.
    """

    CLOSED = "CLOSED"  # drawer is closed → PASS
    OPEN   = "OPEN"    # drawer is open   → FAIL

    def __init__(self, name: str):
        self.name = name
        self.confirmed_state: str = self.CLOSED
        self.consecutive_state: str | None = None
        self.consecutive_start: float | None = None

    def update(self, detected_closed: bool) -> str:
        """Feed a new raw detection and return the (possibly unchanged)
        confirmed state."""
        raw = self.CLOSED if detected_closed else self.OPEN
        now = time.time()

        if raw == self.confirmed_state:
            # Detection agrees with confirmed → reset any pending transition
            self.consecutive_state = None
            self.consecutive_start = None
        else:
            # Detection disagrees
            if self.consecutive_state == raw:
                # Already counting towards this transition
                elapsed = now - self.consecutive_start
                if elapsed >= STATE_CHANGE_DURATION:
                    logger.info(
                        f"[{self.name}] State changed: "
                        f"{self.confirmed_state} → {raw} "
                        f"(after {elapsed:.1f}s consecutive)"
                    )
                    self.confirmed_state = raw
                    self.consecutive_state = None
                    self.consecutive_start = None
            else:
                # Start counting for a new candidate state
                self.consecutive_state = raw
                self.consecutive_start = now

        return self.confirmed_state

    @property
    def is_closed(self) -> bool:
        return self.confirmed_state == self.CLOSED


class FixedSizeQueue:
    def __init__(self, redis_client, name, max_size):
        self.r = redis_client
        self.name = name
        self.max_size = max_size

    def push(self, item):
        """Push item to queue. Oldest items are dropped."""
        pipe = self.r.pipeline()
        pipe.rpush(self.name, item)
        pipe.ltrim(self.name, -self.max_size, -1)
        pipe.execute()


# ── main service ────────────────────────────────────────────────────
class DrawerDetectionService:
    def __init__(self):
        # Redis connection
        self.redis = self.connect_redis(host=REDIS_HOST, port=REDIS_PORT)

        # Detector
        self.detector = DrawerDetector()

        # Thread pool for per-drawer timeout detection
        self._executor = ThreadPoolExecutor(max_workers=2)

        # Per-drawer independent state machines
        self.front_state = DrawerState("FRONT")
        self.back_state  = DrawerState("BACK")

        # Person smoothing (kept as before)
        self.person_history_size = int(os.getenv("PERSON_HISTORY_SIZE", "10"))
        self.person_in_region_history = deque(maxlen=self.person_history_size)

        # Alert State tracking
        self.current_alert_id = None
        self.last_alert_state = False

        # Load settings for regions
        self.settings = self.load_settings()

        # Setup visualization queue if enabled
        if VISUALIZE:
            self.vis_queue = FixedSizeQueue(
                self.redis, f"stream:visualization:drawer:{CAMERA_ID}", 10
            )

        logger.info(
            f"DrawerDetectionService initialized. Camera: {CAMERA_ID}, "
            f"Detect timeout: {DRAWER_DETECT_TIMEOUT}s, "
            f"State change duration: {STATE_CHANGE_DURATION}s"
        )

    # ── Redis helpers ───────────────────────────────────────────────
    def connect_redis(self, host, port, retry_delay=5):
        while True:
            try:
                r = redis.Redis(host=host, port=port, decode_responses=False)
                r.ping()
                logging.info("Connected to Redis successfully.")
                return r
            except Exception as e:
                logging.error(
                    f"Redis connection failed: {e}. Retrying in {retry_delay}s…"
                )
                time.sleep(retry_delay)

    def mark_acked(self, stream: str, msg_id: str, consumer: str):
        cnt_key = f"ack:cnt:{stream}:{msg_id}"
        seen_key = f"ack:seen:{stream}:{msg_id}"
        pipe = self.redis.pipeline()
        pipe.sadd(seen_key, consumer)
        pipe.expire(seen_key, ACK_TTL_SECONDS)
        results = pipe.execute()
        if results[0]:  # was_added
            pipe = self.redis.pipeline()
            pipe.incr(cnt_key)
            pipe.expire(cnt_key, ACK_TTL_SECONDS)
            pipe.execute()

    # ── config / pre-processing ─────────────────────────────────────
    def load_settings(self):
        """Load detector configuration from setting.json."""
        try:
            settings_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "setting.json"
            )
            with open(settings_path, "r") as f:
                data = json.load(f)
            return data.get(str(CAMERA_ID), {})
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
            return {}

    def preprocessing(self, frame_url):
        """Download and decode the frame."""
        try:
            resp = requests.get(frame_url, timeout=10)
            resp.raise_for_status()
            arr = np.frombuffer(resp.content, np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error(f"Failed to download/decode frame from {frame_url}: {e}")
            return None

    # ── YOLO person detection ───────────────────────────────────────
    def get_yolo_detections(self, frame_url: str):
        if not YOLO_MODEL_URL:
            return []
        try:
            resp = requests.post(
                YOLO_MODEL_URL, json={"image_url": frame_url}, timeout=10
            )
            resp.raise_for_status()
            predict_list = resp.json().get("predict", [])
            if not predict_list:
                return []
            parsed = json.loads(predict_list[0])
            return [p for p in parsed if p.get("class") == 0]
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return []

    def check_person_in_region(self, person_detections, frame_shape):
        counter_polys_config = self.settings.get("counter_polygons", [])
        if not counter_polys_config or not person_detections:
            return False

        h, w = frame_shape[:2]
        pixel_polys = []
        for poly in counter_polys_config:
            normalized = poly.get("normalized", [])
            pts = np.array(
                [[int(pt["x"] * w), int(pt["y"] * h)] for pt in normalized],
                dtype=np.int32,
            )
            pixel_polys.append(pts)

        for person in person_detections:
            box = person.get("box", {})
            x1, y1, x2, y2 = (
                box.get("x1"), box.get("y1"),
                box.get("x2"), box.get("y2"),
            )
            if x1 is None:
                continue
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            box_area = (x2 - x1) * (y2 - y1)
            if box_area <= 0:
                continue
            for poly in pixel_polys:
                if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                    return True
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [poly], 255)
                roi = mask[int(y1):int(y2), int(x1):int(x2)]
                if roi.size > 0:
                    intersection = cv2.countNonZero(roi)
                    if intersection / box_area >= OVERLAP_THRESHOLD:
                        return True
        return False

    # ── per-drawer detection with timeout ───────────────────────────
    def _detect_front(self, image):
        """Run _try_fit_pair for the FRONT drawer only."""
        return self.detector._try_fit_pair(
            image,
            strip_bright=self.detector.yellow,
            strip_dark=self.detector.cyan,
            checking_region=self.detector.red_check,
            bright_min_thresh=Config.BRIGHT_MIN_THRESHOLD,
            dark_max_thresh=Config.DARK_MAX_THRESHOLD,
        )

    def _detect_back(self, image):
        """Run _try_fit_pair for the BACK drawer only."""
        return self.detector._try_fit_pair(
            image,
            strip_bright=self.detector.green,
            strip_dark=self.detector.pink,
            checking_region=self.detector.blue_check,
            bright_min_thresh=Config.BRIGHT_MIN_THRESHOLD,
            dark_max_thresh=Config.DARK_MAX_THRESHOLD,
        )

    def _detect_both_drawers_parallel(self, image):
        """Run front and back drawer detection in parallel threads.

        Both share a single wall-clock deadline of DRAWER_DETECT_TIMEOUT
        seconds, so the maximum total time is ~5 s instead of 10 s.

        Returns:
            (front_fit, front_timed_out, back_fit, back_timed_out)
        """
        deadline = time.time() + DRAWER_DETECT_TIMEOUT

        front_fut = self._executor.submit(self._detect_front, image)
        back_fut = self._executor.submit(self._detect_back, image)

        # --- collect front result ---
        remaining = max(deadline - time.time(), 0)
        try:
            front_fit = front_fut.result(timeout=remaining)
            front_timed_out = False
        except concurrent.futures.TimeoutError:
            logger.warning(
                f"[FRONT] Detection timed out "
                f"(>{DRAWER_DETECT_TIMEOUT}s) → defaulting to CLOSED"
            )
            front_fut.cancel()
            front_fit, front_timed_out = None, True

        # --- collect back result (use whatever time remains) ---
        remaining = max(deadline - time.time(), 0)
        try:
            back_fit = back_fut.result(timeout=remaining)
            back_timed_out = False
        except concurrent.futures.TimeoutError:
            logger.warning(
                f"[BACK] Detection timed out "
                f"(>{DRAWER_DETECT_TIMEOUT}s) → defaulting to CLOSED"
            )
            back_fut.cancel()
            back_fit, back_timed_out = None, True

        return front_fit, front_timed_out, back_fit, back_timed_out

    # ── drain-and-process loop ──────────────────────────────────────
    def _drain_to_newest(self):
        """Read ALL pending messages from the consumer group, ack every
        one except the newest, and return (newest_msg_id, newest_message).
        Returns (None, None) if there are no messages at all.
        """
        newest_id = None
        newest_msg = None
        ids_to_ack = []

        while True:
            resp = self.redis.xreadgroup(
                groupname=GROUP_NAME,
                consumername="worker-1",
                streams={INPUT_QUEUE: ">"},
                count=50,
                block=500,   # short block so we don't wait forever
            )
            if not resp:
                break

            stream_name, messages = resp[0]
            for msg_id, message in messages:
                if newest_id is not None:
                    ids_to_ack.append(newest_id)
                newest_id = msg_id
                newest_msg = message

        # Ack all older messages
        for old_id in ids_to_ack:
            self.redis.xack(INPUT_QUEUE, GROUP_NAME, old_id)
            self.mark_acked(INPUT_QUEUE, old_id.decode(), GROUP_NAME)

        if ids_to_ack:
            logger.info(f"Drained {len(ids_to_ack)} old frame(s), keeping newest")

        return newest_id, newest_msg

    # ── main processing ─────────────────────────────────────────────
    def process_frame(self, message: dict, frame_url: str, timestamp_str: str):
        """Process a single frame: detect both drawers independently
        with per-drawer timeout, update state machines, run alert logic,
        and push visualisation."""

        start_time = time.time()
        frame_img = self.preprocessing(frame_url)
        if frame_img is None:
            return

        try:
            # ── Detect both drawers in parallel (shared 5s deadline) ─
            front_fit, front_timed_out, back_fit, back_timed_out = \
                self._detect_both_drawers_parallel(frame_img)

            process_duration = time.time() - start_time

            # Determine raw per-frame detection:
            # fit != None → CLOSED (PASS), fit == None → OPEN (FAIL)
            # BUT if timed out → force CLOSED (PASS)
            front_detected_closed = (front_fit is not None) or front_timed_out
            back_detected_closed  = (back_fit is not None)  or back_timed_out

            # ── Update independent state machines (10s hysteresis) ──
            front_confirmed = self.front_state.update(front_detected_closed)
            back_confirmed  = self.back_state.update(back_detected_closed)

            front_is_closed = self.front_state.is_closed
            back_is_closed  = self.back_state.is_closed

            # ── Output payload ──────────────────────────────────────
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj

            output_payload = {
                "camera_id": CAMERA_ID,
                "timestamp": timestamp_str,
                "front_fit": front_fit,
                "back_fit": back_fit,
                "front_raw_detection": "CLOSED" if front_detected_closed else "OPEN",
                "back_raw_detection": "CLOSED" if back_detected_closed else "OPEN",
                "front_confirmed_state": front_confirmed,
                "back_confirmed_state": back_confirmed,
                "front_timed_out": front_timed_out,
                "back_timed_out": back_timed_out,
                "target_collection": os.getenv(
                    "TARGET_COLLECTION_NAME", "drawer_detection_collection"
                ),
                "processing_time": process_duration,
            }

            if OUTPUT_QUEUE:
                self.redis.xadd(
                    OUTPUT_QUEUE,
                    {"data": json.dumps(output_payload, default=convert_numpy)},
                )

            # ── Alert logic ─────────────────────────────────────────
            person_detections = self.get_yolo_detections(frame_url)
            person_in_region = self.check_person_in_region(
                person_detections, frame_img.shape
            )
            self.person_in_region_history.append(1 if person_in_region else 0)
            person_stable = (
                Counter(self.person_in_region_history).most_common(1)[0][0] == 1
            )

            drawer_open = (not front_is_closed) or (not back_is_closed)
            stable_alert_state = (not person_stable) and drawer_open

            if stable_alert_state and not self.last_alert_state:
                self.current_alert_id = str(uuid.uuid4())
                self.last_alert_state = True
                desc_parts = []
                if not front_is_closed:
                    desc_parts.append("Front drawer OPEN")
                if not back_is_closed:
                    desc_parts.append("Back drawer OPEN")
                desc = (
                    ", ".join(desc_parts)
                    + " — no customer present in counter region."
                )
                self.publish_alert(frame_img, timestamp_str, "raised", desc)
            elif not stable_alert_state and self.last_alert_state:
                self.publish_alert(
                    frame_img,
                    timestamp_str,
                    "resolved",
                    "Drawer(s) closed or customer arrived in counter region.",
                )
                self.last_alert_state = False
                self.current_alert_id = None

            # ── Visualization ───────────────────────────────────────
            if VISUALIZE and hasattr(self, "vis_queue"):
                results = {"front_fit": front_fit, "back_fit": back_fit}
                self.handle_visualization(
                    frame_img,
                    results,
                    frame_url,
                    timestamp_str,
                    front_is_closed,
                    back_is_closed,
                    front_detected_closed,
                    back_detected_closed,
                    front_timed_out,
                    back_timed_out,
                    person_stable,
                    stable_alert_state,
                    process_duration,
                )

            logger.info(
                f"Processed {timestamp_str} in {process_duration:.1f}s | "
                f"Front: raw={'CLOSED' if front_detected_closed else 'OPEN'}"
                f"{'(timeout)' if front_timed_out else ''} "
                f"confirmed={front_confirmed} | "
                f"Back: raw={'CLOSED' if back_detected_closed else 'OPEN'}"
                f"{'(timeout)' if back_timed_out else ''} "
                f"confirmed={back_confirmed} | "
                f"Person={person_stable} Alert={stable_alert_state}"
            )

        except Exception as e:
            logger.error(f"Error processing frame: {e}", exc_info=True)

    # ── alerts ──────────────────────────────────────────────────────
    def publish_alert(self, frame_img, timestamp, alert_type, description):
        if not ALERT_QUEUE:
            return
        try:
            _, buffer = cv2.imencode(".jpg", frame_img)
            frame_b64 = base64.b64encode(buffer).decode("utf-8")
            alert_payload = {
                "camera_id": CAMERA_ID,
                "timestamp": timestamp,
                "frame": frame_b64,
                "description": description,
                "target_collection": "alert_collection",
                "type": alert_type,
                "ticket_id": self.current_alert_id,
            }
            self.redis.xadd(ALERT_QUEUE, {"data": json.dumps(alert_payload)})
            logger.info(
                f"ALERT {alert_type.upper()}: {description} "
                f"(ID: {self.current_alert_id})"
            )
        except Exception as e:
            logger.error(f"Failed to publish alert: {e}")

    # ── visualization ───────────────────────────────────────────────
    def handle_visualization(
        self,
        frame_img,
        results,
        frame_url,
        timestamp,
        front_is_closed,
        back_is_closed,
        front_detected_closed,
        back_detected_closed,
        front_timed_out,
        back_timed_out,
        person_stable,
        alert_active,
        process_duration,
    ):
        try:
            viz_img = frame_img.copy()
            h_img, w_img = viz_img.shape[:2]

            # ── Draw fitted polygons on the frame ───────────────────
            front = results.get("front_fit")
            if front:
                pts_b = np.array(front["points_bright"], dtype=np.int32).reshape((-1, 1, 2))
                pts_d = np.array(front["points_dark"], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(viz_img, [pts_b], True, (0, 255, 255), 2)
                cv2.polylines(viz_img, [pts_d], True, (255, 255, 0), 2)

            back = results.get("back_fit")
            if back:
                pts_b = np.array(back["points_bright"], dtype=np.int32).reshape((-1, 1, 2))
                pts_d = np.array(back["points_dark"], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(viz_img, [pts_b], True, (0, 200, 0), 2)
                cv2.polylines(viz_img, [pts_d], True, (200, 0, 200), 2)

            # Counter region polygon overlay
            for poly in self.settings.get("counter_polygons", []):
                norm = poly.get("normalized", [])
                pts = np.array(
                    [
                        [int(pt["x"] * w_img), int(pt["y"] * h_img)]
                        for pt in norm
                    ],
                    dtype=np.int32,
                )
                cv2.polylines(viz_img, [pts], True, (255, 0, 0), 1)

            # Alert red border
            if alert_active:
                cv2.rectangle(
                    viz_img, (0, 0), (w_img, h_img), (0, 0, 255), 10,
                )

            # ── Build text lines for the info panel ─────────────────
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            line_height = 18

            # Thresholds from Config
            bright_thr = Config.BRIGHT_MIN_THRESHOLD
            bright_avg_thr = Config.BRIGHT_AVG_MIN_THRESHOLD
            dark_thr = Config.DARK_MAX_THRESHOLD
            dark_avg_thr = Config.DARK_AVG_MAX_THRESHOLD

            lines = []  # list of (text, color)
            WHITE = (255, 255, 255)
            GREEN = (0, 255, 0)
            RED = (0, 0, 255)
            YELLOW = (0, 255, 255)
            GRAY = (180, 180, 180)
            CYAN = (255, 255, 0)

            # ─── Header ─────────────────────────────────────────────
            lines.append((f"Camera: {CAMERA_ID}  |  {timestamp}", CYAN))
            lines.append((f"Processing: {process_duration:.2f}s  |  Timeout: {DRAWER_DETECT_TIMEOUT}s", WHITE))
            lines.append(("", WHITE))  # spacer

            # ─── Per-drawer info ─────────────────────────────────────
            now = time.time()
            for label, state_obj, fit, raw_closed, timed_out, confirmed_closed in [
                ("FRONT", self.front_state, front, front_detected_closed,
                 front_timed_out, front_is_closed),
                ("BACK",  self.back_state,  back,  back_detected_closed,
                 back_timed_out,  back_is_closed),
            ]:
                conf_str = "CLOSED" if confirmed_closed else "OPEN"
                raw_str = "CLOSED" if raw_closed else "OPEN"
                if timed_out:
                    raw_str += " (T/O)"
                status_color = GREEN if confirmed_closed else RED

                lines.append((f"=== {label} DRAWER ===", status_color))
                lines.append((
                    f"  Status: {conf_str}  |  Raw: {raw_str}",
                    status_color,
                ))

                # State machine buffer stats
                if state_obj.consecutive_state is not None and state_obj.consecutive_start is not None:
                    elapsed = now - state_obj.consecutive_start
                    remaining = max(STATE_CHANGE_DURATION - elapsed, 0)
                    lines.append((
                        f"  Pending -> {state_obj.consecutive_state}: "
                        f"{elapsed:.1f}s / {STATE_CHANGE_DURATION:.0f}s  "
                        f"(remaining: {remaining:.1f}s)",
                        YELLOW,
                    ))
                else:
                    lines.append((
                        f"  State stable (no pending transition)",
                        GRAY,
                    ))

                # Fitted region details & threshold comparison
                if fit is not None:
                    lines.append((
                        f"  Fit: scale={fit['scale']}  angle={fit['angle']}deg",
                        WHITE,
                    ))
                    # Bright strip: actual vs threshold
                    b_max = fit.get('max_bright', 0)
                    b_avg = fit.get('avg_bright', 0.0)
                    b_max_ok = "PASS" if b_max >= bright_thr else "FAIL"
                    b_avg_ok = "PASS" if b_avg >= bright_avg_thr else "FAIL"
                    lines.append((
                        f"  Bright: max={b_max}/{bright_thr}({b_max_ok}) "
                        f"avg={b_avg:.0f}/{bright_avg_thr}({b_avg_ok})",
                        GREEN if b_max_ok == "PASS" and b_avg_ok == "PASS" else RED,
                    ))
                    # Dark strip: actual vs threshold
                    d_max = fit.get('max_dark', 0)
                    d_avg = fit.get('avg_dark', 0.0)
                    d_max_ok = "PASS" if d_max <= dark_thr else "FAIL"
                    d_avg_ok = "PASS" if d_avg < dark_avg_thr else "FAIL"
                    lines.append((
                        f"  Dark:   max={d_max}/{dark_thr}({d_max_ok}) "
                        f"avg={d_avg:.0f}/{dark_avg_thr}({d_avg_ok})",
                        GREEN if d_max_ok == "PASS" and d_avg_ok == "PASS" else RED,
                    ))
                elif timed_out:
                    lines.append((f"  Fit: TIMED OUT (>{DRAWER_DETECT_TIMEOUT}s)", YELLOW))
                else:
                    lines.append((f"  Fit: NOT FOUND (drawer likely open)", RED))

                lines.append(("", WHITE))  # spacer

            # ─── Person / Alert ──────────────────────────────────────
            person_count = sum(self.person_in_region_history)
            total_count = len(self.person_in_region_history)
            lines.append((
                f"Person: {'YES' if person_stable else 'NO'}  "
                f"({person_count}/{total_count} in buffer)",
                GREEN if person_stable else GRAY,
            ))
            lines.append((
                f"Alert: {'ACTIVE' if alert_active else 'INACTIVE'}",
                RED if alert_active else GREEN,
            ))

            # ── Render the info panel at bottom-left ────────────────
            num_lines = len(lines)
            panel_h = num_lines * line_height + 16  # 8px padding top+bottom
            panel_w = 480
            panel_x = 0
            panel_y = h_img - panel_h

            # Clamp panel to image bounds
            if panel_y < 0:
                panel_y = 0
                panel_h = h_img

            # Semi-transparent dark overlay
            overlay = viz_img.copy()
            cv2.rectangle(
                overlay,
                (panel_x, panel_y),
                (panel_x + panel_w, panel_y + panel_h),
                (0, 0, 0),
                -1,
            )
            cv2.addWeighted(overlay, 0.7, viz_img, 0.3, 0, viz_img)

            # Draw text lines
            text_x = panel_x + 8
            text_y = panel_y + 14
            for text, color in lines:
                if text:  # skip empty spacer lines but still advance y
                    cv2.putText(
                        viz_img, text, (text_x, text_y),
                        font, font_scale, color, thickness, cv2.LINE_AA,
                    )
                text_y += line_height

            # Alert banner text at bottom of panel
            if alert_active:
                cv2.putText(
                    viz_img,
                    "!!! ALERT: DRAWER OPEN !!!",
                    (text_x, text_y + 4),
                    font, 0.7, RED, 2, cv2.LINE_AA,
                )

            # ── Encode & push ───────────────────────────────────────
            _, buffer = cv2.imencode(".jpg", viz_img)
            frame_b64 = base64.b64encode(buffer).decode("utf-8")

            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj

            vis_payload = {
                "camera_id": CAMERA_ID,
                "timestamp": timestamp,
                "frame": frame_b64,
                "frame_url": frame_url,
                "results": results,
            }
            self.vis_queue.push(json.dumps(vis_payload, default=convert_numpy))

        except Exception as e:
            logger.error(f"Visualization error: {e}", exc_info=True)

    # ── main loop ───────────────────────────────────────────────────
    def run(self):
        logger.info("DrawerDetectionService started, listening on input queue.")
        if not INPUT_QUEUE or not GROUP_NAME:
            logger.error("INPUT_QUEUE_NAME or GROUP_NAME not defined. Exiting.")
            return

        # Create consumer group if not exists
        try:
            self.redis.xgroup_create(INPUT_QUEUE, GROUP_NAME, mkstream=True)
            logger.info(f"Created consumer group {GROUP_NAME} on {INPUT_QUEUE}")
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.info(f"Consumer group {GROUP_NAME} already exists.")
            else:
                logger.error(f"Error creating consumer group: {e}")

        while True:
            try:
                # ── Drain all pending messages, keep only the newest ──
                newest_id, newest_msg = self._drain_to_newest()

                if newest_id is None:
                    # Nothing available; block-wait for the next message
                    resp = self.redis.xreadgroup(
                        groupname=GROUP_NAME,
                        consumername="worker-1",
                        streams={INPUT_QUEUE: ">"},
                        count=1,
                        block=5000,
                    )
                    if not resp:
                        continue
                    stream_name, messages = resp[0]
                    newest_id, newest_msg = messages[0]

                # Decode message
                decoded = {
                    k.decode(): v.decode() for k, v in newest_msg.items()
                }
                frame_url = decoded.get("frame_url")
                timestamp_str = decoded.get("timestamp")

                if not frame_url or not timestamp_str:
                    logger.warning(
                        "Message missing 'frame_url' or 'timestamp'. Skipping."
                    )
                else:
                    self.process_frame(decoded, frame_url, timestamp_str)

                # Ack the processed (newest) message
                self.redis.xack(INPUT_QUEUE, GROUP_NAME, newest_id)
                self.mark_acked(INPUT_QUEUE, newest_id.decode(), GROUP_NAME)

            except json.JSONDecodeError:
                logger.error("Failed to decode JSON message from input queue.")
            except Exception as e:
                logger.exception(f"Unexpected error in main loop: {e}")
                time.sleep(1)


if __name__ == "__main__":
    service = DrawerDetectionService()
    service.run()
