import json
from dotenv import load_dotenv
import redis
import logging
import time
import requests
import logging_loki
from multiprocessing import Queue
import os
from collections import deque, defaultdict
import cv2
import numpy as np
from datetime import datetime
import torch
from pathlib import Path
import sys
import base64
import uuid
import statistics
from ultralytics import YOLO

# Import StrongSORT tracking components
ROOT = Path(__file__).resolve().parents[0]
TRACKING_ROOT = ROOT / 'tracking'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(TRACKING_ROOT) not in sys.path:
    sys.path.append(str(TRACKING_ROOT))
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))

from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT

load_dotenv()

REDIS_HOST               = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT               = int(os.getenv("REDIS_PORT", "6379"))
LOKI_URL                 = os.getenv("LOKI_URL", "http://localhost:3100/loki/api/v1/push")
INPUT_QUEUE              = os.getenv("INPUT_QUEUE_NAME")
OUTPUT_QUEUE_1           = os.getenv("OUTPUT_QUEUE_NAME_1")
GROUP_NAME               = os.getenv("GROUP_NAME")
CAMERA_ID                = os.getenv("CAMERA_ID", INPUT_QUEUE.split(":")[-1])
VISUALIZE                = str(os.getenv("VISUALIZE", "False")).lower() == "true"
DURATION_THRESHOLD_MINUTES = int(os.getenv("DURATION_THRESHOLD_MINUTES", "30"))

# History length for majority-vote alerting (mirrors counter_staff maxlen=15)
HISTORY_MAXLEN = int(os.getenv("HISTORY_MAXLEN", "15"))

# Model paths
YOLO_MODEL         = os.getenv("YOLO_MODEL", "person_detection_model_17_02_2026.pt")
STRONGSORT_WEIGHTS = TRACKING_ROOT / 'osnet_x1_0_msmt17.pt'
STRONGSORT_CONFIG  = TRACKING_ROOT / 'strong_sort.yaml'

ACK_TTL_SECONDS = 10 * 60

# Configure Logging
handler = logging_loki.LokiQueueHandler(
    Queue(-1),
    url=LOKI_URL,
    tags={"application": "crew_room_service"},
    version="1",
)
logger = logging.getLogger("crew_room_service")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

class FixedSizeQueue:
    def __init__(self, redis_client, name, max_size):
        self.r        = redis_client
        self.name     = name
        self.max_size = max_size

    def push(self, item):
        pipe = self.r.pipeline()
        pipe.rpush(self.name, item)
        pipe.ltrim(self.name, -self.max_size, -1)
        pipe.execute()

    def pop(self, timeout=0):
        result = self.r.blpop(self.name, timeout=timeout)
        if result:
            _, value = result
            return value
        return None

    def size(self):
        return self.r.llen(self.name)


# ─────────────────────────────────────────────────────────────────────────────
# Duration Tracker  (frame-counter based, original crew_room logic preserved)
# ─────────────────────────────────────────────────────────────────────────────

class PersonDurationTracker:
    """Duration tracking integrated with StrongSORT"""

    def __init__(self, fps=1, threshold_minutes=30):
        self.fps               = fps
        self.threshold_minutes = threshold_minutes
        self.threshold_seconds = threshold_minutes * 60
        self.tracks            = {}
        self.current_frame     = 0
        self.alerted_persons   = set()

    def update(self, track_ids, frame_number, bboxes=None):
        self.current_frame = frame_number

        for idx, track_id in enumerate(track_ids):
            if track_id not in self.tracks:
                self.tracks[track_id] = {
                    'first_frame':          frame_number,
                    'last_frame':           frame_number,
                    'total_frames_visible': 1,
                    'bbox': bboxes[idx] if bboxes and idx < len(bboxes) else None,
                }
                logger.info(f"New person tracked: ID {track_id}")
            else:
                self.tracks[track_id]['last_frame']           = frame_number
                self.tracks[track_id]['total_frames_visible'] += 1
                if bboxes and idx < len(bboxes):
                    self.tracks[track_id]['bbox'] = bboxes[idx]

        # Remove tracks not seen for > 2 seconds
        max_gap_frames = int(self.fps * 2)
        stale_ids = [
            tid for tid, t in self.tracks.items()
            if frame_number - t['last_frame'] > max_gap_frames
        ]
        for tid in stale_ids:
            logger.info(f"Removing stale track ID {tid}")
            del self.tracks[tid]

    def get_duration_seconds(self, track_id):
        if track_id not in self.tracks:
            return 0
        duration_frames = (
            self.tracks[track_id]['last_frame']
            - self.tracks[track_id]['first_frame']
        )
        return duration_frames / self.fps

    def get_tracked_persons(self):
        tracked_persons = []
        for track_id, track in self.tracks.items():
            duration_seconds = self.get_duration_seconds(track_id)
            duration_minutes = duration_seconds / 60
            exceeded         = duration_minutes >= self.threshold_minutes

            if exceeded and track_id not in self.alerted_persons:
                self.alerted_persons.add(track_id)
                logger.warning(
                    f"⚠️ ALERT: Person ID {track_id} exceeded {self.threshold_minutes} minutes "
                    f"(Duration: {duration_minutes:.1f} min)"
                )

            tracked_persons.append({
                'track_id':           track_id,
                'bbox':               track['bbox'],
                'duration_seconds':   duration_seconds,
                'duration_minutes':   duration_minutes,
                'exceeded_threshold': exceeded,
                'total_frames':       track['total_frames_visible'],
            })
        return tracked_persons


def xyxy2xywh_center(boxes):
    xywh = torch.zeros_like(boxes)
    xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
    xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
    xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
    xywh[:, 3] = boxes[:, 3] - boxes[:, 1]
    return xywh.detach().cpu()


# ─────────────────────────────────────────────────────────────────────────────
# Main Detector
# ─────────────────────────────────────────────────────────────────────────────

class CrewRoomDetectorWithStrongSORT:
    def __init__(self):
        self.redis = self.connect_redis(host=REDIS_HOST, port=REDIS_PORT)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")

        logger.info(f"Loading YOLOv11 model: {YOLO_MODEL}...")
        self.model = YOLO(YOLO_MODEL)
        if self.device == 'cuda':
            self.model.to('cuda')
        logger.info("YOLOv11 model loaded successfully")

        logger.info("Loading StrongSORT...")
        cfg = get_config()
        cfg.merge_from_file(str(STRONGSORT_CONFIG))
        self.tracker = StrongSORT(
            STRONGSORT_WEIGHTS,
            torch.device(self.device),
            False,
            max_dist=cfg.STRONGSORT.MAX_DIST,
            max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
            max_age=cfg.STRONGSORT.MAX_AGE,
            n_init=cfg.STRONGSORT.N_INIT,
            nn_budget=cfg.STRONGSORT.NN_BUDGET,
            mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
            ema_alpha=cfg.STRONGSORT.EMA_ALPHA,
        )

        self.duration_tracker = PersonDurationTracker(
            fps=1,
            threshold_minutes=DURATION_THRESHOLD_MINUTES,
        )

        self.frame_idx = 0
        self.history   = deque(maxlen=10)   # rolling avg of person counts

        # ── Per-track frame history for majority-vote alerting ────────────────
        # Each entry: {track_id, frame_url, timestamp,
        #              duration_minutes, exceeded_threshold}
        self.frame_history = defaultdict(lambda: deque(maxlen=HISTORY_MAXLEN))

        # ── Alert state ───────────────────────────────────────────────────────
        self.last_exceeded_state = defaultdict(lambda: False)  # track_id -> bool
        self.current_alert_id    = defaultdict(lambda: None)   # track_id -> uuid str

        if VISUALIZE:
            self.vis_queue = FixedSizeQueue(
                self.redis, f"stream:visualization:{CAMERA_ID}", 10
            )

        # Restore persisted alerts from Redis (survive restarts)
        self.load_alert_state()

        logger.info("Crew Room Detector with StrongSORT initialized successfully")

    # ── Infrastructure ────────────────────────────────────────────────────────

    def connect_redis(self, host, port, retry_delay=5):
        while True:
            try:
                r = redis.Redis(host=host, port=port, decode_responses=False)
                r.ping()
                logger.info("Connected to Redis successfully.")
                return r
            except Exception as e:
                logger.error(f"Redis connection failed: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

    def load_alert_state(self):
        """
        Restore active alert states from Redis on startup so alerts survive
        service restarts without being re-raised.
        """
        try:
            pattern = f"alert_state:crew_room:{CAMERA_ID}:*"
            keys    = self.redis.keys(pattern)
            for key in keys:
                key_str  = key.decode()
                track_id = int(key_str.split(":")[-1])
                alert_id = self.redis.get(key)
                if alert_id:
                    self.last_exceeded_state[track_id] = True
                    self.current_alert_id[track_id]    = alert_id.decode()
                    logger.info(
                        f"Restored Crew Room Alert for Track ID {track_id}: "
                        f"{alert_id.decode()}"
                    )
        except Exception as e:
            logger.error(f"Failed to load alert state from Redis: {e}")

    def mark_acked(self, stream: str, msg_id: str, consumer: str):
        cnt_key  = f"ack:cnt:{stream}:{msg_id}"
        seen_key = f"ack:seen:{stream}:{msg_id}"
        pipe = self.redis.pipeline()
        pipe.sadd(seen_key, consumer)
        pipe.expire(seen_key, ACK_TTL_SECONDS)
        results   = pipe.execute()
        was_added = results[0]
        if was_added:
            pipe = self.redis.pipeline()
            pipe.incr(cnt_key)
            pipe.expire(cnt_key, ACK_TTL_SECONDS)
            pipe.execute()

    def download_image(self, url):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error(f"Failed to download image from {url}: {e}")
            return None

    # ── Detection / Tracking ─────────────────────────────────────────────────

    def run_strongsort(self, frame, detections):
        if not detections:
            self.tracker.increment_ages()
            return []
        xyxy  = torch.tensor([d["bbox"] for d in detections], dtype=torch.float32)
        confs = torch.tensor([d["confidence"] for d in detections], dtype=torch.float32)
        clss  = torch.zeros(len(detections), dtype=torch.float32)
        xywh  = xyxy2xywh_center(xyxy)
        return self.tracker.update(xywh, confs, clss, frame)

    def detect_and_track(self, frame):
        detections      = []
        tracked_persons = []

        with torch.no_grad():
            results = self.model(frame, conf=0.4, iou=0.45, verbose=False)

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            if len(boxes) > 0:
                xyxy_boxes  = boxes.xyxy
                confidences = boxes.conf

                det_list = [
                    {"bbox": xyxy_boxes[i].detach().cpu().tolist(),
                     "confidence": float(confidences[i])}
                    for i in range(len(xyxy_boxes))
                ]
                outputs = self.run_strongsort(frame, det_list)

                if len(outputs) > 0:
                    track_ids = [int(output[4]) for output in outputs]
                    bboxes    = [output[0:4].tolist() for output in outputs]
                    self.duration_tracker.update(track_ids, self.frame_idx, bboxes)
                    tracked_persons = self.duration_tracker.get_tracked_persons()

                    for output in outputs:
                        detections.append({
                            'bbox':       output[0:4].tolist(),
                            'track_id':   int(output[4]),
                            'confidence': float(confidences[0]) if len(confidences) > 0 else 0.0,
                        })
                else:
                    self.duration_tracker.update([], self.frame_idx)

        yolo_found_nothing = not (
            len(results) > 0
            and results[0].boxes is not None
            and len(results[0].boxes) > 0
        )
        if yolo_found_nothing:
            self.tracker.increment_ages()
            self.duration_tracker.update([], self.frame_idx)

        self.frame_idx += 1
        return detections, tracked_persons

    # ── Alert helpers ─────────────────────────────────────────────────────────

    def _best_frame_for_raise(self, track_id):
        """Most recent history frame where the person exceeded the threshold."""
        for entry in reversed(self.frame_history[track_id]):
            if entry.get("exceeded_threshold") and entry.get("frame_url"):
                return entry["frame_url"]
        if self.frame_history[track_id]:
            return self.frame_history[track_id][-1].get("frame_url")
        return None

    def _best_frame_for_resolve(self, track_id):
        """Most recent history frame where the person was no longer exceeding."""
        for entry in reversed(self.frame_history[track_id]):
            if not entry.get("exceeded_threshold") and entry.get("frame_url"):
                return entry["frame_url"]
        if self.frame_history[track_id]:
            return self.frame_history[track_id][-1].get("frame_url")
        return None

    def _annotate_frame(self, img, track_id, bbox, color=(0, 0, 255),
                        label_prefix="CREW"):
        """Draw bbox + label on img in-place."""
        if bbox:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            duration_s = self.duration_tracker.get_duration_seconds(track_id)
            mins       = int(duration_s // 60)
            secs       = int(duration_s % 60)
            label      = f"ID:{track_id} {label_prefix} {mins:02d}:{secs:02d}"
            cv2.putText(img, label, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return img

    def _raise_alert(self, track_id, camera_id, timestamp, frame_url,
                     duration_minutes, bbox):
        """
        Raise a crew-room duration alert for track_id.
        Downloads the best representative proof frame, encodes raw + annotated,
        publishes to OUTPUT_QUEUE_1, persists state to Redis.
        """
        try:
            target_url = self._best_frame_for_raise(track_id) or frame_url
            img        = self.download_image(target_url)
            if img is None:
                raise ValueError(f"Could not download frame: {target_url}")

            # Raw frame (no overlay)
            _, raw_buf    = cv2.imencode(".jpg", img)
            raw_frame_b64 = base64.b64encode(raw_buf).decode("utf-8")

            # Annotated frame
            self._annotate_frame(img, track_id, bbox, color=(0, 0, 255),
                                  label_prefix="EXCEEDED")
            _, buf        = cv2.imencode(".jpg", img)
            frame_b64_ann = base64.b64encode(buf).decode("utf-8")

            alert_id = str(uuid.uuid4())
            self.current_alert_id[track_id]    = alert_id
            self.last_exceeded_state[track_id] = True

            alert_payload = {
                "camera_id":                   camera_id,
                "timestamp":                   timestamp,
                "frame":                       frame_b64_ann,
                "frame_without_visualization": raw_frame_b64,
                "description": (
                    f"Crew member (ID: {track_id}) has been in the crew room "
                    f"for more than {DURATION_THRESHOLD_MINUTES} minutes "
                    f"(Duration: {duration_minutes:.1f} min)."
                ),
                "target_collection": "alert_collection",
                "type":              "raised",
                "ticket_id":         alert_id,
            }
            self.redis.xadd(OUTPUT_QUEUE_1, {"data": json.dumps(alert_payload)})
            # Persist so restarts don't re-raise
            self.redis.set(
                f"alert_state:crew_room:{CAMERA_ID}:{track_id}", alert_id
            )
            logger.warning(
                f"🚨 Raised Crew Room Alert for Track ID {track_id} "
                f"({duration_minutes:.1f} min)"
            )

        except Exception as e:
            logger.error(f"Failed to raise alert for Track ID {track_id}: {e}")

    def _resolve_alert(self, track_id, camera_id, timestamp, frame_url):
        """
        Resolve a crew-room duration alert for track_id.
        Picks best proof frame, publishes resolve payload, clears Redis state.
        """
        try:
            target_url = self._best_frame_for_resolve(track_id) or frame_url
            img        = self.download_image(target_url)
            if img is None:
                raise ValueError(f"Could not download frame: {target_url}")

            bbox = self.duration_tracker.tracks.get(track_id, {}).get('bbox')
            self._annotate_frame(img, track_id, bbox, color=(0, 255, 0),
                                  label_prefix="LEFT")
            _, buf    = cv2.imencode(".jpg", img)
            frame_b64 = base64.b64encode(buf).decode("utf-8")

            alert_id = self.current_alert_id[track_id]
            resolve_payload = {
                "camera_id":         camera_id,
                "timestamp":         timestamp,
                "frame":             frame_b64,
                "description": (
                    f"Crew member (ID: {track_id}) crew room duration alert resolved "
                    f"(person left or duration reset)."
                ),
                "target_collection": "alert_collection",
                "type":              "resolved",
                "ticket_id":         alert_id,
            }
            self.redis.xadd(OUTPUT_QUEUE_1, {"data": json.dumps(resolve_payload)})
            self.redis.delete(f"alert_state:crew_room:{CAMERA_ID}:{track_id}")
            logger.info(f"✅ Resolved Crew Room Alert for Track ID {track_id}")

        except Exception as e:
            logger.error(f"Failed to resolve alert for Track ID {track_id}: {e}")
        finally:
            self.last_exceeded_state[track_id] = False
            self.current_alert_id[track_id]    = None

    # ── Message processing ────────────────────────────────────────────────────

    def process_message(self, message: dict):
        start_time = time.time()
        message   = {k.decode(): v.decode() for k, v in message.items()}
        frame_url = message["frame_url"]
        camera_id = message["camera_id"]
        timestamp = message["timestamp"]

        if not frame_url:
            logger.warning("Message missing 'frame_url'. Skipping.")
            return

        frame = self.download_image(frame_url)
        if frame is None:
            logger.error(f"Could not download frame from {frame_url}")
            return

        detections, tracked_persons = self.detect_and_track(frame)

        num_persons    = len(detections)
        self.history.append(num_persons)
        avg_persons    = int(sum(self.history) / len(self.history)) if self.history else 0
        exceeded_count = sum(1 for p in tracked_persons if p['exceeded_threshold'])

        # ── Update per-track frame history ────────────────────────────────────
        active_track_ids = {p['track_id'] for p in tracked_persons}
        for person in tracked_persons:
            self.frame_history[person['track_id']].append({
                "track_id":           person['track_id'],
                "frame_url":          frame_url,
                "timestamp":          timestamp,
                "duration_minutes":   person['duration_minutes'],
                "exceeded_threshold": person['exceeded_threshold'],
            })

        # ── Majority-vote alert logic ─────────────────────────────────────────
        # Alert fires when person exceeded threshold in >50% of recent frames.
        # This prevents spurious alerts from transient false positives.
        for person in tracked_persons:
            track_id = person['track_id']
            history  = self.frame_history[track_id]

            exceeded_frames      = sum(1 for h in history if h.get("exceeded_threshold"))
            is_exceeded_majority = exceeded_frames > (len(history) / 2)

            if is_exceeded_majority and not self.last_exceeded_state[track_id]:
                # RAISE
                self._raise_alert(
                    track_id, camera_id, timestamp,
                    frame_url, person['duration_minutes'],
                    person['bbox']
                )

            elif not is_exceeded_majority and self.last_exceeded_state[track_id]:
                # RESOLVE — majority flipped back (e.g. track ID reassignment)
                self._resolve_alert(track_id, camera_id, timestamp, frame_url)

        # ── Resolve alerts for tracks that have completely disappeared ─────────
        alerted_track_ids    = {tid for tid, v in self.last_exceeded_state.items() if v}
        disappeared_alerted  = alerted_track_ids - active_track_ids

        for track_id in disappeared_alerted:
            logger.info(
                f"Track ID {track_id} disappeared with active alert — resolving."
            )
            self._resolve_alert(track_id, camera_id, timestamp, frame_url)
            if track_id in self.frame_history:
                del self.frame_history[track_id]

        # ── Regular output payload ────────────────────────────────────────────
        output_payload = {
            "camera_id":               camera_id,
            "timestamp":               timestamp,
            "num_persons":             num_persons,
            "avg_persons":             avg_persons,
            "tracked_persons":         len(tracked_persons),
            "exceeded_threshold_count": exceeded_count,
            "target_collection":       "crew_room_collection",
            "tracking_details":        tracked_persons,
        }

        try:
            self.redis.xadd(OUTPUT_QUEUE_1, {"data": json.dumps(output_payload)})
            end_time = time.time()
            
            if VISUALIZE:
                _, buffer = cv2.imencode(".jpg", frame)
                frame_b64 = base64.b64encode(buffer).decode("utf-8")
                vis_payload = {
                    "camera_id":      camera_id,
                    "timestamp":      timestamp,
                    "frame":          frame_b64,
                    "detections":     detections,
                    "tracked_persons": tracked_persons,
                    "num_persons":    num_persons,
                    "exceeded_count": exceeded_count,
                }
                self.vis_queue.push(json.dumps(vis_payload))

            logger.info(
                f"Processed frame: {num_persons} persons, "
                f"{len(tracked_persons)} tracked, "
                f"{exceeded_count} exceeded threshold"
                f"processed frame at timestamp: {timestamp}"
                f"processing time: {end_time - start_time:.2f} seconds"

            )
        except Exception as e:
            logger.error(f"Failed to push result to output queue: {e}")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        logger.info(
            f"CrewRoomDetector started "
            f"(threshold: {DURATION_THRESHOLD_MINUTES}min, "
            f"history: {HISTORY_MAXLEN} frames)"
        )

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
                    msg_id, message       = messages[0]
                    self.process_message(message)
                    self.redis.xack(INPUT_QUEUE, GROUP_NAME, msg_id)
                    self.mark_acked(INPUT_QUEUE, msg_id.decode(), GROUP_NAME)

            except json.JSONDecodeError:
                logger.error("Failed to decode JSON message from input queue.")
            except Exception as e:
                logger.exception(f"Unexpected error in main loop: {e}")
                time.sleep(1)


if __name__ == "__main__":
    service = CrewRoomDetectorWithStrongSORT()
    service.run()