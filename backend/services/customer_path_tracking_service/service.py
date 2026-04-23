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
from collections import deque, defaultdict
import base64
from ultralytics import YOLO
import datetime
import uuid
import torch
import sys
from pathlib import Path

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

load_dotenv(override=False)

REDIS_HOST             = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT             = int(os.getenv("REDIS_PORT", "6379"))
LOKI_URL               = os.getenv("LOKI_URL", "http://localhost:3100/loki/api/v1/push")
INPUT_QUEUE            = os.getenv("INPUT_QUEUE_NAME")
OUTPUT_QUEUE           = os.getenv("OUTPUT_QUEUE_NAME_1")
GROUP_NAME             = os.getenv("GROUP_NAME")
CAMERA_ID              = os.getenv("CAMERA_ID", INPUT_QUEUE.split(":")[-1])
SAMPLE_FPS             = float(os.getenv("SAMPLE_FPS", "5"))
WAITING_TIME_THRESHOLD = float(os.getenv("WAITING_TIME_THRESHOLD", "300"))
LOST_TRACK_THRESHOLD   = float(os.getenv("LOST_TRACK_THRESHOLD", "10"))
HISTORY_MAXLEN         = int(os.getenv("HISTORY_MAXLEN", "15"))
YOLO_MODEL             = os.getenv("YOLO_MODEL", "person_detection_model_17_02_2026.pt")
STRONGSORT_WEIGHTS     = TRACKING_ROOT / 'osnet_x1_0_msmt17.pt'
STRONGSORT_CONFIG      = TRACKING_ROOT / 'strong_sort.yaml'
VISUALIZATION_ENABLED  = int(os.getenv("VISUALIZATION_ENABLED", "0"))
ACK_TTL_SECONDS        = 10 * 60

handler = logging_loki.LokiQueueHandler(
    Queue(-1), url=LOKI_URL,
    tags={"application": "customer_path_tracking_service"}, version="1",
)
logger = logging.getLogger("customer_path_tracking_service")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.addHandler(logging.StreamHandler())


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

class FixedSizeQueue:
    def __init__(self, redis_client, name, max_size=10):
        self.r = redis_client; self.name = name; self.max_size = max_size

    def push(self, item):
        pipe = self.r.pipeline()
        pipe.rpush(self.name, item)
        pipe.ltrim(self.name, -self.max_size, -1)
        pipe.execute()


def xyxy2xywh_center(boxes):
    xywh = torch.zeros_like(boxes)
    xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
    xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
    xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
    xywh[:, 3] = boxes[:, 3] - boxes[:, 1]
    return xywh.detach().cpu()


# ─────────────────────────────────────────────────────────────────────────────
# Duration Tracker
# ─────────────────────────────────────────────────────────────────────────────

class PersonDurationTracker:
    def __init__(self, lost_track_threshold_seconds=10, waiting_threshold_seconds=300):
        self.lost_track_threshold = lost_track_threshold_seconds
        self.waiting_threshold    = waiting_threshold_seconds
        self.tracks               = {}

    def update(self, track_ids,frame_timestamp:float, bboxes=None):
        now = frame_timestamp
        for idx, track_id in enumerate(track_ids):
            bbox = bboxes[idx] if bboxes and idx < len(bboxes) else None
            if track_id not in self.tracks:
                self.tracks[track_id] = {'first_seen': now, 'last_seen': now, 'bbox': bbox}
                logger.info(f"New person tracked: ID {track_id}")
            else:
                self.tracks[track_id]['last_seen'] = now
                if bbox is not None:
                    self.tracks[track_id]['bbox'] = bbox

        stale = [tid for tid, t in self.tracks.items()
                 if now - t['last_seen'] > self.lost_track_threshold]
        for tid in stale:
            logger.info(f"Removing stale track ID {tid}")
            del self.tracks[tid]

    def get_duration_seconds(self, track_id):
        if track_id not in self.tracks:
            return 0.0
        t = self.tracks[track_id]
        return t['last_seen'] - t['first_seen']

    def get_tracked_persons(self):
        persons = []
        for track_id, track in self.tracks.items():
            dur = self.get_duration_seconds(track_id)
            persons.append({
                'track_id':           track_id,
                'bbox':               track['bbox'],
                'duration_seconds':   dur,
                'duration_minutes':   dur / 60,
                'exceeded_threshold': dur >= self.waiting_threshold,
            })
        return persons


# ─────────────────────────────────────────────────────────────────────────────
# Main Service
# ─────────────────────────────────────────────────────────────────────────────

class CustomerPathTrackingService:
    def __init__(self):
        self.redis      = self.connect_redis(REDIS_HOST, REDIS_PORT)
        self.camera_id  = CAMERA_ID
        self.sample_fps = SAMPLE_FPS

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")

        logger.info(f"Loading YOLO model: {YOLO_MODEL}...")
        self.model = YOLO(YOLO_MODEL)
        if self.device == 'cuda':
            self.model.to('cuda')
        logger.info("YOLO model loaded.")

        logger.info("Loading StrongSORT...")
        cfg = get_config()
        cfg.merge_from_file(str(STRONGSORT_CONFIG))
        self.tracker = StrongSORT(
            STRONGSORT_WEIGHTS, torch.device(self.device), False,
            max_dist=cfg.STRONGSORT.MAX_DIST,
            max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
            max_age=cfg.STRONGSORT.MAX_AGE,
            n_init=cfg.STRONGSORT.N_INIT,
            nn_budget=cfg.STRONGSORT.NN_BUDGET,
            mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
            ema_alpha=cfg.STRONGSORT.EMA_ALPHA,
        )

        self.duration_tracker = PersonDurationTracker(
            lost_track_threshold_seconds=LOST_TRACK_THRESHOLD,
            waiting_threshold_seconds=WAITING_TIME_THRESHOLD,
        )

        self.track_history = defaultdict(lambda: deque(maxlen=60))
        self.frame_history = defaultdict(lambda: deque(maxlen=HISTORY_MAXLEN))

        # ── Per-track first-seen snapshot ─────────────────────────────────────
        # Captured the very first frame a track_id appears in.
        # Schema: track_id -> {'frame_b64': str, 'bbox': [x1,y1,x2,y2]}
        # 'frame_b64' = raw jpg (no annotations) so we can draw fresh bbox on it
        # 'bbox'      = bounding box at first-seen moment
        self.first_seen: dict = {}

        # alerted_tracks: track_ids we have already fired an alert for.
        # We never fire twice for the same track_id.
        self.alerted_tracks: set = set()
        if VISUALIZATION_ENABLED:
            self.vis_queue = FixedSizeQueue(
                self.redis,
                f"stream:visualization:customer_path:{self.camera_id}", 10,
            )
        self.frame_idx = 0

    # ── Infrastructure ────────────────────────────────────────────────────────

    def connect_redis(self, host, port, retry_delay=5):
        while True:
            try:
                r = redis.Redis(host=host, port=port, decode_responses=False)
                r.ping()
                logger.info("Connected to Redis.")
                return r
            except Exception as e:
                logger.error(f"Redis failed: {e}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)

    def mark_acked(self, stream, msg_id, consumer):
        seen_key = f"ack:seen:{stream}:{msg_id}"
        cnt_key  = f"ack:cnt:{stream}:{msg_id}"
        pipe = self.redis.pipeline()
        pipe.sadd(seen_key, consumer)
        pipe.expire(seen_key, ACK_TTL_SECONDS)
        if pipe.execute()[0]:
            pipe = self.redis.pipeline()
            pipe.incr(cnt_key)
            pipe.expire(cnt_key, ACK_TTL_SECONDS)
            pipe.execute()

    def download_image(self, url: str):
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            if len(r.content) < 1024:
                logger.warning(f"download_image: too small ({len(r.content)}B) {url}")
                return None
            img = cv2.imdecode(np.asarray(bytearray(r.content), dtype=np.uint8),
                               cv2.IMREAD_COLOR)
            if img is None:
                logger.warning(f"download_image: imdecode returned None for {url}")
                return None
            if img.std() < 5.0:
                logger.warning(f"download_image: corrupt image (std={img.std():.2f}) {url}")
                return None
            return img
        except Exception as e:
            logger.error(f"download_image failed for {url}: {e}")
            return None

    # ── Detection / Tracking ─────────────────────────────────────────────────

    def run_strongsort(self, frame, detections):
        if not detections:
            self.tracker.increment_ages()
            return []
        xyxy  = torch.tensor([d["bbox"] for d in detections], dtype=torch.float32)
        confs = torch.tensor([d["confidence"] for d in detections], dtype=torch.float32)
        clss  = torch.zeros(len(detections), dtype=torch.float32)
        return self.tracker.update(xyxy2xywh_center(xyxy), confs, clss, frame)
    
    def parse_timestamp(self, ts: str) -> float:
        return datetime.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()

    def detect_and_track(self, frame, frame_timestamp:float):
        detections = []
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
                    track_ids = [int(o[4]) for o in outputs]
                    bboxes    = [o[0:4].tolist() for o in outputs]
                    self.duration_tracker.update(track_ids, frame_timestamp, bboxes)

                    for o in outputs:
                        x1, y1, x2, y2 = map(int, o[0:4])
                        track_id = int(o[4])
                        bbox_now = [x1, y1, x2, y2]

                        self.track_history[track_id].append(((x1 + x2) // 2, y2))

                        # ── Capture first-seen snapshot ───────────────────────
                        # Store the raw frame + bbox the very first time we see
                        # this track_id. This becomes the RAISED image.
                        if track_id not in self.first_seen and track_id not in self.alerted_tracks:
                            # Guard: only store if this track_id has never been alerted.
                            # StrongSORT reuses numeric IDs — without this guard, a new
                            # person getting an old alerted ID would overwrite the snapshot.
                            _, buf = cv2.imencode(".jpg", frame)
                            self.first_seen[track_id] = {
                                'frame_b64':  base64.b64encode(buf).decode("utf-8"),
                                'bbox':       bbox_now,
                                'frame_url':  getattr(self, '_current_frame_url', ''),
                                'captured_at': frame_timestamp,
                            }
                            logger.info(
                                f"First-seen snapshot stored for Track ID {track_id} "
                                f"at url={getattr(self, '_current_frame_url', 'unknown')}"
                            )

                    tracked_persons = self.duration_tracker.get_tracked_persons()
                    for o in outputs:
                        detections.append({
                            'bbox':       o[0:4].tolist(),
                            'track_id':   int(o[4]),
                            'confidence': float(confidences[0]) if len(confidences) > 0 else 0.0,
                        })
                else:
                    self.duration_tracker.update([], frame_timestamp, None)
            else:
                self.tracker.increment_ages()
                self.duration_tracker.update([], frame_timestamp, None)
        else:
            self.tracker.increment_ages()
            self.duration_tracker.update([], frame_timestamp, None)

        active_ids = set(self.duration_tracker.tracks.keys())
        for tid in [t for t in self.track_history if t not in active_ids]:
            del self.track_history[tid]

        self.frame_idx += 1
        return detections, tracked_persons

    # ── Visualization ─────────────────────────────────────────────────────────

    def _draw_paths(self, frame):
        for track_id, trail in self.track_history.items():
            if len(trail) > 1:
                pts = np.array(trail, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=False,
                              color=(0, 0, 255), thickness=2)

    @staticmethod
    def _draw_bbox(img, bbox, label, color):
        """Draw bbox + label on img. Returns img (modified in place)."""
        if bbox is None:
            return img
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return img

    # ── Alert ─────────────────────────────────────────────────────────────────

    def _fire_alert(self, track_id, camera_id, timestamp,
                    duration_seconds, current_frame, current_bbox, frame_timestamp:float):
        """
        Fire a single alert ticket with two distinct frames:

        RAISED  → first frame the person was ever seen, with their bbox
                  at that moment drawn on it. Shows "this is when they arrived."

        RESOLVED → current frame (at 5+ min mark), person still visible,
                   with their current bbox drawn on it.
                   Shows "this is proof they were still here after 5 minutes."

        Both frames show the SAME person identified by the SAME bbox colour/label.
        """
        try:
            snapshot = self.first_seen.get(track_id)
            if snapshot is None:
                logger.warning(f"No first-seen snapshot for Track ID {track_id}, skipping alert.")
                return

            mins  = int(duration_seconds // 60)
            secs  = int(duration_seconds % 60)
            label_first   = f"ID:{track_id}  First seen"
            label_resolve = f"ID:{track_id}  {mins:02d}:{secs:02d} elapsed"
            color = (0, 0, 255)   # red — consistent in both frames

            # ── RAISED frame: first-seen moment ──────────────────────────────
            # Decode the stored raw jpg, draw the bbox from that first moment.
            first_arr = np.frombuffer(base64.b64decode(snapshot['frame_b64']), np.uint8)
            first_img = cv2.imdecode(first_arr, cv2.IMREAD_COLOR)
            if first_img is None or first_img.std() < 5.0:
                logger.warning(f"First-seen frame corrupt for Track ID {track_id}")
                return

            self._draw_bbox(first_img, snapshot['bbox'], label_first, color)
            _, buf = cv2.imencode(".jpg", first_img)
            raised_frame_b64 = base64.b64encode(buf).decode("utf-8")

            # ── RESOLVED frame: current moment (5+ min later) ────────────────
            # Use current_frame (already validated in process_message).
            # Draw the current bbox so viewer can identify the same person.
            resolved_img = current_frame.copy()
            self._draw_bbox(resolved_img, current_bbox, label_resolve, color)
            # Also draw the full path trail to show how long they've been there
            trail = self.track_history.get(track_id)
            if trail and len(trail) > 1:
                pts = np.array(trail, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(resolved_img, [pts], isClosed=False,
                              color=color, thickness=2)
            _, buf = cv2.imencode(".jpg", resolved_img)
            resolved_frame_b64 = base64.b64encode(buf).decode("utf-8")

            ticket_id = str(uuid.uuid4())
            first_seen_iso = datetime.datetime.fromtimestamp(
                snapshot['captured_at'], tz=datetime.timezone.utc
            ).isoformat()

            # ── Publish RAISED ────────────────────────────────────────────────
            self.redis.xadd(OUTPUT_QUEUE, {"data": json.dumps({
                "camera_id":                   camera_id,
                "timestamp":                   first_seen_iso,
                "frame":                       raised_frame_b64,
                "frame_without_visualization": snapshot['frame_b64'],  # truly raw
                "description": (
                    f"Person waiting for more than 5 min."
                ),
                "target_collection": "alert_collection",
                "type":              "raised",
                "ticket_id":         ticket_id,
            })})

            # ── Publish RESOLVED ──────────────────────────────────────────────
            self.redis.xadd(OUTPUT_QUEUE, {"data": json.dumps({
                "camera_id":   camera_id,
                "timestamp":   timestamp,
                "frame":       resolved_frame_b64,
                "description": (
                    f"Person (ID:{track_id}) confirmed still present after "
                    f"{mins} min {secs:02d} sec. Alert closed."
                ),
                "target_collection": "alert_collection",
                "type":              "resolved",
                "ticket_id":         ticket_id,
            })})

            self.alerted_tracks.add(track_id)
            gap = frame_timestamp - snapshot.get('captured_at', frame_timestamp)
            logger.info(
                f"Alert fired for Track ID {track_id} "
                f"({mins}m{secs:02d}s elapsed). "
                f"First-seen was {gap:.1f}s ago. Ticket: {ticket_id}"
            )

        except Exception as e:
            logger.error(f"Failed to fire alert for Track ID {track_id}: {e}")

    # ── Message processing ────────────────────────────────────────────────────

    def process_message(self, message: dict):
        message   = {k.decode(): v.decode() for k, v in message.items()}
        frame_url = message.get("frame_url")
        camera_id = message.get("camera_id", self.camera_id)
        timestamp = message.get(
            "timestamp", datetime.datetime.now().astimezone().isoformat()
        )
        start_time = time.time()
        if not frame_url:
            logger.warning("Message missing 'frame_url'. Skipping.")
            return

        timestamp_str = message.get("timestamp", "")
        if not timestamp_str:
            logger.warning("Message missing 'timestamp', falling back to current time.")
            frame_timestamp = time.time()
            timestamp = datetime.datetime.now().astimezone().isoformat()
        else:
            try:
                frame_timestamp = self.parse_timestamp(timestamp_str)
            except Exception as e:
                logger.error(f"Failed to parse timestamp '{timestamp_str}': {e}. Falling back to current time.")
                frame_timestamp = time.time()
        
        frame = self.download_image(frame_url)
        if frame is None:
            logger.error(f"Could not download frame from {frame_url}")
            return

        # Store for access inside detect_and_track's first_seen capture
        self._current_frame_url = frame_url

        detections, tracked_persons = self.detect_and_track(frame, frame_timestamp)

        num_persons    = len(detections)
        exceeded_count = sum(1 for p in tracked_persons if p['exceeded_threshold'])

        # Update per-track frame history (for majority vote)
        for person in tracked_persons:
            self.frame_history[person['track_id']].append({
                "track_id":           person['track_id'],
                "frame_url":          frame_url,
                "timestamp":          timestamp,
                "duration_seconds":   person['duration_seconds'],
                "exceeded_threshold": person['exceeded_threshold'],
            })

        # ── Alert logic ────────────────────────────────────────────────────────
        # Fire ONE alert per track_id, only when:
        #   1. Person has been tracked >= WAITING_TIME_THRESHOLD seconds
        #   2. Majority of recent history frames confirm this (anti-glitch)
        #   3. Person is STILL VISIBLE in the current frame
        #   4. We have not already alerted for this track_id
        #
        # No alert if person disappeared before 5 min — we simply never fire.

        for person in tracked_persons:
            track_id = person['track_id']

            if track_id in self.alerted_tracks:
                continue

            history = self.frame_history[track_id]
            exceeded_frames = sum(1 for h in history if h.get("exceeded_threshold"))
            if exceeded_frames > (len(history) / 2):
                self._fire_alert(
                    track_id, camera_id, timestamp,
                    person['duration_seconds'],
                    frame,               # current frame → RESOLVED image
                    person['bbox'],      # current bbox  → drawn on RESOLVED image
                    frame_timestamp      # current frame timestamp → used to calculate gap
                )

        # Clean up memory for inactive tracks.
        # IMPORTANT: alerted_tracks is NEVER cleared — once a track_id has fired
        # an alert it stays in alerted_tracks for the entire session lifetime.
        # StrongSORT can reuse track IDs after a track expires, so clearing
        # alerted_tracks would allow the same numeric ID to fire a second alert
        # for a completely different person. Keeping it permanent prevents this.
        active_ids = {p['track_id'] for p in tracked_persons}
        inactive_ids = set(self.first_seen.keys()) - active_ids
        for tid in inactive_ids:
            self.first_seen.pop(tid, None)      # free memory — snapshot no longer needed
            self.frame_history.pop(tid, None)   # free memory — history no longer needed
            # NOTE: do NOT remove from self.alerted_tracks
        end_time = time.time()
        # ── Visualization ──────────────────────────────────────────────────────
        if VISUALIZATION_ENABLED:
            annotated = frame.copy()
            self._draw_paths(annotated)
            _, buf = cv2.imencode(".jpg", annotated)

            self.vis_queue.push(json.dumps({
                "camera_id":        camera_id,
                "timestamp":        timestamp,
                "frame":            base64.b64encode(buf).decode("utf-8"),
                "service":          "customer_path_tracking_service",
                "num_persons":      num_persons,
                "exceeded_count":   exceeded_count,
                "tracking_details": tracked_persons,
            }))

        logger.info(
            f"Processed frame: {num_persons} detected, "
            f"{len(tracked_persons)} tracked, {exceeded_count} exceeded threshold"
            f"processed frame {timestamp} for camera {camera_id}"
            f"processing time {end_time - start_time:.2f} seconds"
        )

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        logger.info(
            f"CustomerPathTrackingService started | Camera: {self.camera_id} | "
            f"Alert threshold: {WAITING_TIME_THRESHOLD}s | "
            f"History: {HISTORY_MAXLEN} frames"
        )
        while True:
            try:
                resp = self.redis.xreadgroup(
                    groupname=GROUP_NAME, consumername="worker-1",
                    streams={INPUT_QUEUE: ">"}, count=1, block=5000,
                )
                if resp:
                    _, messages = resp[0]
                    msg_id, message = messages[0]
                    self.process_message(message)
                    self.redis.xack(INPUT_QUEUE, GROUP_NAME, msg_id)
                    self.mark_acked(INPUT_QUEUE, msg_id.decode(), GROUP_NAME)

            except json.JSONDecodeError:
                logger.error("Failed to decode JSON from input queue.")
            except Exception as e:
                logger.exception(f"Unexpected error: {e}")
                time.sleep(1)


if __name__ == "__main__":
    service = CustomerPathTrackingService()
    service.run()