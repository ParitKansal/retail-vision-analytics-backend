import json
from dotenv import load_dotenv
import redis
import logging
import time
import requests
import logging_loki
from multiprocessing import Queue
import os
from collections import deque
import cv2
import numpy as np
from datetime import datetime
import torch
from pathlib import Path
import sys
import base64

# ── StrongSORT ───────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[0]
TRACKING_ROOT = ROOT / 'tracking'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(TRACKING_ROOT) not in sys.path:
    sys.path.append(str(TRACKING_ROOT))

from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT

load_dotenv()

# ── Environment ──────────────────────────────────────────────
REDIS_HOST                 = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT                 = int(os.getenv("REDIS_PORT", "6379"))
LOKI_URL                   = os.getenv("LOKI_URL", "http://localhost:3100/loki/api/v1/push")
INPUT_QUEUE                = os.getenv("INPUT_QUEUE_NAME")
OUTPUT_QUEUE_1             = os.getenv("OUTPUT_QUEUE_NAME_1")
YOLO_MODEL_URL             = os.getenv("YOLO_MODEL_URL")        # existing YOLO service
GROUP_NAME                 = os.getenv("GROUP_NAME")
CAMERA_ID                  = os.getenv("CAMERA_ID", INPUT_QUEUE.split(":")[-1])
VISUALIZE                  = str(os.getenv("VISUALIZE", "False")).lower() == "true"
DURATION_THRESHOLD_MINUTES = int(os.getenv("DURATION_THRESHOLD_MINUTES", "30"))
DETECTION_CONFIDENCE       = float(os.getenv("DETECTION_CONFIDENCE", "0.6"))  # raise to reduce false positives

STRONGSORT_WEIGHTS = TRACKING_ROOT / 'osnet_x1_0_msmt17.pt'
STRONGSORT_CONFIG  = TRACKING_ROOT / 'strong_sort.yaml'

ACK_TTL_SECONDS = 10 * 60

# ── Logging ──────────────────────────────────────────────────
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


# ── Helpers ──────────────────────────────────────────────────
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


# ── Duration Tracker ─────────────────────────────────────────
class PersonDurationTracker:
    def __init__(self, fps=10, threshold_minutes=30):
        self.fps               = fps
        self.threshold_minutes = threshold_minutes
        self.threshold_seconds = threshold_minutes * 60
        self.tracks            = {}       # track_id -> track info
        self.current_frame     = 0
        self.alerted_persons   = set()

    def update(self, track_ids, frame_number, bboxes=None):
        self.current_frame = frame_number

        # Update / create tracks
        for idx, track_id in enumerate(track_ids):
            if track_id not in self.tracks:
                self.tracks[track_id] = {
                    'first_frame':          frame_number,
                    'last_frame':           frame_number,
                    'total_frames_visible': 1,
                    'bbox':                 bboxes[idx] if bboxes and idx < len(bboxes) else None,
                }
                logger.info(f"New person tracked: ID {track_id}")
            else:
                self.tracks[track_id]['last_frame']           = frame_number
                self.tracks[track_id]['total_frames_visible'] += 1
                if bboxes and idx < len(bboxes):
                    self.tracks[track_id]['bbox'] = bboxes[idx]

        # Remove stale tracks (not seen for > 2 seconds)
        max_gap = int(self.fps * 2)
        stale   = [tid for tid, t in self.tracks.items()
                   if frame_number - t['last_frame'] > max_gap]
        for tid in stale:
            logger.info(f"Removing stale track ID {tid}")
            del self.tracks[tid]

    def get_duration_seconds(self, track_id):
        if track_id not in self.tracks:
            return 0
        frames = self.tracks[track_id]['last_frame'] - self.tracks[track_id]['first_frame']
        return frames / self.fps

    def get_tracked_persons(self):
        result = []
        for track_id, track in self.tracks.items():
            duration_seconds = self.get_duration_seconds(track_id)
            duration_minutes = duration_seconds / 60
            exceeded         = duration_minutes >= self.threshold_minutes

            if exceeded and track_id not in self.alerted_persons:
                self.alerted_persons.add(track_id)
                logger.warning(
                    f"⚠️  ALERT: Person ID {track_id} exceeded "
                    f"{self.threshold_minutes} min ({duration_minutes:.1f} min)"
                )

            result.append({
                'track_id':          track_id,
                'bbox':              track['bbox'],
                'duration_seconds':  duration_seconds,
                'duration_minutes':  duration_minutes,
                'exceeded_threshold': exceeded,
                'total_frames':      track['total_frames_visible'],
            })
        return result


# ── xywh helper ─────────────────────────────────────────────
def xyxy2xywh_center(boxes: torch.Tensor) -> torch.Tensor:
    """[x1,y1,x2,y2] → [cx,cy,w,h]  (detached, on CPU)"""
    xywh      = torch.zeros_like(boxes)
    xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
    xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
    xywh[:, 2] =  boxes[:, 2] - boxes[:, 0]
    xywh[:, 3] =  boxes[:, 3] - boxes[:, 1]
    return xywh.detach().cpu()


# ── Main service ─────────────────────────────────────────────
class CrewRoomDetector:
    def __init__(self):
        self.redis         = self._connect_redis()
        self.yolo_endpoint = YOLO_MODEL_URL
        self.history       = deque(maxlen=10)

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # StrongSORT
        logger.info("Loading StrongSORT...")
        cfg = get_config()
        cfg.merge_from_file(str(STRONGSORT_CONFIG))

        self.tracker = StrongSORT(
            STRONGSORT_WEIGHTS,
            self.device,
            False,
            max_dist        = cfg.STRONGSORT.MAX_DIST,
            max_iou_distance= cfg.STRONGSORT.MAX_IOU_DISTANCE,
            max_age         = cfg.STRONGSORT.MAX_AGE,
            n_init          = cfg.STRONGSORT.N_INIT,
            nn_budget       = cfg.STRONGSORT.NN_BUDGET,
            mc_lambda       = cfg.STRONGSORT.MC_LAMBDA,
            ema_alpha       = cfg.STRONGSORT.EMA_ALPHA,
        )

        # Duration tracker
        self.duration_tracker = PersonDurationTracker(
            fps               = 10,
            threshold_minutes = DURATION_THRESHOLD_MINUTES,
        )

        self.frame_idx = 0

        if VISUALIZE:
            self.vis_queue = FixedSizeQueue(
                self.redis, f"stream:visualization:{CAMERA_ID}", 10
            )

        logger.info("CrewRoomDetector initialised successfully")

    # ── Redis ────────────────────────────────────────────────
    def _connect_redis(self, retry_delay=5):
        while True:
            try:
                r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
                r.ping()
                logger.info("Connected to Redis.")
                return r
            except Exception as e:
                logger.error(f"Redis connection failed: {e}. Retrying...")
                time.sleep(retry_delay)

    def mark_acked(self, stream, msg_id, consumer):
        cnt_key  = f"ack:cnt:{stream}:{msg_id}"
        seen_key = f"ack:seen:{stream}:{msg_id}"
        pipe     = self.redis.pipeline()
        pipe.sadd(seen_key, consumer)
        pipe.expire(seen_key, ACK_TTL_SECONDS)
        results  = pipe.execute()
        if results[0]:
            pipe = self.redis.pipeline()
            pipe.incr(cnt_key)
            pipe.expire(cnt_key, ACK_TTL_SECONDS)
            pipe.execute()

    # ── YOLO via HTTP ────────────────────────────────────────
    def call_yolo_service(self, frame_url):
        """
        Call the existing YOLO service with frame_url.
        Returns list of person dicts: [{"bbox":[x1,y1,x2,y2], "confidence":float}, ...]
        """
        try:
            payload  = {
                "image_url":  frame_url,
                "confidence": DETECTION_CONFIDENCE,   # higher = fewer false positives
            }
            resp     = requests.post(self.yolo_endpoint, json=payload, timeout=20)
            resp.raise_for_status()
            yolo_result = resp.json()
        except Exception as e:
            logger.error(f"YOLO service call failed: {e}")
            return []

        person_detections = []
        try:
            predict_list = yolo_result.get("predict", [])
            if not predict_list:
                return []

            parsed = json.loads(predict_list[0])
            for item in parsed:
                if item.get("class") != 0:   # only person class
                    continue
                box  = item.get("box", {})
                bbox = [box.get("x1"), box.get("y1"),
                        box.get("x2"), box.get("y2")]
                if all(v is not None for v in bbox):
                    person_detections.append({
                        "bbox":       bbox,
                        "confidence": item.get("confidence", 0.0),
                    })
        except Exception as e:
            logger.error(f"Failed to parse YOLO result: {e}")

        return person_detections

    # ── Frame download ───────────────────────────────────────
    def download_frame(self, url):
        try:
            resp  = requests.get(url, timeout=10)
            resp.raise_for_status()
            arr   = np.asarray(bytearray(resp.content), dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            logger.error(f"Failed to download frame: {e}")
            return None

    # ── Track ────────────────────────────────────────────────
    def run_strongsort(self, frame, person_detections):
        """
        Feed YOLO detections into StrongSORT and return outputs.
        person_detections: [{"bbox":[x1,y1,x2,y2], "confidence":float}, ...]
        """
        if not person_detections:
            # No detections — still need to increment ages so tracks age out
            self.tracker.increment_ages()
            return []

        # Build tensors
        xyxy  = torch.tensor(
            [d["bbox"] for d in person_detections], dtype=torch.float32
        )
        confs = torch.tensor(
            [d["confidence"] for d in person_detections], dtype=torch.float32
        )
        # All detections are persons (class 0)
        clss  = torch.zeros(len(person_detections), dtype=torch.float32)

        xywh  = xyxy2xywh_center(xyxy)

        outputs = self.tracker.update(xywh, confs, clss, frame)
        return outputs

    # ── Process one Redis message ────────────────────────────
    def process_message(self, message: dict):
        message   = {k.decode(): v.decode() for k, v in message.items()}
        frame_url = message["frame_url"]
        camera_id = message["camera_id"]
        timestamp = message["timestamp"]

        if not frame_url:
            logger.warning("Missing frame_url. Skipping.")
            return

        # 1. Get person detections from YOLO service
        person_detections = self.call_yolo_service(frame_url)
        num_persons       = len(person_detections)
        self.history.append(num_persons)
        avg_persons = int(sum(self.history) / len(self.history))

        # 2. Download frame (needed for StrongSORT ReID)
        frame = self.download_frame(frame_url)
        if frame is None:
            logger.error(f"Could not download frame from {frame_url}")
            return

        # 3. Run StrongSORT
        outputs = self.run_strongsort(frame, person_detections)

        # 4. Update duration tracker
        tracked_persons = []
        if len(outputs) > 0:
            track_ids = [int(o[4]) for o in outputs]
            bboxes    = [o[0:4].tolist() for o in outputs]
            self.duration_tracker.update(track_ids, self.frame_idx, bboxes)
        else:
            # Still run update so stale tracks get cleaned up
            self.duration_tracker.update([], self.frame_idx)

        tracked_persons = self.duration_tracker.get_tracked_persons()
        exceeded_count  = sum(1 for p in tracked_persons if p['exceeded_threshold'])

        self.frame_idx += 1

        # 5. Publish to output queue
        current_data = {
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
            self.redis.xadd(OUTPUT_QUEUE_1, {"data": json.dumps(current_data)})

            if VISUALIZE:
                _, buf      = cv2.imencode(".jpg", frame)
                frame_b64   = base64.b64encode(buf).decode("utf-8")
                vis_payload = {
                    "camera_id":      camera_id,
                    "timestamp":      timestamp,
                    "frame":          frame_b64,
                    "detections":     person_detections,
                    "tracked_persons": tracked_persons,
                    "num_persons":    num_persons,
                    "exceeded_count": exceeded_count,
                }
                self.vis_queue.push(json.dumps(vis_payload))

            logger.info(
                f"Camera {camera_id}: {num_persons} detected, "
                f"{len(tracked_persons)} tracked, {exceeded_count} exceeded"
            )
        except Exception as e:
            logger.error(f"Failed to publish result: {e}")

    # ── Main loop ────────────────────────────────────────────
    def run(self):
        logger.info(
            f"CrewRoomDetector started — threshold: {DURATION_THRESHOLD_MINUTES}min, "
            f"confidence: {DETECTION_CONFIDENCE}"
        )
        while True:
            try:
                resp = self.redis.xreadgroup(
                    groupname    = GROUP_NAME,
                    consumername = "worker-1",
                    streams      = {INPUT_QUEUE: ">"},
                    count        = 1,
                    block        = 5000,
                )
                if resp:
                    _, messages  = resp[0]
                    msg_id, msg  = messages[0]
                    self.process_message(msg)
                    self.redis.xack(INPUT_QUEUE, GROUP_NAME, msg_id)
                    self.mark_acked(INPUT_QUEUE, msg_id.decode(), GROUP_NAME)

            except json.JSONDecodeError:
                logger.error("Failed to decode JSON from queue.")
            except Exception as e:
                logger.exception(f"Unexpected error: {e}")
                time.sleep(1)


if __name__ == "__main__":
    service = CrewRoomDetector()
    service.run()