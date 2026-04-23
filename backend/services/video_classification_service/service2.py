import json
from dotenv import load_dotenv
import redis
import logging
import time
import requests
import logging_loki
from queue import Queue
import threading
import sys
import os
import cv2
import numpy as np
import tempfile
from minio import Minio
from datetime import datetime
from collections import deque
from settings import SETTINGS
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
LOKI_URL = os.getenv("LOKI_URL", "http://localhost:3100/loki/api/v1/push")
XCLIP_MODEL_URL = os.getenv("XCLIP_MODEL_URL")
OUTPUT_QUEUE_NAME_1 = os.getenv("OUTPUT_QUEUE_NAME_1", "database:queue:0")
TARGET_COLLECTION_NAME = os.getenv(
    "TARGET_COLLECTION_NAME", "video_classification_collection"
)
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "video-clips")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

ACK_TTL_SECONDS = 10 * 60  # 10 minutes
MAX_HISTORY = 32

UPLOAD_POOL = ThreadPoolExecutor(max_workers=4)

handler = logging_loki.LokiQueueHandler(
    Queue(-1),
    url=LOKI_URL,
    tags={"application": "video_classification_service"},
    version="1",
)

logger = logging.getLogger("video_classification_service")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
# Also log to console
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)


class VideoClassificationService:
    def __init__(self, stream_name, group_name, fps, prompts, frame_overlap=5):
        self.redis = self.connect_redis(host=REDIS_HOST, port=REDIS_PORT)
        self.stream_name = stream_name
        self.group_name = group_name
        self.xclip_model_url = XCLIP_MODEL_URL
        self.history = deque(maxlen=MAX_HISTORY)
        self.message_ids = deque(maxlen=MAX_HISTORY)
        self.frame_overlap = frame_overlap
        self.fps = fps
        self.frame_interval = 1 / fps
        self.last_timestamp = None
        self.max_wait_time = 10  # Increased from 5 to 30 to be more robust
        self.prompts = prompts
        self.minio_client = self.get_minio()

    def get_minio(self):
        client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE,
        )

        if not client.bucket_exists(MINIO_BUCKET):
            client.make_bucket(MINIO_BUCKET)

        return client

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

    def classify_video(self):
        frames = [msg["frame_url"] for msg in self.history]

        prompts = self.prompts
        payload = {
            "image_urls": frames,
            "texts": prompts,
        }
        response = requests.post(self.xclip_model_url, json=payload)
        if response.status_code == 200:
            response_json = response.json()
            return response_json
        else:
            logging.error(f"Failed to classify video: {response.status_code}")
            return None
    
    def upload_and_emit(
        self,
        camera_id,
        history_snapshot,
        message_ids_snapshot,
        incidents,
    ):
        """
        Runs in background thread.
        Handles:
        - Video creation
        - Upload to MinIO
        - Emit to Redis
        - Custom ack marking
        """
        try:
            video_clip_url = self.upload_video_clip(history_snapshot)

            output_payload = {
                "camera_id": camera_id,
                "start_timestamp": history_snapshot[0]["timestamp"],
                "end_timestamp": history_snapshot[-1]["timestamp"],
                "results": incidents,
                "target_collection": TARGET_COLLECTION_NAME,
                "video_clip_url": video_clip_url or "no data available",
            }

            self.redis.xadd(
                OUTPUT_QUEUE_NAME_1,
                {"data": json.dumps(output_payload)},
            )

            logger.info(
                f"output emitted for camera {camera_id} "
                f"from {history_snapshot[0]['timestamp']} "
                f"to {history_snapshot[-1]['timestamp']}"
            )

            # Mark all frames as processed in custom ack system
            for mid in message_ids_snapshot:
                m_id_str = mid.decode() if isinstance(mid, bytes) else mid
                self.mark_acked(self.stream_name, m_id_str, self.group_name)

        except Exception:
            logger.exception("Background upload_and_emit failed")

    def create_video_from_frames(self, history):
        """
        Downloads frames from URLs and creates a video file.
        Returns the path to the temporary video file.
        """
        temp_dir = tempfile.gettempdir()
        timestamp = int(time.time())
        video_path = os.path.join(temp_dir, f"clip_{timestamp}.mp4")

        frames = []
        for item in history:
            url = item["frame_url"]
            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    nparr = np.frombuffer(resp.content, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        frames.append(frame)
            except Exception as e:
                logger.error(f"Failed to download frame from {url}: {e}")

        if not frames:
            logger.error("No frames downloaded, cannot create video.")
            return None

        height, width, layers = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(video_path, fourcc, self.fps, (width, height))

        for frame in frames:
            video.write(frame)

        video.release()
        return video_path

    def upload_video_clip(self, history):
        """
        Creates a video clip from history frames and uploads it to Minio.
        Returns the presigned URL of the uploaded video.
        """
        video_path = self.create_video_from_frames(history)
        if not video_path:
            return None

        try:
            file_name = os.path.basename(video_path)
            file_size = os.path.getsize(video_path)

            with open(video_path, "rb") as f:
                self.minio_client.put_object(
                    MINIO_BUCKET,
                    file_name,
                    f,
                    file_size,
                    content_type="video/mp4",
                )

            # Get presigned URL (valid for 7 days by default)
            url = self.minio_client.get_presigned_url(
                "GET",
                MINIO_BUCKET,
                file_name,
            )

            # Cleanup local file
            os.remove(video_path)

            return url
        except Exception as e:
            logger.exception(f"Failed to upload video clip: {e}")
            if os.path.exists(video_path):
                os.remove(video_path)
            return None

    @staticmethod
    def parse_timestamp(ts: str) -> float:
        """
        Convert ISO-8601 timestamp with timezone to epoch seconds
        """
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()

    def process_message(self, message, msg_id):
        # Decode Redis message bytes
        msg_id_str = msg_id.decode() if isinstance(msg_id, bytes) else msg_id
        message = {
            k.decode() if isinstance(k, bytes) else k: (
                v.decode() if isinstance(v, bytes) else v
            )
            for k, v in message.items()
        }

        frame_url = message["frame_url"]
        camera_id = message["camera_id"]
        timestamp_str = message["timestamp"]

        try:
            current_ts = self.parse_timestamp(timestamp_str)
        except Exception as e:
            logger.error(f"Failed to parse timestamp {timestamp_str}: {e}")
            return

        # Skip logic based on target FPS
        if self.last_timestamp is None:
            # This is the first frame, we process it
            self.last_timestamp = current_ts
        else:
            time_diff = current_ts - self.last_timestamp
            if time_diff < self.frame_interval:
                # This frame arrived too soon after the last processed frame
                # Acknowledge it in our custom system and skip processing
                self.mark_acked(self.stream_name, msg_id_str, self.group_name)
                return

            # This frame meets the interval threshold, update last processed time
            self.last_timestamp = current_ts

        # Handle potential stream breaks (e.g., if source was down)
        if self.history:
            last_history_ts = self.parse_timestamp(self.history[-1]["timestamp"])
            time_gap = current_ts - last_history_ts
            if time_gap > self.max_wait_time:
                logger.warning(
                    f"Large time gap detected ({time_gap:.2f}s), resetting state. "
                    f"Current frame: {timestamp_str}, last history frame: {self.history[-1]['timestamp']}"
                )
                # Mark all pending frames in history as seen before clearing
                for mid in self.message_ids:
                    m_id_str = mid.decode() if isinstance(mid, bytes) else mid
                    self.mark_acked(self.stream_name, m_id_str, self.group_name)
                self.history.clear()
                self.message_ids.clear()

        # --- Accumulate frame ---
        self.history.append({"frame_url": frame_url, "timestamp": timestamp_str})
        self.message_ids.append(msg_id)

        if len(self.history) < MAX_HISTORY:
            return

        # --- Sequence full, perform classification ---
        result = self.classify_video()
        if not result:
            logger.error("Classification failed")
            return
        probs = result.get("probs")
        texts = result.get("texts")
        # print(result)
        incidents = []
        for i in range(len(probs)):
            p_arr = probs[i]
            t_arr = texts[i].get("prompt")
            print("p_arr", p_arr)
            print("t_arr", t_arr)
            max_idx = p_arr.index(max(p_arr))
            if max_idx == 0 and p_arr[max_idx] >= 0.7:
                incidents.append(
                    {"incident": t_arr[max_idx], "probability": p_arr[max_idx]}
                )
        # print("incidents", incidents)
        if incidents:
            # Snapshot state BEFORE clearing
            history_snapshot = list(self.history)
            message_ids_snapshot = list(self.message_ids)

            # Submit background job
            UPLOAD_POOL.submit(
                self.upload_and_emit,
                camera_id,
                history_snapshot,
                message_ids_snapshot,
                incidents,
            )

        self.history.clear()
        self.message_ids.clear()

    def run(self, group_name, input_queue, output_queue):
        logger.info("video classification service started, listening on input queue.")
        while True:
            try:
                resp = self.redis.xreadgroup(
                    groupname=group_name,
                    consumername="worker-1",
                    streams={input_queue: ">"},
                    count=10,  # Process in batches for better efficiency with skipped frames
                    block=5000,
                )

                if resp:
                    for stream_name, messages in resp:
                        for msg_id, message in messages:
                            self.process_message(message, msg_id)
                            # Always acknowledge in Redis to move the cursor forward
                            self.redis.xack(input_queue, group_name, msg_id)

            except json.JSONDecodeError:
                logger.error("Failed to decode JSON message from input queue.")
            except Exception as e:
                logger.exception(f"Unexpected error in main loop: {e}")
                time.sleep(1)


def run_service_thread(stream_name, stream_config):
    service = VideoClassificationService(
        stream_name=stream_name,
        group_name=stream_config["group_name"],
        fps=stream_config["fps"],
        prompts=stream_config["prompts"],
    )
    service.run(
        group_name=stream_config["group_name"],
        input_queue=stream_name,
        output_queue=OUTPUT_QUEUE_NAME_1,
    )


def main():
    threads = []

    for stream_name, cfg in SETTINGS.items():
        t = threading.Thread(
            target=run_service_thread,
            args=(stream_name, cfg),
            daemon=True,  # important
            name=f"consumer-{stream_name}",
        )
        t.start()
        threads.append(t)

    # Keep main thread alive
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
        sys.exit(0)


if __name__ == "__main__":
    main()
