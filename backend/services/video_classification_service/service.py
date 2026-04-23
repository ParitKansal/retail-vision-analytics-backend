import json
from dotenv import load_dotenv
import redis
import logging
import time
import requests
import logging_loki
from queue import Queue, Empty
import threading
import sys
import os
import cv2
import numpy as np
import tempfile
from s3_manager import S3Manager
from datetime import datetime
from collections import deque
from settings import SETTINGS
import ffmpeg

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
LOKI_URL = os.getenv("LOKI_URL", "http://localhost:3100/loki/api/v1/push")
XCLIP_MODEL_URL = os.getenv("XCLIP_MODEL_URL")
OUTPUT_QUEUE_NAME_1 = os.getenv("OUTPUT_QUEUE_NAME_1", "database:queue:0")
TARGET_COLLECTION_NAME = os.getenv(
    "TARGET_COLLECTION_NAME", "video_classification_collection"
)

AWS_ACCESS_KEY_ID = os.getenv("ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("SECRET_KEY")
AWS_REGION = os.getenv("REGION", "ap-south-1")
S3_BUCKET = os.getenv("BUCKET_NAME", "xelp-mcd")

ACK_TTL_SECONDS = 10 * 60  # 10 minutes
MAX_HISTORY = 32


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
        self.s3_manager = self.get_s3_manager()
        self.upload_queue = Queue()  # Queue for async uploads
        self.start_upload_worker()  # Start background upload thread

    def get_s3_manager(self):
        return S3Manager(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION,
        )

    def start_upload_worker(self):
        """Start background thread for async S3 uploads"""
        upload_thread = threading.Thread(
            target=self._upload_worker,
            daemon=True,
            name=f"upload-worker-{self.stream_name}",
        )
        upload_thread.start()

    def _upload_worker(self):
        """Background worker that processes S3 uploads from queue"""
        while True:
            try:
                video_path = self.upload_queue.get(timeout=5)
                if video_path is None:  # Sentinel value to stop
                    break

                self._upload_to_s3(video_path)
                self.upload_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in upload worker: {e}")
                time.sleep(1)

    def _transcode_for_browser(self, input_path):
        """
        Transcode video to H.264/AAC MP4 with faststart.
        Keeps the original filename.
        Returns input_path on success, None on failure.
        """
        temp_output_path = f"{input_path}.tmp.mp4"

        try:
            (
                ffmpeg.input(input_path)
                .output(
                    temp_output_path,
                    vcodec="libx264",
                    preset="fast",
                    crf=23,
                    pix_fmt="yuv420p",  # important for browser compatibility
                    acodec="aac",
                    audio_bitrate="128k",
                    movflags="+faststart",
                )
                .overwrite_output()
                .run(quiet=True, capture_stdout=True, capture_stderr=True)
            )

            os.replace(temp_output_path, input_path)
            return input_path

        except ffmpeg.Error as e:
            err_msg = e.stderr.decode('utf-8') if e.stderr else str(e)
            logger.error(f"Error in transcode_for_browser (ffmpeg detail): {err_msg}")
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
            return None
        except Exception as e:
            logger.error(f"Error in transcode_for_browser: {e}")
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
            return None

    def _upload_to_s3(self, video_path):
        """Transcode video for browser compatibility, then upload to S3"""
        transcoded_path = None
        try:
            # _transcode_for_browser now replaces the original file, so video_path itself becomes the transcoded path
            transcoded_path = self._transcode_for_browser(video_path)
            upload_path = transcoded_path if transcoded_path else video_path
            file_name = os.path.basename(upload_path)
            self.s3_manager.upload_file(upload_path, S3_BUCKET, file_name)
            logger.info(f"Video uploaded to S3: {file_name}")
        except Exception as e:
            logger.error(f"Failed to upload video to S3: {e}")
        finally:
            # Cleanup both the original and transcoded temp files
            for path in [video_path, transcoded_path]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception as e:
                        logger.error(f"Failed to remove temp video file {path}: {e}")

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
        Creates a video clip from history frames and queues it for async S3 upload.
        Returns the S3 object name.
        """
        video_path = self.create_video_from_frames(history)
        if not video_path:
            return None

        try:
            file_name = os.path.basename(video_path)
            # Queue the upload asynchronously
            self.upload_queue.put(video_path)
            logger.info(f"Queued video for S3 upload: {file_name}")
            # Return the S3 object name/path
            return f"s3://{S3_BUCKET}/{file_name}"
        except Exception as e:
            logger.exception(f"Failed to queue video upload: {e}")
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

        try:
            # --- Sequence full, perform classification ---
            result = self.classify_video()
            if not result:
                logger.error("Classification failed")
                # We still proceed to acknowledge and clear so we don't get stuck in a loop
            else:
                probs = result.get("probs")
                texts = result.get("texts")
                print(result)
                incidents = []
                for i in range(len(probs)):
                    p_arr = probs[i]
                    t_arr = texts[i].get("prompt")
                    print("p_arr", p_arr)
                    print("t_arr", t_arr)
                    max_idx = p_arr.index(max(p_arr))
                    if max_idx == 0 and p_arr[max_idx] >= 0.75:
                        incidents.append(
                            {"incident": t_arr[max_idx], "probability": p_arr[max_idx]}
                        )
                print("incidents", incidents)
                if incidents:
                    output_payload = {
                        "camera_id": camera_id,
                        "start_timestamp": self.history[0]["timestamp"],
                        "end_timestamp": self.history[-1]["timestamp"],
                        "results": incidents,
                        "target_collection": TARGET_COLLECTION_NAME,
                    }
                    # make video clip out of frames then upload to s3 and get url store it in output payload
                    video_clip_url = self.upload_video_clip(self.history)
                    if not video_clip_url:
                        logger.error("Failed to upload video clip")
                        output_payload["video_clip_url"] = "no data available"
                    else:
                        output_payload["video_clip_url"] = video_clip_url

                    self.redis.xadd(
                        OUTPUT_QUEUE_NAME_1,
                        {"data": json.dumps(output_payload)},
                    )
                    logger.info(
                        f"output emitted for cameras {camera_id} from {self.history[0]['timestamp']} to {self.history[-1]['timestamp']}"
                    )
        except Exception as e:
            logger.exception(f"Error during classification or output: {e}")
        finally:
            # Mark ALL frames in this clip as processed in our custom system, always
            for mid in self.message_ids:
                m_id_str = mid.decode() if isinstance(mid, bytes) else mid
                self.mark_acked(self.stream_name, m_id_str, self.group_name)

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
