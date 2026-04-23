import os
import io
import cv2
import time
import redis
import logging
import datetime
import logging_loki
from multiprocessing import Process, Queue
import multiprocessing as mp
from minio import Minio
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────
# Environment & Timezone
# ─────────────────────────────────────────────────────────────

os.environ.setdefault("TZ", "Asia/Kolkata")
time.tzset()  # unix-only

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "frames")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

LOKI_URL = os.getenv("LOKI_URL", "http://localhost:3100/loki/api/v1/push")

STREAM_NAME_TEMPLATE = "stream:cam:{}"

# ─────────────────────────────────────────────────────────────
# Load Camera Configuration
# ─────────────────────────────────────────────────────────────

from settings import CAMERAS

# ─────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────


def setup_logger():
    handler = logging_loki.LokiQueueHandler(
        Queue(-1),
        url=LOKI_URL,
        tags={"application": "stream_handling_service"},
        version="1",
    )

    logger = logging.getLogger(f"stream_service_{os.getpid()}")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    console = logging.StreamHandler()
    logger.addHandler(console)
    return logger


def get_redis():
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        decode_responses=False,
    )


def wait_for_redis(r):
    print("Waiting for Redis...")
    while True:
        try:
            if r.ping():
                print("Redis is ready")
                break
        except (redis.exceptions.ConnectionError, redis.exceptions.BusyLoadingError):
            time.sleep(1)


def get_minio():
    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE,
    )

    if not client.bucket_exists(MINIO_BUCKET):
        client.make_bucket(MINIO_BUCKET)

    return client


def ensure_stream_group(r, stream_name, group_name):
    try:
        r.xgroup_create(
            name=stream_name,
            groupname=group_name,
            id="$",
            mkstream=True,
        )
        print(f"Created group {group_name} for {stream_name}")
    except redis.exceptions.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise


def get_file_name_from_url(url: str) -> str:
    return url.split("/")[-1].split("?")[0]


def delete_fully_acknowledged_messages(r, minio_client, stream_name: str):
    exp_key = f"ack:exp:{stream_name}"
    expected = int(r.get(exp_key) or 0)

    if expected == 0:
        return

    start_id = "0-0"

    while True:
        msgs = r.xrange(stream_name, min=start_id, max="+", count=500)
        if not msgs:
            break

        for msg_id, data in msgs:
            msg_id_str = msg_id.decode()
            cnt_key = f"ack:cnt:{stream_name}:{msg_id_str}"

            ack_count = r.get(cnt_key)
            if not ack_count or int(ack_count) < expected:
                continue

            try:
                frame_url = data[b"frame_url"].decode()
                file_name = get_file_name_from_url(frame_url)

                pipe = r.pipeline()
                pipe.xdel(stream_name, msg_id)
                pipe.delete(cnt_key)
                pipe.delete(f"ack:seen:{stream_name}:{msg_id_str}")
                pipe.execute()

                minio_client.remove_object(MINIO_BUCKET, file_name)

                print(f"Deleted frame {file_name}")

            except Exception as e:
                print(f"Cleanup failed for {msg_id_str}: {e}")

        # Move start_id to the next possible ID to avoid infinite loop
        last_id = msgs[-1][0].decode()
        parts = last_id.split("-")
        if len(parts) == 2:
            start_id = f"{parts[0]}-{int(parts[1]) + 1}"
        else:
            # Fallback if ID format is unexpected, though highly unlikely in Redis
            break


# ─────────────────────────────────────────────────────────────
# Camera Process
# ─────────────────────────────────────────────────────────────


def set_expected_ack_count(r, stream_name: str, consumer_groups: list[str]):
    exp_key = f"ack:exp:{stream_name}"
    expected = len(consumer_groups)

    # SET ensures count is updated if configuration changes
    r.set(exp_key, expected)


def camera_process(cam_id, rtsp_url, sample_fps, consumer_groups):
    # logger = setup_logger()
    r = get_redis()
    wait_for_redis(r)
    minio_client = get_minio()

    stream_name = STREAM_NAME_TEMPLATE.format(cam_id)

    for group in consumer_groups:
        ensure_stream_group(r, stream_name, group)

    set_expected_ack_count(r, stream_name, consumer_groups)

    print(f"[Camera {cam_id}] Starting process")

    while True:
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            print(f"[Camera {cam_id}] Cannot open RTSP, retrying")
            time.sleep(5)
            continue

        print(f"[Camera {cam_id}] Stream opened")

        next_deadline = time.monotonic()

        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"[Camera {cam_id}] Frame read failed")
                cap.release()
                break

            now = time.monotonic()
            if now < next_deadline:
                continue

            next_deadline += 1.0 / sample_fps

            ok, jpg = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 80],
            )
            if not ok:
                continue

            try:
                ts = datetime.datetime.now().astimezone()
                file_name = f"cam_{cam_id}_{int(ts.timestamp() * 1000)}.jpg"
                data = jpg.tobytes()

                minio_client.put_object(
                    MINIO_BUCKET,
                    file_name,
                    io.BytesIO(data),
                    len(data),
                    content_type="image/jpeg",
                )

                url = minio_client.get_presigned_url(
                    "GET",
                    MINIO_BUCKET,
                    file_name,
                )

                r.xadd(
                    stream_name,
                    {
                        "frame_url": url,
                        "timestamp": str(ts.isoformat()),
                        "camera_id": cam_id,
                        "frame_shape": str(list(frame.shape)),
                    },
                )

                # delete_fully_acknowledged_messages(
                #     r,
                #     minio_client,
                #     stream_name,
                # )

                print(f"[Camera {cam_id}] Frame pushed @ {str(ts.isoformat())}")

            except Exception as e:
                print(f"[Camera {cam_id}] Processing error: {e}")
                time.sleep(1)
                break


# ─────────────────────────────────────────────────────────────
# Main Supervisor
# ─────────────────────────────────────────────────────────────


def main():
    processes = []

    for cam_id, cfg in CAMERAS.items():
        p = Process(
            target=camera_process,
            args=(
                cam_id,
                cfg["rtsp"],
                cfg["sample_fps"],
                cfg["consumer_groups"],
            ),
            daemon=False,
        )
        p.start()
        processes.append(p)

        # Stagger startup to avoid resource storm
        time.sleep(1.5)

    for p in processes:
        p.join()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()