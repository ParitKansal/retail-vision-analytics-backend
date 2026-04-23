from minio import Minio
import os
import redis
from threading import Thread
import time

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "frames")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

DELETE_DUR = int(os.getenv("DELETE_DUR", "5"))
STREAM_NAME_TEMPLATE = "stream:cam:{}"

from settings import CAMERAS


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



def continue_run(stream_name, stream_groups):
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
    minio_client = get_minio()
    while True:
        delete_fully_acknowledged_messages(r, minio_client, stream_name)
        time.sleep(DELETE_DUR)



def main():
    threads = []
    for cam_id, cfg in CAMERAS.items():
        t = Thread(
            target=continue_run,
            args=(
                STREAM_NAME_TEMPLATE.format(cam_id),
                cfg["consumer_groups"],
            ),
            daemon=False,
        )
        t.start()
        threads.append(t)
        time.sleep(1.5)

    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
