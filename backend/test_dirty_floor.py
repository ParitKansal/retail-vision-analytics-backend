
import redis
import time
import os

print("Connecting to Redis...")
try:
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    r.ping()
    print("Connected to Redis")
except Exception as e:
    print(f"Failed to connect: {e}")
    exit(1)

stream_name = "stream:cam:1"
message = {
    "camera_id": "1",
    "frame_url": "/workspace/test_frame.jpg",
    "timestamp": str(time.time())
}

print(f"Adding message to stream {stream_name}: {message}")
try:
    msg_id = r.xadd(stream_name, message)
    print(f"Message added successfully. ID: {msg_id}")
except Exception as e:
    print(f"Failed to add message: {e}")

print("Done. Check dirty_floor_service logs.")
