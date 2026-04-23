# import os
# import json
# import time
# import redis
# from multiprocessing import Queue
# import logging
# import logging_loki
# from pymongo import MongoClient
# from dotenv import load_dotenv
# from datetime import datetime

# # Load environment variables
# load_dotenv()

# # Configuration
# REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
# REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
# REDIS_QUEUE = os.getenv("INPUT_QUEUE_NAME", "database:queue:0")

# MONGO_URI = os.getenv("MONGO_URI")
# MONGO_DB = os.getenv("MONGO_DB", "my_database")
# MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "test")

# LOKI_URL = os.getenv("LOKI_URL", "http://localhost:3100/loki/api/v1/push")
# handler = logging_loki.LokiQueueHandler(
#     Queue(-1),
#     url=LOKI_URL,
#     tags={"application": "db_handling_service"},
#     version="1",
# )

# logger = logging.getLogger("db_handling_service")
# logger.setLevel(logging.INFO)
# logger.addHandler(handler)
# # Also log to console
# console_handler = logging.StreamHandler()
# logger.addHandler(console_handler)


# def get_redis_connection(retry_delay=5):
#     while True:
#         try:
#             r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
#             # Test connection
#             r.ping()
#             logging.info("Connected to Redis successfully.")
#             return r

#         except Exception as e:
#             logging.error(
#                 f"Redis connection failed: {e}. Retrying in {retry_delay} seconds..."
#             )
#             time.sleep(retry_delay)


# def get_mongo_db():
#     client = MongoClient(MONGO_URI)
#     return client[MONGO_DB]


# def process_message(db, default_collection, data):
#     try:
#         raw = data.get(b"data") or data.get("data")
#         if raw is None:
#             logger.error(f"Stream message missing 'data' field: {data}")
#             return

#         # raw is bytes → convert to json
#         data = json.loads(raw.decode())
#         if data is None:
#             logger.error("data is not in the proper format")

#         target_collection = default_collection
#         if "target_collection" in data:
#             collection_name = data.pop("target_collection")
#             target_collection = db[collection_name]
#         if "timestamp" in data and isinstance(data["timestamp"], str):
#             try:
#                 data["timestamp"] = datetime.fromisoformat(data["timestamp"])
#             except Exception:
#                 logger.error(f"Invalid timestamp format: {data['timestamp']}")
#         # Insert into MongoDB
#         insert_result = target_collection.insert_one(data)
#         logger.info(
#             f"Inserted document into '{target_collection.name}' with ID: {insert_result.inserted_id}"
#         )
#     except Exception as e:
#         logger.error(f"Error processing message: {e}")


# def process_data():
#     # Connect to Redis
#     try:
#         r = get_redis_connection()
#         r.ping()
#         logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
#     except Exception as e:
#         logger.error(f"Failed to connect to Redis: {e}")
#         return

#     # Connect to MongoDB
#     try:
#         db = get_mongo_db()
#         default_collection = db[MONGO_COLLECTION]
#         logger.info(f"Connected to MongoDB database: {MONGO_DB}")
#     except Exception as e:
#         logger.error(f"Failed to connect to MongoDB: {e}")
#         return

#     # Default to stream if key is stream or doesn't exist (assuming producer creates stream)
#     is_stream = True

#     group_name = os.getenv("REDIS_GROUP", "db_service_group")
#     consumer_name = os.getenv("HOSTNAME", "consumer-1")

#     if is_stream:
#         try:
#             r.xgroup_create(REDIS_QUEUE, group_name, id="0", mkstream=True)
#         except redis.exceptions.ResponseError as e:
#             if "BUSYGROUP" not in str(e):
#                 logger.warning(f"Warning: Failed to create group: {e}")
#         logger.info(
#             f"Listening for stream messages on {REDIS_QUEUE} (Group: {group_name})..."
#         )
#     else:
#         logger.info(f"Listening for list messages on {REDIS_QUEUE}...")

#     while True:
#         try:
#             if is_stream:
#                 # Stream consumption
#                 entries = r.xreadgroup(
#                     group_name, consumer_name, {REDIS_QUEUE: ">"}, count=1, block=0
#                 )
#                 if entries:
#                     stream, messages = entries[0]
#                     for message_id, data in messages:
#                         logger.info(f"Received stream message: {data}")
#                         process_message(db, default_collection, data)
#                         r.xack(REDIS_QUEUE, group_name, message_id)
#             else:
#                 # List consumption
#                 result = r.blpop(REDIS_QUEUE, timeout=0)
#                 if result:
#                     _, message = result
#                     logger.info(f"Received list message: {message}")
#                     try:
#                         data = json.loads(message)
#                         if not isinstance(data, dict):
#                             data = {"data": data}
#                     except json.JSONDecodeError:
#                         data = {"raw_message": message}

#                     process_message(db, default_collection, data)

#         except redis.exceptions.ResponseError as e:
#             if "WRONGTYPE" in str(e):
#                 logger.error(f"WRONGTYPE detected. Switching mode...")
#                 is_stream = not is_stream
#                 if is_stream:
#                     try:
#                         r.xgroup_create(REDIS_QUEUE, group_name, id="0", mkstream=True)
#                     except redis.exceptions.ResponseError:
#                         pass
#                     logger.error(f"Switched to Stream mode.")
#                 else:
#                     logger.info(f"Switched to List mode.")
#                 time.sleep(1)
#             else:
#                 logger.error(f"Redis error: {e}")
#                 time.sleep(5)

#         except redis.exceptions.ConnectionError:
#             logger.error("Redis connection lost. Retrying in 5 seconds...")
#             time.sleep(5)
#             try:
#                 r = get_redis_connection()
#             except:
#                 pass
#         except Exception as e:
#             logger.error(f"An unexpected error occurred: {e}")
#             time.sleep(1)


# if __name__ == "__main__":
#     if not MONGO_URI:
#         logger.error("Error: MONGO_URI environment variable is not set.")
#         logger.error("Please set MONGO_URI in a .env file or environment variables.")
#     else:
#         process_data()


import os
import orjson as json
import time
import redis
import logging
import logging_loki
from queue import Queue
from datetime import datetime
from pymongo import MongoClient, InsertOne, UpdateOne
from bson import ObjectId
from dotenv import load_dotenv

load_dotenv()


class MongoRedisBatchConsumer:

    def __init__(self):
        # Redis config
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", 6379))
        self.stream_key = os.getenv("INPUT_QUEUE_NAME", "database:queue:0")
        self.group_name = os.getenv("REDIS_GROUP", "db_service_group")
        self.consumer_name = os.getenv("HOSTNAME", "consumer-1")

        # Mongo config
        self.mongo_uri = os.getenv("MONGO_URI")
        self.mongo_db_name = os.getenv("MONGO_DB", "my_database")
        self.default_collection_name = os.getenv("MONGO_COLLECTION", "test")

        # Batch config
        self.redis_batch_size = 200
        self.mongo_bulk_size = 300
        self.flush_interval = 1.0

        # Internal batch buffers
        self.collection_batches = {}
        self.last_flush_time = time.time()

        # Logging
        self.logger = logging.getLogger("MongoRedisBatchConsumer")
        self._setup_logging()

        # Init connections
        self.redis = self._connect_redis()
        self.mongo_db = self._connect_mongo()

        # Ensure consumer group exists
        self._init_stream_group()

    # ------------------------------------------------------------------
    # Logging Setup
    # ------------------------------------------------------------------
    def _setup_logging(self):
        handler = logging_loki.LokiQueueHandler(
            Queue(-1),
            url=os.getenv("LOKI_URL", "http://localhost:3100/loki/api/v1/push"),
            tags={"application": "db_handling_service"},
            version="1",
        )
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(handler)
        self.logger.addHandler(logging.StreamHandler())

    # ------------------------------------------------------------------
    # Connect Redis & Mongo
    # ------------------------------------------------------------------
    def _connect_redis(self):
        while True:
            try:
                r = redis.Redis(
                    host=self.redis_host, port=self.redis_port, decode_responses=False
                )
                r.ping()
                self.logger.info("Connected to Redis.")
                return r
            except Exception as e:
                self.logger.error(f"Redis connection failed: {e}. Retrying...")
                time.sleep(2)

    def _connect_mongo(self):
        try:
            client = MongoClient(self.mongo_uri)
            db = client[self.mongo_db_name]
            self.logger.info(f"Connected to MongoDB: {self.mongo_db_name}")
            return db
        except Exception as e:
            self.logger.error(f"Mongo DB connection failed: {e}")
            raise

    # ------------------------------------------------------------------
    # Create Stream Group if not exists
    # ------------------------------------------------------------------
    def _init_stream_group(self):
        try:
            self.redis.xgroup_create(
                self.stream_key, self.group_name, id="0", mkstream=True
            )
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                self.logger.warning(f"Group create error: {e}")

    # ------------------------------------------------------------------
    # Start Consumer Loop
    # ------------------------------------------------------------------
    def start(self):
        self.logger.info(f"Consumer started on stream={self.stream_key}")

        while True:
            try:
                entries = self.redis.xreadgroup(
                    self.group_name,
                    self.consumer_name,
                    {self.stream_key: ">"},
                    count=self.redis_batch_size,
                    block=500,
                )

                if entries:
                    _, messages = entries[0]
                    self._process_message_batch(messages)

                # Flush periodically
                if time.time() - self.last_flush_time >= self.flush_interval:
                    self._flush_all_batches()

            except redis.exceptions.ConnectionError:
                self.logger.error("Redis disconnected! Reconnecting...")
                self.redis = self._connect_redis()

            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
                time.sleep(1)

    # ------------------------------------------------------------------
    # Process Redis Messages in Batch
    # ------------------------------------------------------------------
    def _process_message_batch(self, messages):
        processed_ids = []  # For bulk ACK + DEL

        for message_id, msg_data in messages:
            try:
                raw = msg_data.get(b"data") or msg_data.get("data")
                data = json.loads(raw.decode())

                # Determine collection
                collection_name = data.pop(
                    "target_collection", self.default_collection_name
                )

                # Fix timestamp format
                for k, v in data.items():
                    if "timestamp" in k and isinstance(v, str):
                        data[k] = datetime.fromisoformat(v)

                # Timestamp 1: Buffer Entry
                data["db_received_at"] = datetime.now()

                # Prepare for batch
                if collection_name not in self.collection_batches:
                    self.collection_batches[collection_name] = []
                self.collection_batches[collection_name].append(data)

                processed_ids.append(message_id)

            except Exception as e:
                self.logger.error(f"Message processing error: {e}")

        # ---------------------------
        # BULK ACK + BULK DELETE HERE
        # ---------------------------
        if processed_ids:
            try:
                self.redis.xack(self.stream_key, self.group_name, *processed_ids)
                self.redis.xdel(self.stream_key, *processed_ids)
            except Exception as e:
                self.logger.error(f"ACK/DEL error: {e}")

        # auto-flush if any batch is too large
        for col_name, ops in self.collection_batches.items():
            if len(ops) >= self.mongo_bulk_size:
                self._flush_collection(col_name)

    # ------------------------------------------------------------------
    # Flush one collection batch
    # ------------------------------------------------------------------
    def _flush_collection(self, col_name):
        ops = self.collection_batches.get(col_name, [])
        if not ops:
            return

        try:
            # Timestamp 2: Flush Start (Client-side)
            # Timestamp 3: Mongo Arrival (Server-side via $currentDate)

            now_dt = datetime.now()
            bulk_ops = []

            for doc in ops:
                # Ensure ID exists for upsert
                if "_id" not in doc:
                    doc["_id"] = ObjectId()

                # Add flush timestamp
                doc["batch_pushed_at"] = now_dt

                # Use UpdateOne with upsert to allow $currentDate
                op = UpdateOne(
                    {"_id": doc["_id"]},
                    {"$set": doc, "$currentDate": {"mongo_reached_at": True}},
                    upsert=True,
                )
                bulk_ops.append(op)

            collection = self.mongo_db[col_name]
            collection.bulk_write(bulk_ops, ordered=False)
            self.logger.info(f"Flushed {len(ops)} docs → {col_name}")
        except Exception as e:
            self.logger.error(f"Bulk write failed for {col_name}: {e}")

        self.collection_batches[col_name] = []

    # ------------------------------------------------------------------
    # Flush ALL batches
    # ------------------------------------------------------------------
    def _flush_all_batches(self):
        for col_name in list(self.collection_batches.keys()):
            self._flush_collection(col_name)

        self.last_flush_time = time.time()


# ----------------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------------
if __name__ == "__main__":
    consumer = MongoRedisBatchConsumer()
    consumer.start()
