# import os
# import threading
# import time
# import logging
# import logging_loki
# import pymongo.errors as errors
# from dotenv import load_dotenv
# from pymongo import MongoClient, InsertOne
# from multiprocessing import Queue

# load_dotenv()

# MONGODB_SOURCE = os.getenv("MONGODB_SOURCE")
# MONGODB_DESTINATION = os.getenv("MONGODB_DESTINATION")
# MONGODB_NAME = os.getenv("MONGODB_NAME")
# SYNC_DURATION = float(os.getenv("SYNC_DURATION", 1)) * 60
# MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", 100))
# LOKI_URL = os.getenv("LOKI_URL", "http://localhost:3100/loki/api/v1/push")

# handler = logging_loki.LokiQueueHandler(
#     Queue(-1),
#     url=LOKI_URL,
#     tags={"application": "db_sync_service"},
#     version="1",
# )

# logger = logging.getLogger("db_sync_service")
# logger.setLevel(logging.INFO)
# logger.addHandler(handler)
# console_handler = logging.StreamHandler()
# logger.addHandler(console_handler)


# class DBSyncService:
#     def __init__(self):
#         self.mongodb_source = MONGODB_SOURCE
#         self.mongodb_destination = MONGODB_DESTINATION
#         self.mongodb_name = MONGODB_NAME
#         self.mongodb_source_client = MongoClient(self.mongodb_source)
#         self.mongodb_destination_client = MongoClient(self.mongodb_destination)
#         self.mongodb_source_db = self.mongodb_source_client[self.mongodb_name]
#         self.mongodb_destination_db = self.mongodb_destination_client[self.mongodb_name]

#     def check_health(self, db):
#         try:
#             health = db.command("ping")
#             if health.get("ok"):
#                 return True
#             else:
#                 return False
#         except Exception as e:
#             return False

#     def wait_for_destination(self, retry_interval=5):
#         while not self.check_health(self.mongodb_destination_db):
#             logger.info("Destination DB not healthy, waiting...")
#             time.sleep(retry_interval)

#     def get_resume_token(self, collection):
#         doc = self.mongodb_source_db["_sync_tokens"].find_one(
#             {"collection": collection}
#         )
#         return doc["resume_token"] if doc else None

#     def save_resume_token(self, collection, token):
#         self.mongodb_source_db["_sync_tokens"].update_one(
#             {"collection": collection}, {"$set": {"resume_token": token}}, upsert=True
#         )

#     def _flush_in_batches(self, dest_coll, collection_name, buffer):
#         """
#         Flush buffer in fixed-size batches to avoid huge bulk_write calls.
#         Resume token is advanced incrementally after each successful batch.
#         """
#         idx = 0
#         total = len(buffer)

#         while idx < total:
#             batch = buffer[idx : idx + MAX_BATCH_SIZE]

#             # Health check BEFORE each batch
#             if not self.check_health(self.mongodb_destination_db):
#                 logger.error("Destination DB not healthy during batch flush")
#                 raise RuntimeError("Destination DB unhealthy")

#             operations = [InsertOne(change["fullDocument"]) for change in batch]

#             try:
#                 dest_coll.bulk_write(operations, ordered=True)

#                 # Advance resume token safely to last successful event in this batch
#                 last_token = batch[-1]["_id"]
#                 self.save_resume_token(collection_name, last_token)

#                 logger.info(
#                     f"Flushed batch {idx // MAX_BATCH_SIZE + 1} ({len(batch)} docs) to {collection_name}"
#                 )

#                 idx += MAX_BATCH_SIZE

#             except errors.BulkWriteError as bwe:
#                 details = bwe.details or {}
#                 write_errors = details.get("writeErrors", [])

#                 if write_errors:
#                     first_error_index = write_errors[0]["index"]

#                     if first_error_index > 0:
#                         safe_token = batch[first_error_index - 1]["_id"]
#                         self.save_resume_token(collection_name, safe_token)

#                 logger.error(f"Bulk write error in batch: {bwe}")
#                 raise

#     def sync_collection(self, collection_name):
#         source = self.mongodb_source_db[collection_name]
#         dest = self.mongodb_destination_db[collection_name]

#         resume_token = self.get_resume_token(collection_name)

#         watch_kwargs = {
#             "full_document": "required",
#             # batch_size still useful for server getMore, but NOT logic
#             "batch_size": 100,
#             "max_await_time_ms": 1000,  # allows periodic wake-up
#         }

#         if resume_token:
#             watch_kwargs["resume_after"] = resume_token

#         pipeline = [{"$match": {"operationType": "insert"}}]

#         buffer = []
#         window_start = time.monotonic()

#         while True:
#             try:
#                 with source.watch(pipeline, **watch_kwargs) as stream:
#                     while True:
#                         change = stream.try_next()

#                         now = time.monotonic()

#                         # 1. Accumulate if event arrived
#                         if change is not None:
#                             buffer.append(change)

#                         # 2. Check time window
#                         if now - window_start >= SYNC_DURATION:
#                             if buffer:
#                                 self._flush_in_batches(dest, collection_name, buffer)
#                                 buffer.clear()

#                             # Reset window
#                             window_start = now

#                         # 3. Prevent busy spinning
#                         if change is None:
#                             time.sleep(0.1)

#             except (errors.PyMongoError, RuntimeError) as e:
#                 logger.error(f"[WARN] {collection_name}: {e}")

#                 # Flush anything accumulated BEFORE failure
#                 if buffer:
#                     try:
#                         self._flush_in_batches(dest, collection_name, buffer)
#                         buffer.clear()
#                     except Exception:
#                         pass  # do not advance token if this fails

#                 self.wait_for_destination()

#                 resume_token = self.get_resume_token(collection_name)
#                 watch_kwargs["resume_after"] = resume_token

#     def sync(self):
#         if not self.check_health(self.mongodb_source_db):
#             logger.error("Source DB not healthy")
#             raise RuntimeError("Source DB not healthy")

#         collections = self.mongodb_source_db.list_collection_names()
#         if "_sync_tokens" in collections:
#             collections.remove("_sync_tokens")

#         threads = []

#         for coll in collections:
#             t = threading.Thread(target=self.sync_collection, args=(coll,), daemon=True)
#             t.start()
#             threads.append(t)
#         for t in threads:
#             t.join()


# if __name__ == "__main__":
#     time.sleep(5)
#     logger.info("DB Sync Service starting...")
#     DBSyncService().sync()


import os
import threading
import time
import logging
import logging_loki
import pymongo.errors as errors
from dotenv import load_dotenv
from pymongo import MongoClient, InsertOne
from multiprocessing import Queue

load_dotenv()

MONGODB_SOURCE = os.getenv("MONGODB_SOURCE")
MONGODB_DESTINATION = os.getenv("MONGODB_DESTINATION")
MONGODB_NAME = os.getenv("MONGODB_NAME")
SYNC_DURATION = float(os.getenv("SYNC_DURATION", 1)) * 60
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", 100))
LOKI_URL = os.getenv("LOKI_URL", "http://localhost:3100/loki/api/v1/push")

handler = logging_loki.LokiQueueHandler(
    Queue(-1),
    url=LOKI_URL,
    tags={"application": "db_sync_service"},
    version="1",
)

logger = logging.getLogger("db_sync_service")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.addHandler(logging.StreamHandler())


class DBSyncService:
    def __init__(self):
        self.mongodb_source_client = MongoClient(MONGODB_SOURCE)
        self.mongodb_destination_client = MongoClient(MONGODB_DESTINATION)

        self.mongodb_source_db = self.mongodb_source_client[MONGODB_NAME]
        self.mongodb_destination_db = self.mongodb_destination_client[MONGODB_NAME]

    # ------------------ HEALTH ------------------

    def check_health(self, db):
        try:
            return bool(db.command("ping").get("ok"))
        except Exception:
            return False

    def wait_for_destination(self, retry_interval=5):
        while not self.check_health(self.mongodb_destination_db):
            logger.warning("Destination DB not healthy, waiting...")
            time.sleep(retry_interval)

    # ------------------ RESUME TOKEN ------------------

    def get_resume_token(self, collection):
        doc = self.mongodb_source_db["_sync_tokens"].find_one(
            {"collection": collection}
        )
        return doc["resume_token"] if doc else None

    def save_resume_token(self, collection, token):
        self.mongodb_source_db["_sync_tokens"].update_one(
            {"collection": collection},
            {"$set": {"resume_token": token}},
            upsert=True,
        )

    # ------------------ FLUSH LOGIC ------------------

    def _flush_in_batches(self, dest_coll, collection_name, buffer):
        idx = 0
        total = len(buffer)

        while idx < total:
            batch = buffer[idx : idx + MAX_BATCH_SIZE]

            if not self.check_health(self.mongodb_destination_db):
                raise RuntimeError("Destination DB unhealthy during flush")

            operations = [
                InsertOne(change["fullDocument"]) for change in batch
            ]

            try:
                # UNORDERED = allow duplicates to fail without stopping batch
                dest_coll.bulk_write(operations, ordered=False)

                # advance resume token
                self.save_resume_token(collection_name, batch[-1]["_id"])

                logger.info(
                    f"{collection_name}: inserted {len(batch)} docs"
                )

            except errors.BulkWriteError as bwe:
                write_errors = bwe.details.get("writeErrors", [])

                non_duplicate_errors = [
                    err for err in write_errors if err.get("code") != 11000
                ]

                if non_duplicate_errors:
                    logger.error(
                        f"{collection_name}: non-duplicate bulk error {non_duplicate_errors}"
                    )
                    raise

                # only duplicate errors → safe
                self.save_resume_token(collection_name, batch[-1]["_id"])

                logger.warning(
                    f"{collection_name}: ignored {len(write_errors)} duplicate inserts"
                )

            idx += MAX_BATCH_SIZE

    # ------------------ SYNC COLLECTION ------------------

    def sync_collection(self, collection_name):
        source = self.mongodb_source_db[collection_name]
        dest = self.mongodb_destination_db[collection_name]

        resume_token = self.get_resume_token(collection_name)

        watch_kwargs = {
            "full_document": "required",
            "batch_size": 100,
            "max_await_time_ms": 1000,
        }

        if resume_token:
            watch_kwargs["resume_after"] = resume_token

        pipeline = [{"$match": {"operationType": "insert"}}]

        buffer = []
        window_start = time.monotonic()

        while True:
            try:
                with source.watch(pipeline, **watch_kwargs) as stream:
                    while True:
                        change = stream.try_next()
                        now = time.monotonic()

                        if change is not None:
                            buffer.append(change)

                        if now - window_start >= SYNC_DURATION:
                            if buffer:
                                self._flush_in_batches(
                                    dest, collection_name, buffer
                                )
                                buffer.clear()

                            window_start = now

                        if change is None:
                            time.sleep(0.1)

            except (errors.PyMongoError, RuntimeError) as e:
                logger.error(f"[WARN] {collection_name}: {e}")

                if buffer:
                    try:
                        self._flush_in_batches(
                            dest, collection_name, buffer
                        )
                        buffer.clear()
                    except Exception:
                        pass

                self.wait_for_destination()

                resume_token = self.get_resume_token(collection_name)
                watch_kwargs["resume_after"] = resume_token

    # ------------------ MAIN ------------------

    def sync(self):
        if not self.check_health(self.mongodb_source_db):
            raise RuntimeError("Source DB not healthy")

        collections = self.mongodb_source_db.list_collection_names()
        collections = [c for c in collections if c != "_sync_tokens"]

        threads = []

        for coll in collections:
            t = threading.Thread(
                target=self.sync_collection,
                args=(coll,),
                daemon=True,
            )
            t.start()
            threads.append(t)

        for t in threads:
            t.join()


if __name__ == "__main__":
    time.sleep(5)
    logger.info("DB Sync Service starting...")
    DBSyncService().sync()
