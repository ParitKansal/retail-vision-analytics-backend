# Redis to MongoDB Service

This service continuously reads messages from a Redis queue and pushes them to a MongoDB collection.

## Prerequisites

- Python 3.x
- Redis server running
- MongoDB (Cloud/Atlas or local)

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configuration:**
    Copy `.env.example` to `.env` and update the values:
    ```bash
    cp .env.example .env
    ```
    
    Update `.env` with your actual credentials:
    - `REDIS_HOST`: Hostname of your Redis server.
    - `REDIS_PORT`: Port of your Redis server.
    - `REDIS_QUEUE`: Name of the Redis list/queue to pop from.
    - `MONGO_URI`: Your MongoDB connection string (e.g., from MongoDB Atlas).
    - `MONGO_DB`: The database name.
    - `MONGO_COLLECTION`: The collection name.

## Running the Service

Run the service using:

```bash
python service.py
```

## How it works

- The service connects to Redis and MongoDB on startup.
- It uses `blpop` to block and wait for new messages in the specified Redis queue.
- When a message arrives:
    - It attempts to parse it as JSON.
    - If parsing fails, it wraps the raw message in a dictionary `{"raw_message": ...}`.
    - **Dynamic Collection Routing:** It checks for a `target_collection` field in the JSON data.
        - If present, it uses that value as the collection name and removes the field from the document.
        - If absent, it uses the default `MONGO_COLLECTION` defined in `.env`.
    - It inserts the document into the selected MongoDB collection.

## Message Format Example

To send data to a specific collection (e.g., `user_logs`), push a JSON string like this to Redis:

```json
{
  "target_collection": "user_logs",
  "user_id": 123,
  "action": "login"
}
```

If `target_collection` is omitted, it goes to the default collection.
