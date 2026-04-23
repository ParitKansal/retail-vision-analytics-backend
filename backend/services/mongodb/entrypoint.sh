#!/bin/bash
set -e

HOSTNAME=${MONGO_RS_HOST:-mongodb}
REPL_SET_NAME=${REPL_SET_NAME:-rs0}
MONGO_PORT=${MONGO_PORT:-27017}
ADMIN_USER=${MONGO_INITDB_ROOT_USERNAME:-admin}
ADMIN_PASS=${MONGO_INITDB_ROOT_PASSWORD:-admin123}
KEYFILE=/etc/mongo/keyfile

mkdir -p /data/db
chown -R mongodb:mongodb /data/db

if [ ! -f /data/db/.initialized ]; then
    echo "[Mongo] First startup - initializing replica set and admin user"

    mongod \
      --replSet "$REPL_SET_NAME" \
      --bind_ip_all \
      --port "$MONGO_PORT" \
      --dbpath /data/db \
      --logpath /var/log/mongodb.log \
      --logappend &

    MONGO_PID=$!

    until mongosh --quiet --eval "db.runCommand({ ping: 1 })"; do
        sleep 1
    done

    echo "[Mongo] Initiating replica set"
    mongosh --eval "
        rs.initiate({
            _id: '$REPL_SET_NAME',
            members: [
            { _id: 0, host:'mongodb:27017'},
            ]
        })"


    sleep 3

    echo "[Mongo] Creating admin user"
    mongosh --eval "
      db.getSiblingDB('admin').createUser({
        user: '$ADMIN_USER',
        pwd: '$ADMIN_PASS',
        roles: [{ role: 'root', db: 'admin' }]
      })
    "

    echo "[Mongo] Shutting down temporary mongod"
    set +e
    mongosh --eval "db.shutdownServer()"
    set -e

    wait $MONGO_PID || true
    touch /data/db/.initialized
fi

echo "[Mongo] Starting MongoDB with authentication enabled"

exec mongod \
  --replSet "$REPL_SET_NAME" \
  --auth \
  --keyFile "$KEYFILE" \
  --bind_ip_all \
  --port "$MONGO_PORT" \
  --dbpath /data/db
