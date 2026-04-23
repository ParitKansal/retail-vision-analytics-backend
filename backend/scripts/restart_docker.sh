#!/bin/bash

LOG_FILE="/home/xelomoc/Desktop/McD_backend-1/docker_restart.log"

echo "$(date): Starting scheduled Docker Compose restart..." >> "$LOG_FILE"

# Go to the directory containing docker-compose.yml
cd /home/xelomoc/Desktop/McD_backend-1/backend || {
    echo "$(date): ERROR: Could not find backend directory." >> "$LOG_FILE"
    exit 1
}

# Bring down the containers and bring them back up
if docker compose down >> "$LOG_FILE" 2>&1 && docker compose up -d >> "$LOG_FILE" 2>&1; then
    echo "$(date): Successfully restarted all Docker services." >> "$LOG_FILE"
else
    echo "$(date): ERROR: Failed to restart Docker services." >> "$LOG_FILE"
fi
