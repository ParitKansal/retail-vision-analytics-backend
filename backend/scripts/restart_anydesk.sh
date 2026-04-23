#!/bin/bash

# restart_anydesk.sh
# Kills AnyDesk, waits 5 minutes, then restarts it.
# Designed to be run as a cron job every 1 hour.

LOG_FILE="/home/xelomoc/Desktop/McD_backend-1/anydesk_restart.log"
WAIT_SECONDS=300  # 5 minutes

echo "$(date): Stopping AnyDesk..." >> "$LOG_FILE"
pkill -x anydesk 2>/dev/null || pkill -f anydesk 2>/dev/null

sleep 2

# Confirm it's dead
if pgrep -x anydesk > /dev/null 2>&1; then
    echo "$(date): AnyDesk still running, force killing..." >> "$LOG_FILE"
    pkill -9 -x anydesk 2>/dev/null
fi

echo "$(date): AnyDesk stopped. Waiting ${WAIT_SECONDS} seconds (5 min)..." >> "$LOG_FILE"
sleep "$WAIT_SECONDS"

echo "$(date): Restarting AnyDesk..." >> "$LOG_FILE"

# Restart AnyDesk in background (detached from this shell)
nohup anydesk > /dev/null 2>&1 &

echo "$(date): AnyDesk restarted (PID: $!)." >> "$LOG_FILE"
