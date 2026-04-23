#!/bin/bash

# Prevent multiple instances from running simultaneously
LOCK_FILE="/tmp/check_wifi.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    exit 0  # Another instance is already running, exit silently
fi

# Configuration
CONNECTION_NAMES=("McD_Tolichowki")
INTERFACE_NAME="wlp2s0"
CHECK_HOST="8.8.8.8"
LOG_FILE="/home/xelomoc/Desktop/McD_backend-1/wifi_check.log"

# Check connection by pinging 3 times through the specific interface.
if ! ping -I "$INTERFACE_NAME" -c 3 -W 5 "$CHECK_HOST" > /dev/null 2>&1; then
    echo "$(date): Network down (3 pings failed). Attempting to reconnect..." >> "$LOG_FILE"
    
    # attempt to reconnect
    for conn in "${CONNECTION_NAMES[@]}"; do
        echo "$(date): Trying connection: $conn" >> "$LOG_FILE"
        if output=$(nmcli connection up "$conn" 2>&1); then
            echo "$(date): Successfully connected to $conn." >> "$LOG_FILE"
            break # Stop trying once a connection is successful
        else
            echo "$(date): Failed to connect to $conn. Error: $output" >> "$LOG_FILE"
        fi
    done
else
    # Optional: Log success for debugging, but might fill disk if every minute.
    # echo "$(date): Network is up." >> "$LOG_FILE"
    :
fi
