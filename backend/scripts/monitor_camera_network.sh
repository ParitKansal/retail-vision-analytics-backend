#!/bin/bash

# Camera Network Health Monitor
# This script checks if the camera network is accessible and fixes it if needed
# Designed to run via cron every hour

# Configuration
CAMERA_INTERFACE="enx00e04c411b80"
CAMERA_IP="192.168.10.200/24"
TEST_CAMERA="192.168.10.13"
LOG_FILE="/var/log/camera_network_monitor.log"

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to check if interface has IP
check_interface_ip() {
    ip addr show "$CAMERA_INTERFACE" 2>/dev/null | grep -q "inet 192.168.10"
    return $?
}

# Function to check if cameras are reachable
check_camera_reachable() {
    ping -c 1 -W 2 "$TEST_CAMERA" > /dev/null 2>&1
    return $?
}

# Main health check
log_message "=== Camera Network Health Check ==="

# Check 1: Does interface have IP?
if check_interface_ip; then
    log_message "✓ Interface $CAMERA_INTERFACE has IP assigned"
    
    # Check 2: Are cameras reachable?
    if check_camera_reachable; then
        log_message "✓ Camera network is healthy (can reach $TEST_CAMERA)"
        log_message "=== No action needed ==="
        exit 0
    else
        log_message "⚠ Interface has IP but cameras unreachable"
        log_message "→ This might be a camera/network issue, not a connection issue"
        log_message "=== No action taken ==="
        exit 0
    fi
else
    log_message "✗ Interface $CAMERA_INTERFACE has NO IP assigned"
    log_message "→ Connection appears to be down, attempting to fix..."
    
    # Attempt to bring up the connection
    if sudo nmcli con up "$CAMERA_INTERFACE" >> "$LOG_FILE" 2>&1; then
        log_message "✓ Successfully activated $CAMERA_INTERFACE"
        
        # Wait a moment for network to stabilize
        sleep 2
        
        # Verify the fix worked
        if check_interface_ip; then
            log_message "✓ Interface now has IP assigned"
            
            if check_camera_reachable; then
                log_message "✓ Cameras are now reachable"
                log_message "=== Problem fixed successfully ==="
                exit 0
            else
                log_message "⚠ Interface has IP but cameras still unreachable"
                log_message "=== Partial fix - may need manual intervention ==="
                exit 1
            fi
        else
            log_message "✗ Failed to assign IP to interface"
            log_message "=== Fix failed - manual intervention needed ==="
            exit 1
        fi
    else
        log_message "✗ Failed to activate connection"
        log_message "=== Fix failed - manual intervention needed ==="
        exit 1
    fi
fi
