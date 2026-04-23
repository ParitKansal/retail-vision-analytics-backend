#!/bin/bash

# Setup script to install the camera network monitor as a cron job
# This will run the health check every hour

SCRIPT_DIR="/home/xelomoc/Desktop/McD_backend-1/backend/scripts"
MONITOR_SCRIPT="$SCRIPT_DIR/monitor_camera_network.sh"
CRON_SCHEDULE="0 * * * *"  # Every hour at minute 0

echo "=========================================="
echo "Camera Network Monitor - Cron Setup"
echo "=========================================="
echo ""

# Check if monitor script exists
if [ ! -f "$MONITOR_SCRIPT" ]; then
    echo "✗ Error: Monitor script not found at $MONITOR_SCRIPT"
    exit 1
fi

# Make sure script is executable
chmod +x "$MONITOR_SCRIPT"
echo "✓ Monitor script is executable"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "monitor_camera_network.sh"; then
    echo "⚠ Cron job already exists. Removing old entry..."
    crontab -l 2>/dev/null | grep -v "monitor_camera_network.sh" | crontab -
fi

# Add new cron job
echo "→ Adding cron job to run every hour..."
(crontab -l 2>/dev/null; echo "$CRON_SCHEDULE $MONITOR_SCRIPT") | crontab -

if [ $? -eq 0 ]; then
    echo "✓ Cron job installed successfully!"
    echo ""
    echo "Schedule: Every hour at minute 0 (e.g., 10:00, 11:00, 12:00, etc.)"
    echo "Script: $MONITOR_SCRIPT"
    echo "Log file: /var/log/camera_network_monitor.log"
    echo ""
    echo "Current crontab:"
    crontab -l | grep "monitor_camera_network.sh"
    echo ""
    echo "=========================================="
    echo "Setup Complete!"
    echo "=========================================="
    echo ""
    echo "The monitor will:"
    echo "  1. Check camera network health every hour"
    echo "  2. Auto-fix if connection is down"
    echo "  3. Log all actions to /var/log/camera_network_monitor.log"
    echo ""
    echo "To view logs:"
    echo "  sudo tail -f /var/log/camera_network_monitor.log"
    echo ""
    echo "To remove the cron job:"
    echo "  crontab -e"
    echo "  (then delete the line with 'monitor_camera_network.sh')"
    echo ""
else
    echo "✗ Failed to install cron job"
    exit 1
fi
