#!/bin/bash

# Camera Network Fix Script
# This script activates the camera network interface and ensures WiFi is prioritized for internet

set -e

echo "=========================================="
echo "Camera Network Configuration Script"
echo "=========================================="
echo ""

# Configuration
CAMERA_INTERFACE="enx00e04c411b80"
CAMERA_IP="192.168.10.200/24"
WIFI_CONNECTION="ACTFIBERNET_5G"
WIRED_CONNECTION_1="Wired connection 1"

echo "Step 1: Checking current network status..."
echo ""
echo "Current routing table:"
ip route
echo ""

echo "Step 2: Configuring camera network interface..."
echo "Setting static IP: $CAMERA_IP on $CAMERA_INTERFACE"
sudo nmcli con mod "$CAMERA_INTERFACE" ipv4.addresses "$CAMERA_IP"
sudo nmcli con mod "$CAMERA_INTERFACE" ipv4.method manual
sudo nmcli con mod "$CAMERA_INTERFACE" ipv4.never-default yes
sudo nmcli con mod "$CAMERA_INTERFACE" ipv6.method ignore

echo ""
echo "Step 3: Ensuring other wired connections don't become default gateway..."
sudo nmcli con mod "$WIRED_CONNECTION_1" ipv4.never-default yes 2>/dev/null || echo "Wired connection 1 not found, skipping..."

echo ""
echo "Step 4: Ensuring WiFi can be default gateway..."
sudo nmcli con mod "$WIFI_CONNECTION" ipv4.never-default no

echo ""
echo "Step 5: Activating camera network interface..."
sudo nmcli con down "$CAMERA_INTERFACE" 2>/dev/null || true
sudo nmcli con up "$CAMERA_INTERFACE"

echo ""
echo "Step 6: Verifying configuration..."
echo ""
echo "Camera interface IP:"
ip addr show "$CAMERA_INTERFACE" | grep "inet "
echo ""
echo "Updated routing table:"
ip route
echo ""

echo "Step 7: Testing camera connectivity..."
TEST_CAMERA_IP="192.168.10.13"
if ping -c 2 -W 2 "$TEST_CAMERA_IP" > /dev/null 2>&1; then
    echo "✓ SUCCESS: Can reach camera at $TEST_CAMERA_IP"
else
    echo "✗ WARNING: Cannot reach camera at $TEST_CAMERA_IP"
    echo "  This may be normal if the camera is offline or at a different IP"
fi

echo ""
echo "Step 8: Testing internet connectivity..."
if ping -c 2 -W 2 8.8.8.8 > /dev/null 2>&1; then
    echo "✓ SUCCESS: Internet is accessible"
    echo ""
    echo "Internet route:"
    ip route get 8.8.8.8
else
    echo "✗ WARNING: Cannot reach internet"
fi

echo ""
echo "=========================================="
echo "Configuration Complete!"
echo "=========================================="
echo ""
echo "Summary:"
echo "- WiFi ($WIFI_CONNECTION) is used for Internet"
echo "- Wired LAN ($CAMERA_INTERFACE) is used for cameras (192.168.10.x)"
echo ""
echo "You can now restart your Docker services to connect to cameras."
