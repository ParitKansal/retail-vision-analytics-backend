# Camera Network Troubleshooting Guide

**Purpose:** This guide enables any engineer or technician to diagnose and resolve camera connection issues on the backend system, applicable to any deployment environment.

---

## Quick Fix (Automated)

If you're seeing stream timeout errors in `docker logs stream_handling_service -f`, run this script to automatically configure the network:

```bash
cd /home/xelomoc/Desktop/McD_backend-1/backend
./scripts/fix_camera_network.sh
```

This script will:
1. Configure the camera network interface with static IP (192.168.10.200)
2. Ensure WiFi is prioritized for internet traffic
3. Activate the camera network connection
4. Test connectivity to cameras and internet

**After running the script, restart your Docker services:**
```bash
docker-compose restart stream_handling_service
```

If the automated fix doesn't work, continue with the manual diagnostic steps below.

---

## 1. Symptom Identification
If the `stream_handling_service` logs show the following errors:
- `[WARN] ... Stream timeout triggered`
- `[WARN] ... CAP_IMAGES: error, expected pattern ...`
- `Cannot open RTSP, retrying`

**And** you cannot `ping` the camera IPs (e.g., `192.168.10.x`) from the host terminal.

**Then:** The issue is likely a **Network Configuration Problem** on the host machine.

---

## 2. Diagnostic Steps (Root Cause Analysis)
Run these commands in the terminal to identify the cause:

### Step A: Identify the Interface
Check which network interface is physically connected to the cameras.
```bash
nmcli con show
```
*Look for the Ethernet connection (e.g., `Wired connection 1` or `enx...`).*

### Step B: Check IP Configuration
Check if the interface has an IP address assigned.
```bash
ip addr show <interface_name>
```
*Example:* `ip addr show enx00e04c411b80`

- **Bad State:** You see `state UP` but **NO `inet 192.168.x.x`** line.
  - *Cause:* Interface is set to DHCP (Auto) but there is no DHCP server on the camera network.
- **Good State:** You see `inet 192.168.10.200/24 ...`.

### Step C: Check Routing
Verify that traffic to `192.168.10.x` is going through the correct interface.
```bash
ip route
```
- **Bad State:** `192.168.10.0/24` is missing or going through the wrong interface (like WiFi `wlp...`).
- **Good State:** `192.168.10.0/24 dev <camera_interface_name> ...`

---

## 3. Resolution Procedure
If the interface is missing an IP (Step B) or routing is wrong (Step C):

### 1. Assign a Static IP
You must manually assign an IP address to the host interface that is **in the same subnet** as the cameras but **unique** (not used by any camera).

**Command Syntax:**
```bash
# 1. Set the static IP (Replace [IP] and [ConnectionName])
# Make sure [IP] is NOT used by any camera!
sudo nmcli con mod "[ConnectionName]" ipv4.addresses [IP]/24

# 2. Set method to Manual
sudo nmcli con mod "[ConnectionName]" ipv4.method manual

# 3. Disable IPv6 (prevents timeout delays)
sudo nmcli con mod "[ConnectionName]" ipv6.method ignore

# 4. Apply changes (Down then Up)
sudo nmcli con down "[ConnectionName]"
sudo nmcli con up "[ConnectionName]"
```

**Example (This System):**
```bash
sudo nmcli con mod enx00e04c411b80 ipv4.addresses 192.168.10.200/24
sudo nmcli con mod enx00e04c411b80 ipv4.method manual
sudo nmcli con mod enx00e04c411b80 ipv6.method ignore
sudo nmcli con up enx00e04c411b80
```

### 2. Verify Connectivity
After applying the fix, test connectivity immediately:
```bash
ping -c 3 <camera_ip_address>
```
*Example:* `ping -c 3 192.168.10.13`
- If you get replies, the network is fixed.
- If you get "Destination Host Unreachable", check the IP/Subnet again.

---

## 4. Common Pitfalls & Lessons Learned
| Issue | Symptom | Solution |
| :--- | :--- | :--- |
| **IP Conflict** | `NM` Error: "IP address ... already in use" | **Do NOT** use IPs ending in `.1`, `.100`, or `.101` without checking. Use a high number like `.200` or `.250`. |
| **IPv6 Timeout** | Connection takes 30s+ to activate | Disable IPv6 (`ipv6.method ignore`). The OS waits for a router that isn't there. |
| **DHCP Failure** | Interface stays "Connecting..." forever | Switch to **Manual** IP immediately for private camera networks. |
