# Complete Network Configuration & Monitoring Documentation

**Last Updated:** 2026-02-11 15:12 IST  
**Status:** FULLY OPERATIONAL ✅

---

## 📑 Table of Contents

### Part 1: Network Priority Configuration
1. [Overview & Problem Statement](#part-1-network-priority-configuration)
2. [Solution Overview](#solution-overview)
3. [Step-by-Step Configuration](#step-by-step-configuration)
4. [Current System Configuration](#current-system-configuration)
5. [Troubleshooting Network Priority](#troubleshooting-network-priority)
6. [Quick Reference Commands](#quick-reference-commands-network)

### Part 2: What Happened & Root Cause
7. [Timeline of Events](#timeline-of-events)
8. [Root Cause Analysis](#root-cause-analysis)
9. [Why It Broke](#why-it-broke)
10. [What the Fix Did](#what-the-fix-did)
11. [Will This Happen Again?](#will-this-happen-again)

### Part 3: Do I Need to Run Daily?
12. [Quick Answer](#quick-answer-do-i-need-to-run-daily)
13. [Why Auto-Connect is Enabled](#why-auto-connect-is-enabled)
14. [Configuration Persistence](#configuration-persistence)
15. [When to Re-Run](#when-to-re-run)
16. [Health Check Procedures](#health-check-procedures)

### Part 4: Automatic Monitoring System
17. [Monitoring System Overview](#monitoring-system-overview)
18. [How Monitoring Works](#how-monitoring-works)
19. [Files Created for Monitoring](#files-created-for-monitoring)
20. [Monitoring Verification](#monitoring-verification)
21. [Log Examples](#log-examples)
22. [Protection Coverage](#protection-coverage)
23. [Cable Unplug/Replug Scenario](#cable-unplug-replug-scenario)

### Part 5: Management & Maintenance
24. [Management Commands](#management-commands)
25. [Expected Behavior](#expected-behavior)
26. [Why Hourly is Better](#why-hourly-is-better)
27. [Troubleshooting Guide](#troubleshooting-guide)
28. [Maintenance Tasks](#maintenance-tasks)

### Part 6: Summary & Reference
29. [Complete Summary](#complete-summary)
30. [All Important Files](#all-important-files)
31. [Configuration Details](#configuration-details)
32. [Quick Reference Card](#quick-reference-card)

---

# Part 1: Network Priority Configuration

## Overview & Problem Statement

### The Problem
When both WiFi and LAN are connected, Linux by default often prioritizes the Ethernet (LAN) connection for all traffic, including the Internet. If the LAN connection does not have internet access or is very slow, this breaks services that need to sync data to the cloud (like `db_sync_service`).

### The Goal
Configure the system to **prioritize WiFi for Internet** (sending data to MongoDB) while simultaneously using **LAN (Ethernet) for RTSP Camera streams**.

---

## Solution Overview

Configure **ALL wired/ethernet connections** to **never** be used as the default gateway. This forces the system to use the WiFi connection for all Internet traffic, while still routing local traffic (e.g., `192.168.10.x` for cameras) through the LAN interface.

**Key Principle:** WiFi = Internet | Wired LAN = Cameras Only

---

## Step-by-Step Configuration

### 1. Identify Network Interfaces
Run the following to list all connections:
```bash
nmcli con show
```

**Look for:**
- **WiFi connection** (e.g., `ACTFIBERNET_5G` on device `wlp2s0`)
- **Wired/Ethernet connections** (e.g., `enx00e04c411b80`, `Wired connection 1`, etc.)

### 2. Configure WiFi Connection (Ensure it CAN be Default)
Verify that your WiFi connection is allowed to be the default gateway:

```bash
nmcli con show "ACTFIBERNET_5G" | grep ipv4.never-default
```

**Expected:** `ipv4.never-default: no` (or blank)

If it shows `yes`, fix it:
```bash
sudo nmcli con mod "ACTFIBERNET_5G" ipv4.never-default no
sudo nmcli con down "ACTFIBERNET_5G"
sudo nmcli con up "ACTFIBERNET_5G"
```
*(Replace "ACTFIBERNET_5G" with your actual WiFi connection name)*

### 3. Configure ALL Wired Connections to NEVER be Default Gateway

For **each** wired connection, run these commands:

#### For Camera Network Connection (e.g., `enx00e04c411b80`):
```bash
# Set static IP for camera network
sudo nmcli con mod "enx00e04c411b80" ipv4.addresses 192.168.10.200/24
sudo nmcli con mod "enx00e04c411b80" ipv4.method manual

# CRITICAL: Prevent this from becoming default gateway
sudo nmcli con mod "enx00e04c411b80" ipv4.never-default yes

# Disable IPv6 to prevent delays
sudo nmcli con mod "enx00e04c411b80" ipv6.method ignore

# Apply changes
sudo nmcli con down "enx00e04c411b80"
sudo nmcli con up "enx00e04c411b80"
```

#### For Other Wired Connections (e.g., `Wired connection 1`):
```bash
# Prevent from becoming default gateway
sudo nmcli con mod "Wired connection 1" ipv4.never-default yes

# Apply changes
sudo nmcli con down "Wired connection 1"
sudo nmcli con up "Wired connection 1"
```

**Important:** Replace connection names with your actual connection names from step 1.

### 4. Verify Configuration

#### Check that wired connections have `never-default` set:
```bash
nmcli con show enx00e04c411b80 | grep ipv4.never-default
nmcli con show "Wired connection 1" | grep ipv4.never-default
```
**Expected:** Both should show `ipv4.never-default: yes`

#### Check the routing table:
```bash
ip route
```

**Expected Output Example:**
```
default via 192.168.0.1 dev wlp2s0 ...  <- Internet goes through WiFi ✓
192.168.10.0/24 dev enx00e04c411b80 ... <- Cameras use LAN ✓
192.168.0.0/24 dev wlp2s0 ...           <- WiFi local network
```

**Key Check:** The `default` route MUST point to your WiFi device (e.g., `wlp2s0`), NOT any ethernet device.

#### Test Internet connectivity through WiFi:
```bash
# Check which interface is used for internet
ip route get 8.8.8.8
```
**Expected:** Should show `dev wlp2s0` (your WiFi interface)

#### Test camera network connectivity:
```bash
ping -c 3 192.168.10.13
```
**Expected:** Should get replies from camera

---

## Current System Configuration

Based on the current setup:
- **WiFi**: `ACTFIBERNET_5G` on `wlp2s0` (192.168.0.118) - Used for Internet ✓
- **Camera LAN**: `enx00e04c411b80` (192.168.10.200/24) - Used for cameras only ✓
- **Other Wired**: `Wired connection 1` - Configured to never be default ✓

### Current Routing Table
```
default via 192.168.0.1 dev wlp2s0              <- Internet via WiFi ✓
192.168.0.0/24 dev wlp2s0                       <- WiFi local network
192.168.10.0/24 dev enx00e04c411b80             <- Camera network ✓
```

### Camera Network Connectivity
All camera IPs are now **reachable**:
- ✅ 192.168.10.12 - Reachable
- ✅ 192.168.10.13 - Reachable (Camera 7)
- ✅ 192.168.10.14 - Reachable
- ✅ 192.168.10.15 - Reachable
- ✅ 192.168.10.21 - Reachable

---

## Troubleshooting Network Priority

### Problem: Internet is slow or not working
**Check:**
```bash
ip route | grep default
```
- If default route shows an ethernet device (e.g., `enp...` or `enx...`), the wired connection is stealing internet traffic
- **Fix:** Apply step 3 to set `ipv4.never-default yes` on that connection

### Problem: Cannot access cameras
**Check:**
```bash
ip addr show enx00e04c411b80
ping 192.168.10.13
```
- If no IP is assigned, the connection is down
- **Fix:** `sudo nmcli con up enx00e04c411b80`
- See troubleshooting guide below for detailed camera network diagnostics

### Problem: Both connections keep switching priority
**Root Cause:** Metric values determine priority when both connections try to be default
**Fix:** Ensure `ipv4.never-default yes` is set on ALL wired connections (this removes them from default gateway consideration entirely)

---

## Quick Reference Commands (Network)

```bash
# List all connections
nmcli con show

# Check routing table
ip route

# Check specific connection settings
nmcli con show "connection-name" | grep ipv4

# Set wired connection to never be default
sudo nmcli con mod "connection-name" ipv4.never-default yes

# Restart a connection
sudo nmcli con down "connection-name" && sudo nmcli con up "connection-name"

# Test internet routing
ip route get 8.8.8.8

# Test camera connectivity
ping 192.168.10.13
```

---

# Part 2: What Happened & Root Cause

## Timeline of Events

### 1. Earlier (When it was working):
```
┌─────────────────────────────────────────────────┐
│  Your System                                    │
│                                                 │
│  WiFi (wlp2s0) ────────────► Internet ✓        │
│                                                 │
│  Wired LAN (enx00e04c411b80)                   │
│  IP: 192.168.10.200 ────────► Cameras ✓        │
│                                                 │
│  Stream Service ────────────► Cameras ✓        │
└─────────────────────────────────────────────────┘
```
- The camera network interface `enx00e04c411b80` was **active** with IP 192.168.10.200
- Cameras were accessible
- Everything worked fine

### 2. Something Changed (Recently):
```
┌─────────────────────────────────────────────────┐
│  Your System                                    │
│                                                 │
│  WiFi (wlp2s0) ────────────► Internet ✓        │
│                                                 │
│  Wired LAN (enx00e04c411b80)                   │
│  IP: NONE ❌ (Connection inactive)              │
│                                                 │
│  Stream Service ─────X─────► Cameras ❌        │
│                    (timeout)                    │
└─────────────────────────────────────────────────┘
```
- The wired network connection became **inactive** (lost its IP address)
- Possible causes:
  - System reboot without auto-connect enabled
  - Network cable was unplugged/replugged
  - NetworkManager reset or connection was manually stopped
  - System update that reset network settings

### 3. Result:
- The interface was physically UP (cable connected)
- But it had **NO IP address assigned**
- Docker containers couldn't reach cameras on 192.168.10.x network
- Stream timeouts every 30 seconds

---

## Root Cause Analysis

### What We Found

```bash
# Before the fix:
$ ip addr show enx00e04c411b80
217: enx00e04c411b80: <BROADCAST,MULTICAST,UP,LOWER_UP> ...
    link/ether 00:e0:4c:41:1b:80 ...
    # ❌ NO "inet 192.168.10.200/24" line!

# After the fix:
$ ip addr show enx00e04c411b80
217: enx00e04c411b80: <BROADCAST,MULTICAST,UP,LOWER_UP> ...
    link/ether 00:e0:4c:41:1b:80 ...
    inet 192.168.10.200/24 ...  # ✅ IP is assigned!
```

### The Issue
The connection was **configured** correctly in NetworkManager (it had all the right settings saved), but it was simply **not activated** (not running).

**Analogy:** It's like having a car with a full tank of gas and the keys in the ignition, but the engine isn't running. The car is ready to go, it just needs to be started.

---

## Why It Broke

Likely causes:
1. **System reboot** - Connection didn't auto-start
2. **Cable unplugged/replugged** - Connection didn't auto-reconnect
3. **Manual disconnection** - Someone ran a disconnect command
4. **Network service restart** - Connection didn't come back up

---

## What the Fix Did

### The Script Did Two Things:

1. **Activated the connection** (turned it on)
   ```bash
   sudo nmcli con up enx00e04c411b80
   ```
   This is like turning the key to start the car.

2. **Ensured proper priority settings** (WiFi for internet, LAN for cameras)
   ```bash
   sudo nmcli con mod enx00e04c411b80 ipv4.never-default yes
   sudo nmcli con mod ACTFIBERNET_5G ipv4.never-default no
   ```
   This ensures the right "roads" are used for the right traffic.

### After Our Fix (Now):
```
┌─────────────────────────────────────────────────┐
│  Your System                                    │
│                                                 │
│  WiFi (wlp2s0) ────────────► Internet ✓        │
│  (Priority: Default Gateway)                    │
│                                                 │
│  Wired LAN (enx00e04c411b80)                   │
│  IP: 192.168.10.200 ────────► Cameras ✓        │
│  (Auto-connect: YES)                            │
│  (Never-default: YES)                           │
│                                                 │
│  Stream Service ────────────► Cameras ✓        │
└─────────────────────────────────────────────────┘
```

---

## Will This Happen Again?

### Short Answer: **Probably Not**

Here's why:

### 1. **Auto-Connect is Enabled**
NetworkManager connections have an `autoconnect` property:

```bash
$ nmcli con show enx00e04c411b80 | grep autoconnect
connection.autoconnect:          yes
connection.autoconnect-retries:  -1 (unlimited)
```

This means the connection will **automatically activate** on:
- System boot
- Cable plug-in
- Network service restart

### 2. **Configuration is Persistent**
All settings are saved in `/etc/NetworkManager/system-connections/` and survive reboots.

### 3. **Priority Settings are Locked In**
The `ipv4.never-default yes` setting ensures the wired connection will never try to become the internet gateway, even if it activates.

---

# Part 3: Do I Need to Run Daily?

## Quick Answer: Do I Need to Run Daily?

## 🎯 **NO - This is a ONE-TIME FIX**

---

## Why Auto-Connect is Enabled

```bash
$ nmcli con show enx00e04c411b80 | grep autoconnect
connection.autoconnect:                 yes  ← This means automatic!
connection.autoconnect-retries:         -1   ← Unlimited retries!
```

**What this means:**
- ✅ Activates automatically on **system boot**
- ✅ Activates automatically when **cable is plugged in**
- ✅ Activates automatically after **network service restart**
- ✅ Retries **unlimited times** if it fails

---

## Configuration Persistence

All settings are stored in:
```
/etc/NetworkManager/system-connections/enx00e04c411b80.nmconnection
```

This file survives:
- ✅ Reboots
- ✅ Power outages
- ✅ System updates (usually)

### Priority Settings are Locked

```bash
# Wired LAN will NEVER become default gateway
ipv4.never-default: yes

# WiFi CAN be default gateway
ipv4.never-default: no
```

This prevents the "priority battle" between WiFi and wired.

---

## When to Re-Run

### Very Rare Scenarios:

| Scenario | Likelihood | Solution |
|----------|-----------|----------|
| System reinstall/OS upgrade | Rare | Run script once |
| Someone manually disables connection | Very rare | Run script once |
| Network adapter replacement | Very rare | Run script once |
| NetworkManager settings reset | Extremely rare | Run script once |

**Normal operations (reboots, cable unplugs, etc.):** ❌ **NO script needed**

---

## Health Check Procedures

### One-Line Health Check:

```bash
ip addr show enx00e04c411b80 | grep "inet 192.168.10"
```

**If you see output:** ✅ Everything is working, cameras are accessible

**If you see nothing:** ❌ Connection is down, run the fix script

### Complete Health Check:

```bash
# 1. Check if camera network is active
ip addr show enx00e04c411b80 | grep "inet 192.168.10"

# Expected: inet 192.168.10.200/24 ...
# If you see this line, cameras are accessible ✓

# 2. Check if internet goes through WiFi
ip route | grep default

# Expected: default via X.X.X.X dev wlp2s0 ...
# If you see wlp2s0 (WiFi), internet priority is correct ✓

# 3. Test camera connectivity
ping -c 2 192.168.10.13

# Expected: 2 packets transmitted, 2 received
# If you get replies, cameras are reachable ✓
```

### If Any Check Fails:

Just run the fix script once:
```bash
cd /home/xelomoc/Desktop/McD_backend-1/backend
./scripts/fix_camera_network.sh
```

---

## Comparison: Before vs After

### BEFORE the fix:
```
┌──────────────────────────────────────┐
│ Configuration: ✓ Correct             │
│ Connection State: ❌ Inactive         │
│ Auto-connect: ? (maybe not set)      │
│ Result: Cameras unreachable          │
└──────────────────────────────────────┘
```

### AFTER the fix:
```
┌──────────────────────────────────────┐
│ Configuration: ✓ Correct             │
│ Connection State: ✓ Active           │
│ Auto-connect: ✓ YES (guaranteed)     │
│ Result: Cameras accessible           │
│ Survives: Reboots, reconnects        │
└──────────────────────────────────────┘
```

---

## Real-World Analogy

### Think of it like a light switch:

**Before:**
- The light bulb was installed ✓
- The wiring was correct ✓
- But the switch was OFF ❌

**What we did:**
- Turned the switch ON ✓
- Set it to "auto-on" when power is available ✓

**Now:**
- Light turns on automatically when power is available
- No need to manually flip the switch daily

---

## Bottom Line

### ❌ You DO NOT need to:
- Run the script daily
- Run the script weekly
- Run the script on every reboot
- Add it to cron jobs (we did this for extra safety, but it's not required)
- Worry about it

### ✅ The system will:
- Auto-connect on boot
- Auto-reconnect if cable is unplugged/replugged
- Maintain WiFi priority for internet
- Keep cameras accessible 24/7

### 🔧 Only re-run the script if:
- You see camera timeout errors again
- The health check fails (no IP on interface)
- After major system changes (OS reinstall, etc.)

---

## Summary in One Sentence

**The connection was sleeping, we woke it up and told it to stay awake forever - no daily alarm needed.** ⏰✅

---

# Part 4: Automatic Monitoring System

## Monitoring System Overview

An **automatic monitoring system** that checks the camera network health **every hour** and auto-fixes any connection issues.

### ✅ STATUS: ACTIVE

**Monitoring:** Every hour at minute 0  
**Auto-fix:** Enabled  
**Logs:** `/var/log/camera_network_monitor.log`

### What You Asked For

> "Can u create a script that run automatically every 1 hr bcoz it is possible somebody removed connect and then again plugged it?"

### ✅ What You Got

**An automatic monitoring system that:**
1. ✅ Checks camera network health **every hour**
2. ✅ Auto-fixes connection if it's down
3. ✅ Logs all checks and fixes
4. ✅ Handles cable unplug/replug scenarios
5. ✅ Runs automatically in the background

---

## How Monitoring Works

### Protection Layers

```
┌─────────────────────────────────────────────────────────┐
│                    PROTECTION LAYERS                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Layer 1: Auto-Connect (Instant)                        │
│  ├─ On system boot                                      │
│  ├─ On cable replug                                     │
│  └─ On network service restart                          │
│                                                          │
│  Layer 2: Hourly Monitor (Within 60 min)               │
│  ├─ Checks: Interface has IP?                           │
│  ├─ Checks: Cameras reachable?                          │
│  ├─ Auto-fix: If connection down                        │
│  └─ Logs: All actions                                   │
│                                                          │
│  Layer 3: Manual Scripts (On-demand)                    │
│  ├─ fix_camera_network.sh                               │
│  └─ monitor_camera_network.sh                           │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Every Hour (at minute 0):

```
┌─────────────────────────────────────────────────┐
│  Cron Job Runs                                  │
│  ↓                                              │
│  Check: Does interface have IP?                 │
│  ├─ YES → Check: Can ping cameras?             │
│  │   ├─ YES → ✓ All good, do nothing          │
│  │   └─ NO → ⚠ Log warning (camera issue)     │
│  └─ NO → ✗ Connection down!                    │
│      ↓                                          │
│      Run: sudo nmcli con up enx00e04c411b80    │
│      ↓                                          │
│      Verify: Did it fix the problem?            │
│      ├─ YES → ✓ Log success                    │
│      └─ NO → ✗ Log failure (needs manual fix) │
└─────────────────────────────────────────────────┘
```

### Schedule:
- **Runs:** Every hour at minute 0 (10:00, 11:00, 12:00, etc.)
- **Script:** `/home/xelomoc/Desktop/McD_backend-1/backend/scripts/monitor_camera_network.sh`
- **Logs:** `/var/log/camera_network_monitor.log`

---

## Files Created for Monitoring

### 1. **`scripts/monitor_camera_network.sh`**
The monitoring script that does the health check and auto-fix.

**What it checks:**
- ✓ Does `enx00e04c411b80` have IP `192.168.10.200`?
- ✓ Can we ping camera at `192.168.10.13`?

**What it does if broken:**
- Runs `sudo nmcli con up enx00e04c411b80`
- Verifies the fix worked
- Logs everything

### 2. **`scripts/setup_camera_monitor_cron.sh`**
One-time setup script to install the cron job.

**Already executed** - cron job is now active!

### 3. **`scripts/fix_camera_network.sh`**
Full network configuration script (for manual use if needed).

**What it does:**
- Configures camera interface with static IP
- Ensures WiFi priority for internet
- Activates camera network
- Tests connectivity to both cameras and internet

---

## Monitoring Verification

### Check if cron job is installed:
```bash
crontab -l
```

**Expected output:**
```
0 * * * * /home/xelomoc/Desktop/McD_backend-1/backend/scripts/monitor_camera_network.sh
```
✅ **Confirmed installed!**

### Check if monitoring is active:
```bash
crontab -l | grep monitor_camera_network
```

### View monitoring logs:
```bash
sudo tail -f /var/log/camera_network_monitor.log
```

**Current log shows:**
```
[2026-02-11 15:01:44] === Camera Network Health Check ===
[2026-02-11 15:01:44] ✓ Interface enx00e04c411b80 has IP assigned
[2026-02-11 15:01:44] ✓ Camera network is healthy (can reach 192.168.10.13)
[2026-02-11 15:01:44] === No action needed ===
```
✅ **Working perfectly!**

---

## Log Examples

### When everything is healthy:
```
[2026-02-11 16:00:00] === Camera Network Health Check ===
[2026-02-11 16:00:00] ✓ Interface enx00e04c411b80 has IP assigned
[2026-02-11 16:00:00] ✓ Camera network is healthy (can reach 192.168.10.13)
[2026-02-11 16:00:00] === No action needed ===
```

### When connection is down and gets auto-fixed:
```
[2026-02-11 16:00:00] === Camera Network Health Check ===
[2026-02-11 16:00:00] ✗ Interface enx00e04c411b80 has NO IP assigned
[2026-02-11 16:00:00] → Connection appears to be down, attempting to fix...
[2026-02-11 16:00:02] ✓ Successfully activated enx00e04c411b80
[2026-02-11 16:00:04] ✓ Interface now has IP assigned
[2026-02-11 16:00:04] ✓ Cameras are now reachable
[2026-02-11 16:00:04] === Problem fixed successfully ===
```

### When there's a problem that can't be auto-fixed:
```
[2026-02-11 16:00:00] === Camera Network Health Check ===
[2026-02-11 16:00:00] ✗ Interface enx00e04c411b80 has NO IP assigned
[2026-02-11 16:00:00] → Connection appears to be down, attempting to fix...
[2026-02-11 16:00:02] ✗ Failed to activate connection
[2026-02-11 16:00:02] === Fix failed - manual intervention needed ===
```

---

## Protection Coverage

### ✅ Scenarios that will be auto-fixed:
1. **Cable unplugged and replugged** - Fixed within 1 hour
2. **Connection manually stopped** - Fixed within 1 hour
3. **Network service restart** - Fixed within 1 hour
4. **Temporary network glitch** - Fixed within 1 hour
5. **Interface lost IP for any reason** - Fixed within 1 hour

### ⚠️ Scenarios that need manual intervention:
1. **Hardware failure** (network adapter broken)
2. **Cable physically damaged**
3. **NetworkManager service not running**
4. **Permissions issue** (sudo not working)

### Protection Summary:

✅ **Auto-connect on boot** - Instant  
✅ **Auto-reconnect on cable replug** - Instant  
✅ **Hourly health check** - Within 60 minutes  
✅ **Auto-fix if down** - Within 60 minutes  
✅ **Full logging** - All events recorded  

**Maximum downtime:** 59 minutes (if issue occurs right after a check)

---

## Cable Unplug/Replug Scenario

### Scenario: Someone unplugs the camera network cable

**Timeline:**

| Time | Event | System Response |
|------|-------|-----------------|
| 10:30 | Cable unplugged | Cameras become unreachable immediately |
| 10:30 | Auto-connect tries | May reconnect instantly if cable replugged quickly |
| 11:00 | Hourly monitor runs | Detects no IP, runs auto-fix, connection restored |
| 11:00 | Log entry | "Problem fixed successfully" |

**Maximum downtime:** 59 minutes (if unplugged right after 10:00 check)  
**Average downtime:** ~30 minutes  
**Best case:** Instant (auto-connect catches it)

---

# Part 5: Management & Maintenance

## Management Commands

### View current cron jobs:
```bash
crontab -l
```

### View monitoring logs (live):
```bash
sudo tail -f /var/log/camera_network_monitor.log
```

### View last 50 log entries:
```bash
sudo tail -50 /var/log/camera_network_monitor.log
```

### View recent logs:
```bash
sudo tail -20 /var/log/camera_network_monitor.log
```

### Manually run the monitor (test):
```bash
sudo /home/xelomoc/Desktop/McD_backend-1/backend/scripts/monitor_camera_network.sh
```

### Check camera network status:
```bash
ip addr show enx00e04c411b80 | grep "inet 192.168.10"
```

### Ping a camera:
```bash
ping -c 2 192.168.10.13
```

### Check routing table:
```bash
ip route
```

### Check which interface is used for internet:
```bash
ip route get 8.8.8.8
```

### Remove the cron job (if needed):
```bash
crontab -e
# Delete the line with 'monitor_camera_network.sh'
# Save and exit
```

### Re-install the cron job:
```bash
cd /home/xelomoc/Desktop/McD_backend-1/backend
./scripts/setup_camera_monitor_cron.sh
```

### Run full network fix manually:
```bash
cd /home/xelomoc/Desktop/McD_backend-1/backend
./scripts/fix_camera_network.sh
```

---

## Expected Behavior

### Normal Operation:
- **Every hour:** Script runs, checks health, logs "No action needed"
- **Log file:** Grows slowly with hourly health checks
- **Your cameras:** Keep working without interruption

### If Cable is Unplugged:
1. **Immediately:** Cameras become unreachable
2. **Within 1 hour:** Monitor detects issue and auto-fixes
3. **Result:** Cameras working again, incident logged

### Maximum Downtime:
- **Worst case:** Up to 59 minutes (if cable unplugged right after a check)
- **Average case:** ~30 minutes
- **Best case:** Detected on next hourly check

---

## Why Hourly is Better

| Frequency | Detection Time | Protection Level |
|-----------|---------------|------------------|
| Daily | Up to 24 hours | Low |
| Every 6 hours | Up to 6 hours | Medium |
| **Every 1 hour** | **Up to 59 minutes** | **High** ✅ |
| Every 15 minutes | Up to 15 minutes | Very High (overkill) |

**1 hour is the sweet spot:**
- ✅ Fast enough to catch issues quickly
- ✅ Light enough to not waste resources
- ✅ Logs stay manageable

---

## Troubleshooting Guide

### If cameras are down:
1. **Wait up to 1 hour** - Monitor will auto-fix
2. **Or manually run:** `sudo ./scripts/monitor_camera_network.sh`
3. **Check logs:** `sudo tail /var/log/camera_network_monitor.log`

### If auto-fix fails:
1. **Check logs** for error messages
2. **Run full fix:** `./scripts/fix_camera_network.sh`
3. **Verify hardware:** Check cable connections

### Common Issues:

| Issue | Check | Solution |
|-------|-------|----------|
| Cameras unreachable | `ping 192.168.10.13` | Wait for hourly check or run monitor manually |
| No IP on interface | `ip addr show enx00e04c411b80` | Run `sudo nmcli con up enx00e04c411b80` |
| Cron not running | `crontab -l` | Re-run `./scripts/setup_camera_monitor_cron.sh` |
| Logs not updating | Check `/var/log/camera_network_monitor.log` | Verify cron job is active |
| Internet slow | `ip route \| grep default` | Ensure default route is through WiFi |
| Priority switching | `nmcli con show \| grep never-default` | Set `ipv4.never-default yes` on wired |

---

## Maintenance Tasks

### Log File Rotation (Optional):
The log file will grow over time. To prevent it from getting too large, you can set up log rotation:

```bash
# Create log rotation config (optional)
sudo tee /etc/logrotate.d/camera-network-monitor << EOF
/var/log/camera_network_monitor.log {
    weekly
    rotate 4
    compress
    missingok
    notifempty
}
EOF
```

This will:
- Keep last 4 weeks of logs
- Compress old logs
- Rotate weekly

### Regular Checks (Optional):
While the system is self-monitoring, you may want to:
- Check logs weekly: `sudo tail /var/log/camera_network_monitor.log`
- Verify cron job monthly: `crontab -l`
- Test manual run quarterly: `sudo ./scripts/monitor_camera_network.sh`

---

# Part 6: Summary & Reference

## Complete Summary

### Before:
- ❌ If cable unplugged: Manual intervention needed
- ❌ If connection down: Manual fix required
- ❌ No monitoring or logging
- ❌ WiFi and wired competing for priority

### After:
- ✅ If cable unplugged: Auto-fixes within 1 hour
- ✅ If connection down: Auto-fixes within 1 hour
- ✅ Full monitoring and logging
- ✅ Two layers of protection (auto-connect + hourly check)
- ✅ WiFi prioritized for internet
- ✅ Wired LAN dedicated to cameras
- ✅ Peace of mind

### What You Have Now:

1. **Primary Protection:** Auto-connect on boot/cable replug (instant)
2. **Secondary Protection:** Hourly health check and auto-fix (within 1 hour)
3. **Monitoring:** Full logging of all checks and fixes
4. **Peace of Mind:** System self-heals automatically
5. **Network Priority:** WiFi for internet, LAN for cameras
6. **Persistence:** Configuration survives reboots

### You're Protected Against:
- ✅ Cable unplugs/replugs
- ✅ Manual connection stops
- ✅ Network service restarts
- ✅ Temporary glitches
- ✅ System reboots
- ✅ WiFi/wired priority conflicts

---

## All Important Files

| File | Purpose |
|------|---------|
| `scripts/monitor_camera_network.sh` | Hourly health check script |
| `scripts/fix_camera_network.sh` | Full network configuration script |
| `scripts/setup_camera_monitor_cron.sh` | Cron job installer |
| `/var/log/camera_network_monitor.log` | Monitoring logs |
| `COMPLETE_NETWORK_DOCUMENTATION.md` | This complete guide |
| `CAMERA_NETWORK_MONITORING_COMPLETE_GUIDE.md` | Monitoring-specific guide |
| `NETWORK_SETUP.md` | Network routing configuration guide |
| `docs/Camera_Network_Troubleshooting.md` | Camera troubleshooting guide |

---

## Configuration Details

### Cron Job:
```
0 * * * * /home/xelomoc/Desktop/McD_backend-1/backend/scripts/monitor_camera_network.sh
```

### Network Configuration:
- **Camera Interface:** `enx00e04c411b80`
- **Camera IP:** `192.168.10.200/24`
- **Test Camera:** `192.168.10.13`
- **WiFi Interface:** `wlp2s0`
- **WiFi Connection:** `ACTFIBERNET_5G`
- **WiFi IP:** `192.168.0.118`

### Auto-Connect Settings:
```
connection.autoconnect: yes
connection.autoconnect-retries: -1 (unlimited)
ipv4.never-default: yes (on wired)
ipv4.never-default: no (on WiFi)
ipv4.method: manual (on camera interface)
ipv4.addresses: 192.168.10.200/24 (on camera interface)
ipv6.method: ignore (on camera interface)
```

### Routing Configuration:
```
default via 192.168.0.1 dev wlp2s0              <- Internet via WiFi
192.168.0.0/24 dev wlp2s0                       <- WiFi local network
192.168.10.0/24 dev enx00e04c411b80             <- Camera network
```

---

## Quick Reference Card

### System Status Check:
```bash
# One-line health check
ip addr show enx00e04c411b80 | grep "inet 192.168.10"
# If you see output: ✅ Everything is working
# If you see nothing: ❌ Connection is down
```

### Emergency Commands:
```bash
# Quick fix
sudo nmcli con up enx00e04c411b80

# Full fix
./scripts/fix_camera_network.sh

# Check logs
sudo tail /var/log/camera_network_monitor.log

# Test monitor
sudo ./scripts/monitor_camera_network.sh
```

### Verification Commands:
```bash
# Check cron job
crontab -l | grep monitor_camera_network

# Check routing
ip route | grep default

# Check camera connectivity
ping -c 2 192.168.10.13

# Check internet routing
ip route get 8.8.8.8

# Check auto-connect
nmcli con show enx00e04c411b80 | grep autoconnect
```

---

## Next Steps

### For You:
**Nothing!** The system is self-monitoring and self-healing.

### Optional:
- Check logs weekly: `sudo tail /var/log/camera_network_monitor.log`
- Set up log rotation (shown in maintenance section)
- Review this guide if issues occur

---

## Verification Checklist

**Cron Job:** ✅ Installed and active  
**Monitoring Script:** ✅ Tested and working  
**Camera Network:** ✅ Healthy (192.168.10.200 assigned)  
**Cameras:** ✅ Reachable (ping successful)  
**Logs:** ✅ Being written  
**WiFi Priority:** ✅ Internet via WiFi  
**Auto-Connect:** ✅ Enabled (unlimited retries)  
**Configuration:** ✅ Persistent across reboots  

**Everything is working perfectly!** 🎉

---

## Contact & Support

If you encounter issues:
1. Check this guide's troubleshooting section
2. Review logs: `sudo tail /var/log/camera_network_monitor.log`
3. Verify cron job: `crontab -l`
4. Run manual test: `sudo ./scripts/monitor_camera_network.sh`
5. Run full fix: `./scripts/fix_camera_network.sh`

**Remember:** The system is designed to self-heal. Most issues will be automatically resolved within 1 hour.

---

**Setup completed:** 2026-02-11 15:01 IST  
**Next automatic check:** Top of next hour  
**Status:** FULLY OPERATIONAL ✅

---

**End of Complete Network Documentation**
