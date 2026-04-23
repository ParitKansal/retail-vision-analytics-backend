#!/bin/bash
# Redis Service-Level Lag Checker
# This script lists all camera streams and the lag for each associated consumer group.

printf "%-25s | %-12s | %-35s | %-10s | %-10s\n" "Stream" "Total Frames" "Consumer Group" "Lag" "Pending"
printf "%s\n" "--------------------------------------------------------------------------------------------------------------"

streams=$(docker exec redis redis-cli KEYS "stream:cam:*" | tr -d '\r' | sort)

for stream in $streams; do
    # Get total frames in stream
    total=$(docker exec redis redis-cli XLEN "$stream" | tr -d '\r')
    
    # Get consumer groups for this stream
    docker exec redis redis-cli --raw XINFO GROUPS "$stream" 2>/dev/null | awk -v st="$stream" -v tot="$total" '
    BEGIN { name=""; lag="0"; pending="0" }
    {
        if ($0 == "name") {
            if (name != "") printf "%-25s | %-12s | %-35s | %-10s | %-10s\n", st, tot, name, lag, pending
            getline; name=$0; lag="0"; pending="0"
        } else if ($0 == "lag") {
            getline; lag=$0;
        } else if ($0 == "pending") {
            getline; pending=$0;
        }
    }
    END { if (name != "") printf "%-25s | %-12s | %-35s | %-10s | %-10s\n", st, tot, name, lag, pending }'
done
