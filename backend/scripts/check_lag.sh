#!/bin/bash
# Redis Stream Consumer Lag Checker
# Usage: ./check_lag.sh

echo "=========================================="
echo "Redis Stream Consumer Lag Report"
echo "=========================================="
echo ""

for stream in $(docker exec redis redis-cli KEYS "stream:cam:*" | tr -d '\r'); do
    if [ ! -z "$stream" ]; then
        camera=$(echo $stream | cut -d: -f3)
        total=$(docker exec redis redis-cli XLEN "$stream")
        
        echo "📹 Camera $camera (Total: $total messages)"
        
        # Get consumer groups for this stream
        docker exec redis redis-cli XINFO GROUPS "$stream" 2>/dev/null | awk '
            BEGIN { name="" }
            /^name$/ { getline; name=$0; gsub(/^cg:cam:/, "", name) }
            /^lag$/ { 
                getline; 
                lag=$0
                status = (lag == 0) ? "✅" : (lag < 10) ? "⚠️ " : "❌"
                printf "   %s %-35s Lag: %5s\n", status, name, lag
            }
        '
        echo ""
    fi
done

echo "=========================================="
echo "Legend:"
echo "  ✅ = No lag (real-time)"
echo "  ⚠️  = Minor lag (<10 frames)"
echo "  ❌ = Significant lag (≥10 frames)"
echo "=========================================="
