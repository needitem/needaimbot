#!/bin/bash
# Before/after probe for the network latency tuning. RTT exposes NIC coalescing
# directly (512us of rx-usecs shows up as ~0.5ms of added average RTT), so this
# works without the game running. Run it before and after tools/net_tune.sh.
HOST=${1:-192.168.137.1}
echo "== $(date +%H:%M:%S)  rx-usecs=$(ethtool -c eno1 2>/dev/null | grep -w rx-usecs | awk '{print $2}')  backlog=$(sysctl -n net.core.netdev_max_backlog) =="
ping -c 300 -i 0.01 -q "$HOST" 2>&1 | tail -1
