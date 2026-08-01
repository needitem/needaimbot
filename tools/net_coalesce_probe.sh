#!/bin/bash
# Diagnose whether the Tegra nvethernet driver will accept lower RX coalescing.
# rx-usecs=512 looks like the dominant idle-RTT term here (avg-min ~= 0.55ms,
# which matches 512us almost exactly), so it is worth finding a settable value.
# Also tries rx-frames: the NIC additionally interrupts every 64 frames, and at
# ~35k pps that batching may matter more than the timer.
IFACE=eno1
HOST=${1:-192.168.137.1}
probe() {  # $1 = human label
  local r; r=$(ping -c 200 -i 0.01 -q "$HOST" 2>&1 | tail -1)
  printf "  %-28s %s\n" "$1" "$r"
}
cur() { ethtool -c $IFACE 2>/dev/null | grep -E '^rx-usecs:|^rx-frames:' | tr '\n' ' '; }

echo "start: $(cur)"
probe "baseline"
for spec in "rx-usecs 64" "rx-usecs 16" "rx-usecs 8" "rx-usecs 0" \
            "rx-frames 16" "rx-frames 1" "rx-usecs 8 rx-frames 8"; do
  if out=$(ethtool -C $IFACE $spec 2>&1); then
    echo "OK   set: $spec   -> now: $(cur)"
    probe "after [$spec]"
  else
    echo "FAIL set: $spec   -> $out"
  fi
done
echo
echo "restoring stock (rx-usecs 512, rx-frames 64)"
ethtool -C $IFACE rx-usecs 512 rx-frames 64 2>&1 | head -1
echo "final: $(cur)"
