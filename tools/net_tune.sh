#!/bin/bash
# Latency tuning for the UDP frame path (Jetson side).
#   apply  : lower NIC coalescing, keep NIC IRQs off the app's realtime cores,
#            deepen the backlog for the ~35k pps frame stream.
#   revert : restore the stock values recorded below.
# Nothing here persists across reboot by design - reboot is the ultimate revert.
IFACE=eno1
# --- stock values, captured 2026-07-25 before any change ---
STOCK_RX_USECS=512
STOCK_BACKLOG=1000
STOCK_IRQ_AFFINITY="0-11"     # 163..167 were unpinned; 162 was pinned to core 4
STOCK_MTU=1500
JUMBO_MTU=9000                # NIC maxmtu is 16383; 9000 is the interoperable size
# The app pins its realtime threads to cores 4-7 (main/recv/callback/sender),
# so NIC IRQs are steered to 0-3 to stop them preempting those threads.
APP_FREE_CORES="0-3"
PEER=192.168.137.1            # the SSH session runs over this link - see mtu-on
WATCHDOG_S=20
STATEF=/tmp/net_tune_mtu_state


# Coalescing has the same constraint as the MTU on this driver: it is rejected
# while the netdev is running. Callers must have the link DOWN. Tries a ladder of
# values and keeps the first that actually sticks, because the driver silently
# clamps or refuses some of them.
#   rx-usecs  : the timer. 512us is the stock value and shows up 1:1 in latency.
#   rx-frames : the packet-count trigger. Stock 64 was fine at 256 packets per
#               video frame, but with jumbo there are only ~35 - the threshold can
#               never fire, so EVERY frame now waits the full timer. Lower it too.
set_coalesce() {
  local want_us want_fr got_us got_fr
  for want_us in 0 8 16 32 64; do
    ethtool -C $IFACE rx-usecs $want_us >/dev/null 2>&1
    got_us=$(ethtool -c $IFACE 2>/dev/null | awk '/^rx-usecs:/{print $2}')
    [ "$got_us" = "$want_us" ] && break
  done
  for want_fr in 4 8 16 32; do
    ethtool -C $IFACE rx-frames $want_fr >/dev/null 2>&1
    got_fr=$(ethtool -c $IFACE 2>/dev/null | awk '/^rx-frames:/{print $2}')
    [ "$got_fr" = "$want_fr" ] && break
  done
  echo "  coalesce = rx-usecs $got_us / rx-frames $got_fr (stock 512/64)"
}

irqs() { grep -iE "$IFACE" /proc/interrupts | awk -F: '{gsub(/ /,"",$1);print $1}'; }

case "$1" in
  apply)
    ethtool -C $IFACE rx-usecs 0 2>/dev/null && echo "  rx-usecs -> 0" || echo "  rx-usecs: FAILED"
    for i in $(irqs); do echo "$APP_FREE_CORES" > /proc/irq/$i/smp_affinity_list 2>/dev/null \
      && echo "  IRQ $i -> $APP_FREE_CORES" || echo "  IRQ $i: not settable"; done
    sysctl -qw net.core.netdev_max_backlog=8192 && echo "  netdev_max_backlog -> 8192"
    ;;
  mtu-on|mtu-off)
    # The Tegra nvethernet driver returns EBUSY for a MTU change while the netdev
    # is running, and NetworkManager's "connection down" only deconfigures IP -
    # the link stays administratively up, so the driver still refuses. The device
    # must be explicitly link-down. (The platform itself allows it: the device
    # tree advertises nvidia,max-platform-mtu = 16383.)
    #
    # DANGER: the admin's SSH runs over THIS link, so this is detached (setsid)
    # and guarded - the watchdog requires BOTH the new MTU and a reachable peer,
    # otherwise it puts everything back. Worst case self-heals in ~20s.
    want=$([ "$1" = "mtu-on" ] && echo $JUMBO_MTU || echo $STOCK_MTU)
    con=$(nmcli -t -f NAME,DEVICE connection show --active | awk -F: -v d=$IFACE '$2==d{print $1;exit}')
    [ -z "$con" ] && { echo "  no active NM profile for $IFACE"; exit 1; }
    rm -f $STATEF
    echo "  profile '$con' : mtu -> $want (detached; link WILL drop)"
    setsid bash -c "
      set_mtu() {   # \$1 = mtu ; needs the netdev fully down for this driver
        nmcli device disconnect $IFACE >/dev/null 2>&1
        ip link set dev $IFACE down
        sleep 1
        ip link set dev $IFACE mtu \$1
        rc=\$?
        ip link set dev $IFACE up
        nmcli connection up '$con' >/dev/null 2>&1
        return \$rc
      }
      nmcli connection modify '$con' 802-3-ethernet.mtu $want
      set_mtu $want
      sleep 6
      got=\$(cat /sys/class/net/$IFACE/mtu)
      if [ \"\$got\" = \"$want\" ] && ping -c 3 -W 2 $PEER >/dev/null 2>&1; then
        echo \"ok mtu=\$got\" > $STATEF
      else
        nmcli connection modify '$con' 802-3-ethernet.mtu $STOCK_MTU
        set_mtu $STOCK_MTU
        sleep 3
        echo \"reverted (wanted $want, got \$got, now \$(cat /sys/class/net/$IFACE/mtu), peer \$(ping -c1 -W2 $PEER >/dev/null 2>&1 && echo up || echo down))\" > $STATEF
      fi
    " >/dev/null 2>&1 < /dev/null &
    echo "  wait ~20s then: cat $STATEF"
    ;;
  boot-apply)
    # Called by needaimbot-nettune.service. Applies everything that does not
    # survive a reboot. The MTU is the fragile one: NetworkManager stores it on
    # the profile but cannot apply it (this driver refuses a MTU change while the
    # netdev is running), so it must be done here - and at boot we race NM, so we
    # WAIT for the connection to be active first. Skipping that wait is what made
    # this fail on the previous boot: `--active` returned nothing, the MTU block
    # was skipped, and the frame stream silently died (game_pc sends jumbo).
    con=""
    for i in $(seq 1 30); do
      con=$(nmcli -t -f NAME,DEVICE connection show --active | awk -F: -v d=$IFACE '$2==d{print $1;exit}')
      [ -n "$con" ] && [ "$(cat /sys/class/net/$IFACE/operstate)" = "up" ] && break
      sleep 1
    done
    if [ -z "$con" ]; then
      echo "  ERROR: $IFACE has no active NM profile after 30s - cannot set MTU"
      exit 1
    fi
    echo "  profile  = $con (waited ${i}s for activation)"

    # Apply MTU (+ coalescing, which needs the same down-window). Retry: the first
    # attempt can still lose a race with NM's own IP configuration.
    for attempt in 1 2 3; do
      [ "$(cat /sys/class/net/$IFACE/mtu)" = "$JUMBO_MTU" ] && break
      nmcli connection modify "$con" 802-3-ethernet.mtu $JUMBO_MTU
      nmcli device disconnect $IFACE >/dev/null 2>&1
      ip link set dev $IFACE down; sleep 1
      ip link set dev $IFACE mtu $JUMBO_MTU
      set_coalesce
      ip link set dev $IFACE up
      nmcli connection up "$con" >/dev/null 2>&1
      sleep 4
    done
    # Coalescing may still be stock if the MTU was already right and we never
    # opened a down-window above; do it now.
    if [ "$(ethtool -c $IFACE 2>/dev/null | awk '/^rx-usecs:/{print $2}')" -gt 64 ] 2>/dev/null; then
      nmcli device disconnect $IFACE >/dev/null 2>&1
      ip link set dev $IFACE down; sleep 1
      set_coalesce
      ip link set dev $IFACE up
      nmcli connection up "$con" >/dev/null 2>&1
      sleep 3
    fi

    for i in $(irqs); do echo "$APP_FREE_CORES" > /proc/irq/$i/smp_affinity_list 2>/dev/null; done
    sysctl -qw net.core.netdev_max_backlog=8192
    mtu_now=$(cat /sys/class/net/$IFACE/mtu)
    echo "  mtu      = $mtu_now (want $JUMBO_MTU)"
    echo "  coalesce = rx-usecs $(ethtool -c $IFACE 2>/dev/null | awk '/^rx-usecs:/{print $2}') / rx-frames $(ethtool -c $IFACE 2>/dev/null | awk '/^rx-frames:/{print $2}')"
    echo "  irq      = $APP_FREE_CORES"
    echo "  backlog  = $(sysctl -n net.core.netdev_max_backlog)"
    echo "  peer     = $(ping -c2 -W2 $PEER >/dev/null 2>&1 && echo reachable || echo UNREACHABLE)"
    # Fail loudly: a silent 1500 here kills the jumbo frame stream.
    [ "$mtu_now" = "$JUMBO_MTU" ] || { echo "  ERROR: MTU did not apply"; exit 1; }
    ;;
  coalesce)
    # Standalone: bounce the link just to apply the coalescing values. Detached +
    # watchdog because the admin's SSH runs over this link.
    con=$(nmcli -t -f NAME,DEVICE connection show --active | awk -F: -v d=$IFACE '$2==d{print $1;exit}')
    rm -f $STATEF
    echo "  applying coalescing (detached; link WILL drop)"
    setsid bash -c "
      nmcli device disconnect $IFACE >/dev/null 2>&1
      ip link set dev $IFACE down; sleep 1
      $(declare -f set_coalesce)
      IFACE=$IFACE; set_coalesce > /tmp/net_tune_coalesce.out 2>&1
      ip link set dev $IFACE up
      nmcli connection up '$con' >/dev/null 2>&1
      sleep 5
      if ping -c 3 -W 2 $PEER >/dev/null 2>&1; then
        echo \"ok \$(cat /tmp/net_tune_coalesce.out)\" > $STATEF
      else
        ethtool -C $IFACE rx-usecs 512 rx-frames 64 >/dev/null 2>&1
        nmcli connection up '$con' >/dev/null 2>&1
        echo 'reverted (peer unreachable)' > $STATEF
      fi
    " >/dev/null 2>&1 < /dev/null &
    echo "  wait ~15s then: cat $STATEF"
    ;;
  verify)
    # 8972 = 9000 - 20(IP) - 8(ICMP). Must PASS end-to-end before the sender's
    # payload is raised; if it fails, the path cannot carry jumbo - revert.
    if ping -c 3 -W 2 -M do -s 8972 ${2:-192.168.137.1} >/dev/null 2>&1; then
      echo "  JUMBO PATH OK (8972B unfragmented) - safe to raise the sender payload"
    else
      echo "  JUMBO PATH FAILED - do NOT raise the sender payload; run mtu-off"
    fi
    ;;
  revert)
    ethtool -C $IFACE rx-usecs $STOCK_RX_USECS 2>/dev/null && echo "  rx-usecs -> $STOCK_RX_USECS"
    for i in $(irqs); do echo "$STOCK_IRQ_AFFINITY" > /proc/irq/$i/smp_affinity_list 2>/dev/null \
      && echo "  IRQ $i -> $STOCK_IRQ_AFFINITY"; done
    sysctl -qw net.core.netdev_max_backlog=$STOCK_BACKLOG && echo "  netdev_max_backlog -> $STOCK_BACKLOG"
    ip link set dev $IFACE mtu $STOCK_MTU 2>/dev/null && echo "  mtu -> $STOCK_MTU"
    ;;
  show)
    echo "  mtu      : $(cat /sys/class/net/$IFACE/mtu)"
    echo "  rx-usecs : $(ethtool -c $IFACE 2>/dev/null | grep -w rx-usecs | awk '{print $2}')"
    echo "  backlog  : $(sysctl -n net.core.netdev_max_backlog)"
    for i in $(irqs); do echo "  IRQ $i   : $(cat /proc/irq/$i/smp_affinity_list)"; done
    ;;
  *) echo "usage: sudo $0 {apply|revert|show|mtu-on|mtu-off|verify [host]}"; exit 1;;
esac
