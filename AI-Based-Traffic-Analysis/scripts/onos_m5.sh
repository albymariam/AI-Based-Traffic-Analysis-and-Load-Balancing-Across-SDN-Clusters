#!/usr/bin/env bash
# M5 stage 1: replay the TE controller's flow installs on real ONOS and measure the control-plane cost.
#   bash scripts/onos_m5.sh probe          one dry round trip on the cluster (devices, one rule, ADDED latency); run this first
#   bash scripts/onos_m5.sh run [A B C10 C25 C50 C100]    the conditions (default: all six, about 1.5 h)
#   bash scripts/onos_m5.sh down           stop everything
# A = one ONOS node, B = 3-node cluster, C<ms> = cluster with <ms> of added network delay per hop (tc netem)
# Needs: Docker, sudo, data/onos_m5/events_s1.csv (git pull), .venv. Results -> results/onos_m5/<cond>/
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
EVENTS=${EVENTS:-data/onos_m5/events_s1.csv}
SETTLE=${SETTLE:-90}                       # extra seconds after the nodes report READY
NODES_B=localhost:8182,localhost:8183,localhost:8184
NODES_A=localhost:8185
CTRS_B="atomix-b1 atomix-b2 atomix-b3 onos-b1 onos-b2 onos-b3"
ATOMIX_B="atomix-b1 atomix-b2 atomix-b3"
export PYTHONPATH=.

stop_net() { sudo pkill -f "[o]nos_m5_net.py" 2>/dev/null; sleep 2; sudo mn -c >/dev/null 2>&1; }
stop_all() {
  stop_net
  ( cd experiments/onos_single && docker compose down -v >/dev/null 2>&1 )
  ( cd experiments/onos_cluster_b && docker compose down -v >/dev/null 2>&1 )
}
tc_all() { for c in $CTRS_B; do docker run --rm --net=container:$c --cap-add NET_ADMIN nicolaka/netshoot sh -c "$1" >/dev/null || return 1; done; }

start_cond() {   # $1 = A | B   -> sets NODES, OFP, MEMBERS
  if [ "$1" = A ]; then
    NODES=$NODES_A; OFP=6657; MEMBERS=1
    ( cd experiments/onos_single && docker compose up -d ) || return 1
  else
    NODES=$NODES_B; OFP=6654,6655,6656; MEMBERS=3
    ( cd experiments/onos_cluster_b && docker compose up -d ) || return 1
  fi
  printf "waiting for ONOS to become READY "
  for i in $(seq 1 90); do
    $PY scripts/onos_b_check.py --nodes $NODES --expect-members $MEMBERS >/dev/null 2>&1 && break
    printf "."; sleep 5
    [ "$i" = 90 ] && { echo " TIMEOUT (docker logs onos-b1 | tail -50)"; return 1; }
  done
  echo " ready; settling ${SETTLE}s"; sleep $SETTLE
}

start_mininet() {
  sudo -b env T11_TOPOLOGY=abilene12 PYTHONPATH=.:/usr/lib/python3/dist-packages .venv/bin/python experiments/onos_m5_net.py --of-ports $OFP >/tmp/onos_m5_net.log 2>&1
  printf "waiting for the 12 switches "
  for i in $(seq 1 60); do
    grep -q READY /tmp/onos_m5_net.log 2>/dev/null && break
    printf "."; sleep 3
    [ "$i" = 60 ] && { echo " TIMEOUT"; tail -20 /tmp/onos_m5_net.log; return 1; }
  done
  for i in $(seq 1 60); do
    $PY scripts/onos_b_check.py --nodes $NODES --expect-members $MEMBERS --expect-devices 12 >/dev/null 2>&1 && { echo " 12 devices on all nodes"; return 0; }
    sleep 3
  done
  echo " devices did not appear on every node"; return 1
}

run_cond() {   # $1 = A | B | C<ms>
  local cond=$1 base=$1 ms=0 out=results/onos_m5/$1
  case $cond in C*) base=B; ms=${cond#C};; esac
  echo; echo "######## condition $cond  $(date +%T)"
  stop_all; mkdir -p $out
  start_cond $base || { echo "condition $cond FAILED to start"; return 1; }
  if [ "$ms" != 0 ]; then tc_all "tc qdisc replace dev eth0 root netem delay ${ms}ms" && echo "added ${ms} ms per hop" || return 1; fi
  start_mininet || return 1
  if [ "$base" = B ]; then
    $PY scripts/onos_b_check.py --nodes $NODES --expect-devices 12 --flow-test --repeat 30 --json $out/propagation.json 2>&1 | tail -12
    python3 scripts/docker_net_bytes.py $ATOMIX_B > $out/net_idle0.json
    sleep 30
    python3 scripts/docker_net_bytes.py $ATOMIX_B > $out/net_idle1.json
    python3 scripts/docker_net_bytes.py $ATOMIX_B > $out/net_before.json
  else
    $PY scripts/onos_b_check.py --nodes $NODES --expect-members 1 --expect-devices 12 --flow-test --repeat 30 --json $out/propagation.json 2>&1 | tail -6
  fi
  $PY experiments/onos_m5_replay.py --nodes $NODES --events $EVENTS --out $out 2>&1 | tee $out/replay.log
  [ "$base" = B ] && python3 scripts/docker_net_bytes.py $ATOMIX_B > $out/net_after.json
  stop_net
  echo "condition $cond done -> $out"
}

case "$1" in
  probe)
    stop_all
    start_cond B && start_mininet && $PY experiments/onos_m5_replay.py --nodes $NODES_B --probe
    $PY scripts/onos_b_check.py --nodes $NODES_B --expect-devices 12 --flow-test | tail -12
    stop_all ;;
  run)
    shift; CONDS=${@:-A B C10 C25 C50 C100}
    [ -f "$EVENTS" ] || { echo "missing $EVENTS (git pull)"; exit 1; }
    for c in $CONDS; do run_cond $c || echo "!! $c failed, continuing"; done
    stop_all
    $PY evaluation/onos_m5_analysis.py ;;
  down) stop_all ;;
  *) echo "usage: bash scripts/onos_m5.sh probe | run [A B C10 C25 C50 C100] | down"; exit 1 ;;
esac
