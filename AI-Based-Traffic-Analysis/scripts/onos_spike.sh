#!/usr/bin/env bash
# ONOS 3-node cluster spike (condition B).  usage: bash scripts/onos_spike.sh up | check | measure | delay <ms> | nodelay | sweep | down
#   up       start the cluster and wait until all three nodes answer, then check membership and apps (first start pulls images)
#   check    membership, apps, devices and one cross-node flow-rule test (start spike_topology.py first)
#   measure  the same with 30 repeated flow tests: median / p95 of the state-propagation delay
#   delay    add <ms> of network delay to every packet leaving each of the six containers (tc netem); nodelay removes it
#   sweep    measure at several delays and log to results/onos_spike_sweep.txt (does the cluster stay healthy? how does sync delay grow?)
#   down     stop and delete the cluster and its data
cd "$(dirname "$0")/.."
NODES=localhost:8182,localhost:8183,localhost:8184
CHECK="env PYTHONPATH=. .venv/bin/python scripts/onos_b_check.py --nodes $NODES"
CTRS="atomix-b1 atomix-b2 atomix-b3 onos-b1 onos-b2 onos-b3"

tc_all() {   # run a tc command inside each container's network namespace (works with Docker Desktop too)
  for c in $CTRS; do
    docker run --rm --net=container:$c --cap-add NET_ADMIN nicolaka/netshoot sh -c "$1" >/dev/null || { echo "tc failed on $c"; return 1; }
  done
}

case "$1" in
  up)
    ( cd experiments/onos_cluster_b && docker compose up -d ) || exit 1
    for n in 8182 8183 8184; do
      printf "waiting for ONOS on :%s " "$n"
      for i in $(seq 1 90); do
        curl -sf -u onos:rocks "http://localhost:$n/onos/v1/devices" >/dev/null && { echo "up"; break; }
        printf "."; sleep 5
        [ "$i" = 90 ] && { echo " TIMEOUT (see: docker logs onos-b1 | tail -50)"; exit 1; }
      done
    done
    docker ps --format '{{.Names}}  {{.Status}}'
    $CHECK ;;
  check)
    $CHECK --expect-devices 4 --flow-test ;;
  measure)
    $CHECK --expect-devices 4 --flow-test --repeat 30 ;;
  delay)
    [ -n "$2" ] || { echo "usage: bash scripts/onos_spike.sh delay <ms>"; exit 1; }
    tc_all "tc qdisc replace dev eth0 root netem delay ${2}ms" && echo "added ${2} ms per hop" ;;
  nodelay)
    tc_all "tc qdisc del dev eth0 root 2>/dev/null; true" && echo "delay removed" ;;
  sweep)
    OUT=results/onos_spike_sweep.txt; mkdir -p results; : > $OUT
    for ms in 0 10 50 100 250 500 1000; do
      echo "######## delay ${ms} ms per hop  $(date +%T)" | tee -a $OUT
      if [ "$ms" = 0 ]; then bash "$0" nodelay >/dev/null; else bash "$0" delay $ms || break; fi
      sleep 20                                   # let the cluster settle before measuring
      bash "$0" measure 2>&1 | tee -a $OUT
    done
    bash "$0" nodelay | tee -a $OUT
    echo "saved to $OUT" ;;
  down)
    ( cd experiments/onos_cluster_b && docker compose down -v ) ;;
  *) echo "usage: bash scripts/onos_spike.sh up | check | measure | delay <ms> | nodelay | sweep | down"; exit 1 ;;
esac
