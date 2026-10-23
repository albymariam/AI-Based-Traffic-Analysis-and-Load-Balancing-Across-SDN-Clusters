#!/usr/bin/env bash
# ONOS 3-node cluster spike (condition B).  usage: bash scripts/onos_spike.sh up | check | down
#   up     start the cluster and wait until all three REST APIs answer (first start pulls images; ~5 min)
#   check  membership, apps, devices and cross-node flow-rule visibility (start spike_topology.py first)
#   down   stop and delete the cluster and its data
cd "$(dirname "$0")/.."
NODES=localhost:8182,localhost:8183,localhost:8184
case "$1" in
  up)
    ( cd experiments/onos_cluster_b && docker compose up -d ) || exit 1
    for n in 8182 8183 8184; do
      printf "waiting for ONOS on :%s " "$n"
      for i in $(seq 1 90); do
        curl -sf -u onos:rocks "http://localhost:$n/onos/v1/cluster" >/dev/null && { echo "up"; break; }
        printf "."; sleep 5
        [ "$i" = 90 ] && { echo " TIMEOUT (see: docker logs onos-b1 | tail -50)"; exit 1; }
      done
    done
    docker ps --format '{{.Names}}  {{.Status}}'
    PYTHONPATH=. .venv/bin/python scripts/onos_b_check.py --nodes $NODES ;;
  check)
    PYTHONPATH=. .venv/bin/python scripts/onos_b_check.py --nodes $NODES --expect-devices 4 --flow-test ;;
  down)
    ( cd experiments/onos_cluster_b && docker compose down -v ) ;;
  *) echo "usage: bash scripts/onos_spike.sh up | check | down"; exit 1 ;;
esac
