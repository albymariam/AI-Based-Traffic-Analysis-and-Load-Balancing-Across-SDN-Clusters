#!/usr/bin/env bash
# Staleness sweep with estimated demand (dist_lp_est_s<D>). 5 ages x 10 seeds = 50 runs x ~5.4 min = ~4.5 h; resumable.
# usage: [DELTAS="0 5 10 20 40"] scripts/abilene_est_staleness.sh 2>&1 | tee results/abilene/est_staleness_log.txt
cd "$(dirname "$0")/.."
DELTAS=${DELTAS:-"0 5 10 20 40"}; SEEDS=${SEEDS:-"1 2 3 4 5 6 7 8 9 10"}
M=""; for d in $DELTAS; do M="$M dist_lp_est_s$d"; done
METHODS="$M" SEEDS="$SEEDS" ./scripts/abilene_matrix.sh
