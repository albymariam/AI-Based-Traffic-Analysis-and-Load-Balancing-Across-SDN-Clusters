#!/usr/bin/env bash
# usage: [DELTAS="0 2 5 20"] [SEEDS="1 2 3"] scripts/run_staleness.sh
cd "$(dirname "$0")/.."
DELTAS=${DELTAS:-"0 1 2 5 10 20"}; SEEDS=${SEEDS:-"1 2 3"}
M=""; for d in $DELTAS; do M="$M dist_lp_s$d"; done
METHODS="$M" LOADS="near_saturation" PROFILE=steady ./scripts/run_matrix.sh "$SEEDS" 120