#!/usr/bin/env bash
# Stale-view sweep on the Abilene replay (dist_lp = Ryu-based emulation of C domain controllers with a delayed view of remote state).
# usage: [DELTAS="0 2 5 10 20"] [SEEDS="1 2 3 4 5"] scripts/abilene_staleness.sh 2>&1 | tee results/abilene/staleness_log.txt
# Needs Ryu running (./scripts/start_ryu.sh). Resumable. transformer_minmax_lp and static_shortest come from the main matrix.
cd "$(dirname "$0")/.."
DELTAS=${DELTAS:-"0 2 5 10 20"}; SEEDS=${SEEDS:-"1 2 3 4 5"}
M=""; for d in $DELTAS; do M="$M dist_lp_s$d"; done
METHODS="$M" SEEDS="$SEEDS" ./scripts/abilene_matrix.sh
