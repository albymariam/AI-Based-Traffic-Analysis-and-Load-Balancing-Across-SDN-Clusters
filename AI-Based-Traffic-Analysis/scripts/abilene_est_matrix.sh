#!/usr/bin/env bash
# Demand ESTIMATED from OpenFlow flow byte counters (no configured rates): the three LP variants on the 10 evaluation seeds.
# static_shortest and the oracle runs come from the main matrix; nothing else needs re-running.
# usage: scripts/abilene_est_matrix.sh 2>&1 | tee results/abilene/est_matrix_log.txt        (30 runs x ~5.4 min = ~2.7 h; resumable)
cd "$(dirname "$0")/.."
METHODS="transformer_minmax_lp_est tcn_minmax_lp_est persistence_minmax_lp_est" SEEDS=${SEEDS:-"1 2 3 4 5 6 7 8 9 10"} ./scripts/abilene_matrix.sh
