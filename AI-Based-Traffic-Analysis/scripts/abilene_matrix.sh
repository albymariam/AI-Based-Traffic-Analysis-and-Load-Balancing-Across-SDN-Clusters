#!/usr/bin/env bash
# Evaluation matrix (paired: every method replays the same schedule for a given seed). Resumable: finished runs are skipped.
# usage: [METHODS="..."] [SEEDS="..."] scripts/abilene_matrix.sh
cd "$(dirname "$0")/.."
METHODS=${METHODS:-"static_shortest static_tracked ecmp round_robin lstm_hysteresis transformer_minmax_lp tcn_minmax_lp"}
SEEDS=${SEEDS:-"1 2 3 4 5 6 7 8 9 10"}
for seed in $SEEDS; do for m in $METHODS; do
  if ls results/runs/*_${m}_abilene_replay_s${seed}/summary.json >/dev/null 2>&1; then echo "skip $m s$seed (done)"; continue; fi
  ./scripts/run_abilene.sh "$m" "$seed" || echo "FAILED $m seed $seed"
done; done
