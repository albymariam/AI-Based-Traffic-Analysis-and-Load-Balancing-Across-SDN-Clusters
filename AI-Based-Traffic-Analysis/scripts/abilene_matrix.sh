#!/usr/bin/env bash
# Evaluation matrix (paired: every method replays the same schedule for a given seed). Resumable: finished runs are skipped.
# usage: [METHODS="..."] [SEEDS="..."] scripts/abilene_matrix.sh 2>&1 | tee results/abilene/matrix_log.txt
cd "$(dirname "$0")/.."
METHODS=${METHODS:-"static_shortest static_tracked ecmp round_robin lstm_hysteresis transformer_minmax_lp tcn_minmax_lp persistence_minmax_lp"}
SEEDS=${SEEDS:-"1 2 3 4 5 6 7 8 9 10"}
curl -s http://127.0.0.1:8080/te/stats | grep -q abilene12 || { echo "Ryu is not running with the Abilene topology. Start it: ./scripts/start_ryu.sh"; exit 1; }
sudo -v || exit 1                                              # keep sudo alive for hours (the default cache lasts 15 min)
( while true; do sudo -n true; sleep 50; done ) & KEEP=$!
trap 'kill $KEEP 2>/dev/null' EXIT
mkdir -p results/abilene/schedules
for seed in $SEEDS; do                                         # build all schedules first: fail fast if the data is missing
  [ -f results/abilene/schedules/seed_$seed.csv ] || .venv/bin/python -m abilene.replay --seed "$seed" --out results/abilene/schedules/seed_$seed.csv >/dev/null || exit 1
done
for seed in $SEEDS; do for m in $METHODS; do
  if ls results/runs/*_${m}_abilene_replay_s${seed}/summary.json >/dev/null 2>&1; then echo "skip $m s$seed (done)"; continue; fi
  echo "=== $(date +%H:%M:%S) $m seed $seed"
  ./scripts/run_abilene.sh "$m" "$seed" || echo "FAILED $m seed $seed"
done; done
