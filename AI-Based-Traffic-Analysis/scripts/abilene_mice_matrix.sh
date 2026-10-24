#!/usr/bin/env bash
# FCT scenario (paired: every method replays the same schedule for a given seed). Resumable: finished runs are skipped.
# usage: [METHODS="..."] [SEEDS="..."] scripts/abilene_mice_matrix.sh 2>&1 | tee results/abilene/mice_matrix_log.txt
cd "$(dirname "$0")/.."
METHODS=${METHODS:-"static_shortest ecmp round_robin lstm_hysteresis transformer_minmax_lp persistence_minmax_lp"}
SEEDS=${SEEDS:-"1 2 3 4 5"}
curl -s http://127.0.0.1:8080/te/stats | grep -q abilene12 || { echo "Ryu is not running with the Abilene topology. Start it: ./scripts/start_ryu.sh"; exit 1; }
sudo -v || exit 1
( while true; do sudo -n true; sleep 50; done ) & KEEP=$!
trap 'kill $KEEP 2>/dev/null' EXIT
mkdir -p results/abilene/schedules_mice
for seed in $SEEDS; do
  [ -f results/abilene/schedules_mice/seed_$seed.csv ] || .venv/bin/python -m abilene.replay --seed "$seed" --mice-per-s "${MICE_PER_S:-4}" --out results/abilene/schedules_mice/seed_$seed.csv >/dev/null || exit 1
done
for seed in $SEEDS; do for m in $METHODS; do
  if ls results/runs/*_${m}_abilene_replay_mice_s${seed}/summary.json >/dev/null 2>&1; then echo "skip $m s$seed (done)"; continue; fi
  echo "=== $(date +%H:%M:%S) $m seed $seed"
  ./scripts/run_abilene_mice.sh "$m" "$seed" || echo "FAILED $m seed $seed"
done; done
