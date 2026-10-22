#!/usr/bin/env bash
# Telemetry for predictor training: replays TRAIN weeks (seeds >= 100) with static_tracked. Resumable.
# usage: [SEEDS="101 ... 112"] scripts/abilene_collect.sh 2>&1 | tee results/abilene/collect_log.txt
cd "$(dirname "$0")/.."
SEEDS=${SEEDS:-"101 102 103 104 105 106 107 108 109 110 111 112"}
curl -s http://127.0.0.1:8080/te/stats | grep -q abilene12 || { echo "Ryu is not running with the Abilene topology. Start it: ./scripts/start_ryu.sh"; exit 1; }
sudo -v || exit 1
( while true; do sudo -n true; sleep 50; done ) & KEEP=$!
trap 'kill $KEEP 2>/dev/null' EXIT
mkdir -p results/abilene/schedules
for seed in $SEEDS; do
  [ -f results/abilene/schedules/seed_$seed.csv ] || .venv/bin/python -m abilene.replay --seed "$seed" --out results/abilene/schedules/seed_$seed.csv >/dev/null || exit 1
done
for seed in $SEEDS; do
  if ls results/runs/*_static_tracked_abilene_replay_s${seed}/summary.json >/dev/null 2>&1; then echo "skip s$seed (done)"; continue; fi
  echo "=== $(date +%H:%M:%S) static_tracked seed $seed"
  ./scripts/run_abilene.sh static_tracked "$seed" || echo "FAILED seed $seed"
done
