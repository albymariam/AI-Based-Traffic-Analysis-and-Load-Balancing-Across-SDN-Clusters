#!/usr/bin/env bash
# Telemetry for predictor training: replays TRAIN weeks (seeds >= 100) with static_tracked. Resumable.
# usage: [SEEDS="101 ... 112"] scripts/abilene_collect.sh
cd "$(dirname "$0")/.."
SEEDS=${SEEDS:-"101 102 103 104 105 106 107 108 109 110 111 112"}
for seed in $SEEDS; do
  if ls results/runs/*_static_tracked_abilene_replay_s${seed}/summary.json >/dev/null 2>&1; then echo "skip s$seed (done)"; continue; fi
  ./scripts/run_abilene.sh static_tracked "$seed" || echo "FAILED seed $seed"
done
