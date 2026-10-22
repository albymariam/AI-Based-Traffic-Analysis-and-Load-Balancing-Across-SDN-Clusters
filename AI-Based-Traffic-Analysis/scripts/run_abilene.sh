#!/usr/bin/env bash
# usage: scripts/run_abilene.sh <method> <seed>
# Builds the replay schedule for <seed> (seed<100: TEST weeks, seed>=100: TRAIN weeks), then runs one Mininet experiment on it.
set -e
cd "$(dirname "$0")/.."
export T11_TOPOLOGY=abilene12
M=$1; SEED=$2
SCHED=results/abilene/schedules/seed_${SEED}.csv
[ -f "$SCHED" ] || .venv/bin/python -m abilene.replay --seed "$SEED" --out "$SCHED" >/dev/null
sudo mn -c >/dev/null 2>&1
sudo -E env T11_TOPOLOGY=abilene12 PYTHONPATH=.:/usr/lib/python3/dist-packages .venv/bin/python experiments/run_baseline.py \
     --method "$M" --load abilene --profile replay --seed "$SEED" --schedule "$SCHED"
