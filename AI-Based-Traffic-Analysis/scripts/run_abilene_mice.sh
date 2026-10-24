#!/usr/bin/env bash
# usage: scripts/run_abilene_mice.sh <method> <seed>      (FCT scenario: the Abilene replay plus short TCP flows; scenario name abilene_replay_mice)
set -e
cd "$(dirname "$0")/.."
export T11_TOPOLOGY=abilene12
M=$1; SEED=$2
SCHED=results/abilene/schedules_mice/seed_${SEED}.csv
[ -f "$SCHED" ] || .venv/bin/python -m abilene.replay --seed "$SEED" --mice-per-s "${MICE_PER_S:-4}" --out "$SCHED" >/dev/null
sudo mn -c >/dev/null 2>&1
sudo -E env T11_TOPOLOGY=abilene12 PYTHONPATH=.:/usr/lib/python3/dist-packages .venv/bin/python experiments/run_baseline.py \
     --method "$M" --load abilene --profile replay_mice --seed "$SEED" --schedule "$SCHED"
