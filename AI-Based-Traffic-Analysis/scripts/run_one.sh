#!/usr/bin/env bash
# usage: scripts/run_one.sh <method> <load> <profile> <seed> [duration_s]
cd "$(dirname "$0")/.."
sudo mn -c >/dev/null 2>&1
sudo -E env PYTHONPATH=. ~/venvs/ml/bin/python experiments/run_baseline.py \
     --method "$1" --load "$2" --profile "$3" --seed "$4" ${5:+--duration "$5"}