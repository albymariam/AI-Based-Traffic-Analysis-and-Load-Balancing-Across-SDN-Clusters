#!/usr/bin/env bash
# Make sure Ryu is up with the Abilene topology; start it (detached) if not.
# Needed because `sudo mn -c` (run before every experiment) also kills any running ryu-manager.
cd "$(dirname "$0")/.."
ok() { curl -s --max-time 3 http://127.0.0.1:8080/te/stats | grep -q abilene12; }
ok && exit 0
mkdir -p results/abilene
echo "[ensure_ryu] starting Ryu (log: results/abilene/ryu.log)"
T11_TOPOLOGY=abilene12 setsid nohup ./scripts/start_ryu.sh >> results/abilene/ryu.log 2>&1 < /dev/null &
for _ in $(seq 40); do sleep 1; ok && { echo "[ensure_ryu] Ryu is up"; exit 0; }; done
echo "[ensure_ryu] Ryu did not come up, see results/abilene/ryu.log"; exit 1
