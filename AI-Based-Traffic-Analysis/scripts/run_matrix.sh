#!/usr/bin/env bash
# Usage: METHODS="..." LOADS="..." PROFILES="..." ./scripts/run_matrix.sh "101 102 103 104" 120

cd "$(dirname "$0")/.."

SEEDS=${1:-"101 102 103 104"}
DUR=${2:-120}

METHODS=${METHODS:-"static_tracked ecmp round_robin lstm_hysteresis"}
LOADS=${LOADS:-"light moderate near_saturation"}
PROFILES=${PROFILES:-"steady bursty"}

for profile in $PROFILES; do
  for load in $LOADS; do
    for seed in $SEEDS; do
      for method in $METHODS; do
        echo "Running: method=$method load=$load profile=$profile seed=$seed duration=$DUR"
        ./scripts/run_one.sh "$method" "$load" "$profile" "$seed" "$DUR" ||
          echo "FAILED: $method $load $profile $seed"
      done
    done
  done
done
