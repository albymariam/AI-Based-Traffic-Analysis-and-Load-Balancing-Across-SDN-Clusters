#!/usr/bin/env bash
# usage: [METHODS="..."] [LOADS="..."] [PROFILE=steady] scripts/run_matrix.sh "<seeds>" <duration_s>
cd "$(dirname "$0")/.."
SEEDS=${1:-"1 2 3"}; DUR=${2:-120}
METHODS=${METHODS:-"static_shortest ecmp round_robin static_tracked"}
LOADS=${LOADS:-"light near_saturation"}; PROFILE=${PROFILE:-steady}
for load in $LOADS; do for seed in $SEEDS; do for m in $METHODS; do
  ./scripts/run_one.sh $m $load $PROFILE $seed $DUR || echo "FAILED $m $load $seed"
done; done; done