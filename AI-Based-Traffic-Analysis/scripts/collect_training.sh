#!/usr/bin/env bash
cd "$(dirname "$0")/.."
for prof in steady bursty; do
  METHODS="static_shortest" LOADS="light moderate near_saturation" PROFILE=$prof \
    ./scripts/run_matrix.sh "101 102 103 104" 120
done