#!/usr/bin/env bash
# Records exactly what M3 runs with. Do not change these files after this point.
cd "$(dirname "$0")/.."
mkdir -p results/m3/configs
{
  echo "frozen_at: $(date -Is)"
  echo "git: $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (DIRTY: commit first!)')"
  sha256sum models/checkpoints/transformer.pt models/transformer_predictor.py controller/lp_policy.py \
            optimization/minmax_routing.py configs/experiment.yaml configs/onos_m3.yaml 2>&1
  echo "--- lp settings ---"
  sed -n '/^lp:/,/^[a-z_]*:/p' configs/experiment.yaml | head -12
} | tee results/m3/configs/frozen.txt