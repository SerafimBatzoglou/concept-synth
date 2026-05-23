#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

OUT="${OUT:-benchmarks/induction/generated_instances/induction_generated_sample.yaml}"

python -m concept_synth.induction.cli generate \
  --output "$OUT" \
  "$@"
