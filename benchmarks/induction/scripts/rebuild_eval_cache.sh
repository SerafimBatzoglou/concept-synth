#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

OUT="${OUT:-benchmarks/induction/generated_eval/induction_eval_cache_v1.jsonl}"

python -m concept_synth.induction.cli evaluate \
  --output "$OUT" \
  "$@"
