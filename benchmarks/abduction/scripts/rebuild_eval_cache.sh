#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT_DIR/../.." && pwd)"
OUTPUT_PATH="${OUTPUT_PATH:-$ROOT_DIR/generated_eval/abd_combined_v1_eval_cache.jsonl}"

mkdir -p "$(dirname "$OUTPUT_PATH")"

PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m concept_synth.abd_cli evaluate \
  --dataset "$ROOT_DIR/data/abd_combined_v1.yaml.gz" \
  --holdouts "$ROOT_DIR/data/abd_combined_v1.yaml.holdout_k5_seed0_delta12.jsonl" \
  --output "$OUTPUT_PATH" \
  "$@"

echo
echo "Wrote rebuilt eval cache to: $OUTPUT_PATH"
