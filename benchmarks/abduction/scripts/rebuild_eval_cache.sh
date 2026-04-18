#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT_DIR/../.." && pwd)"
DATASET_PATH="${DATASET_PATH:-$ROOT_DIR/data/abd_instances_v1.yaml.gz}"
HOLDOUTS_PATH="${HOLDOUTS_PATH:-$ROOT_DIR/data/abd_holdouts_v1.jsonl.gz}"
PREDICTIONS_PATH="${PREDICTIONS_PATH:-$ROOT_DIR/predictions/abd_predictions_v1.jsonl.gz}"
OUTPUT_PATH="${OUTPUT_PATH:-$ROOT_DIR/generated_eval/abd_eval_cache_v1.jsonl}"

mkdir -p "$(dirname "$OUTPUT_PATH")"

PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m concept_synth.abduction.cli evaluate \
  --dataset "$DATASET_PATH" \
  --holdouts "$HOLDOUTS_PATH" \
  --predictions "$PREDICTIONS_PATH" \
  --output "$OUTPUT_PATH" \
  "$@"

echo
echo "Wrote rebuilt eval cache to: $OUTPUT_PATH"
