#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT_DIR/generated_tables"
CACHE_PATH="${1:-$ROOT_DIR/eval/abd_combined_v1_eval_cache.jsonl}"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

echo "== Regenerating tables from eval cache =="
python3 "$ROOT_DIR/analysis/make_tables.py" \
  --input "$CACHE_PATH" \
  --outdir "$OUT_DIR" \
  --manifest

echo
echo "Wrote regenerated tables to: $OUT_DIR"
echo "Input cache: $CACHE_PATH"
