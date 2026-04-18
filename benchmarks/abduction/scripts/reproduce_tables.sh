#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT_DIR/generated_tables"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

echo "== Regenerating tables from frozen eval cache =="
python3 "$ROOT_DIR/analysis/make_tables.py" \
  --input "$ROOT_DIR/eval/abd_combined_v1_eval_cache.jsonl" \
  --outdir "$OUT_DIR" \
  --manifest

echo
echo "Wrote regenerated tables to: $OUT_DIR"
