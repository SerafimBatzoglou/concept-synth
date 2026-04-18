#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT_DIR/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/generated_tables}"
CACHE_PATH="${1:-$ROOT_DIR/eval/abd_eval_cache_v1.jsonl}"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

echo "== Regenerating tables from eval cache =="
PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
python3 "$ROOT_DIR/analysis/make_tables.py" \
  --input "$CACHE_PATH" \
  --outdir "$OUT_DIR" \
  --manifest

echo
echo "Wrote regenerated tables to: $OUT_DIR"
echo "Input cache: $CACHE_PATH"
