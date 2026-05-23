#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

EVAL_CACHE="${1:-benchmarks/induction/eval/induction_eval_cache_v1.jsonl}"
OUT_DIR="${OUT_DIR:-benchmarks/induction/generated_tables}"

python benchmarks/induction/analysis/make_tables.py "$EVAL_CACHE" --out "$OUT_DIR"
