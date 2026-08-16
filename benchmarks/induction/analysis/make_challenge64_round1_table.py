#!/usr/bin/env python3
"""Render the public Challenge64 Round-1 pre-symbolic summary table.

This file is copied into a public release by
``export_induction_challenge64_round1_public.py``.  It deliberately consumes
only release artifacts, so the Markdown table can be reproduced without the
private pipeline stores used to assemble the initial source ledger.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import yaml


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_dataset_size(path: Path) -> int:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = yaml.safe_load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"expected a list dataset in {path}")
    return len(rows)


def pct(num: int, den: int) -> str:
    return "N/A" if not den else f"{100.0 * num / den:.1f}%"


def render(
    *,
    dataset_path: Path,
    eval_path: Path,
    holdout_path: Path,
    registry_path: Path,
) -> str:
    denominator = load_dataset_size(dataset_path)
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    models = registry["models"]
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in iter_jsonl(eval_path):
        by_model.setdefault(str(row["model_id"]), []).append(row)
    holdout_by_model: dict[str, list[dict[str, Any]]] = {}
    for row in iter_jsonl(holdout_path):
        holdout_by_model.setdefault(str(row["model_id"]), []).append(row)

    rendered: list[tuple[int, int, str, str]] = []
    for model in models:
        model_id = str(model["id"])
        rows = by_model.get(model_id, [])
        if len(rows) != denominator:
            raise ValueError(
                f"{model_id}: expected {denominator} evaluation rows, found {len(rows)}"
            )
        evaluable = sum(bool(row.get("parse_ok")) for row in rows)
        valid = sum(bool(row.get("valid")) for row in rows)
        asts = [
            row.get("prediction", {}).get("ast_size")
            for row in rows
            if row.get("valid")
            and isinstance(row.get("prediction", {}).get("ast_size"), int)
        ]
        holdout_rows = [
            row
            for row in holdout_by_model.get(model_id, [])
            if (row.get("metadata") or {}).get("eligible_train_valid")
            and (row.get("metadata") or {}).get("holdout_available")
        ]
        holdout_valid = sum(bool(row.get("valid")) for row in holdout_rows)
        holdout = (
            f"{pct(holdout_valid, len(holdout_rows))} ({holdout_valid}/{len(holdout_rows)})"
            if holdout_rows
            else "N/A"
        )
        complexity = (
            f"{mean(asts):.1f} / {median(asts):.1f}" if asts else "N/A"
        )
        rendered.append(
            (
                -valid,
                -evaluable,
                str(model["display_name"]),
                "| {name} | {evaluable} | {valid} | {holdout} | {complexity} |".format(
                    name=str(model["display_name"]),
                    evaluable=f"{evaluable}/{denominator}",
                    valid=f"{valid}/{denominator} ({pct(valid, denominator)})",
                    holdout=holdout,
                    complexity=complexity,
                ),
            )
        )

    rendered.sort()
    lines = [
        "# INDUCTION Challenge64 Leaderboard: Round 1 (Pre-Symbolic)",
        "",
        "Each configuration contributes one direct Round-1 formula per task. The release contains no "
        "symbolic repair or simplification outputs. Rows are ranked by train-set Correct, then "
        "Evaluable coverage, then model name.",
        "",
        "| Model | Evaluable | Correct | Holdout Correct<br>(among train-correct) | Formula Complexity<br>(AST mean/median) |",
        "|---|---:|---:|---:|---:|",
        *(line for _, _, _, line in rendered),
        "",
        "Evaluable: parser-valid formula under the exact FullObs evaluator. Correct: train-world exact-match "
        "validity, with the fixed 64-task denominator. Holdout Correct: conditional exact-match validity among "
        "train-correct formulas with generated holdout worlds available. Formula complexity summarizes "
        "train-correct direct formulas.",
        "",
        "The fixed holdout sidecar contains five generated worlds where generation succeeded (63/64 tasks); "
        "it is only a post-selection reporting diagnostic. It was not used for model prompting, candidate "
        "selection, or symbolic search.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--holdout-eval", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    text = render(
        dataset_path=args.dataset,
        eval_path=args.eval,
        holdout_path=args.holdout_eval,
        registry_path=args.registry,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
