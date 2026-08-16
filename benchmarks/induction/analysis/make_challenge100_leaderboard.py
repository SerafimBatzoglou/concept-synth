#!/usr/bin/env python3
"""Render the public Challenge100 and Challenge64 leaderboard."""

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


def pct(num: int, den: int) -> str:
    return "N/A" if not den else f"{100.0 * num / den:.1f}%"


def render_challenge100(
    *, registry: dict[str, Any], eval_path: Path, holdout_path: Path
) -> list[str]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in iter_jsonl(eval_path):
        by_model.setdefault(str(row["model_id"]), []).append(row)
    holdout_by_model: dict[str, list[dict[str, Any]]] = {}
    for row in iter_jsonl(holdout_path):
        holdout_by_model.setdefault(str(row["model_id"]), []).append(row)

    rows: list[tuple[int, int, str, str]] = []
    for model in registry["models"]:
        c64 = model["challenge64"]
        new36 = model["benchmarked36"]
        c100 = model["challenge100"]
        c64_model_id = str(c64["registry_id"])
        c64_rows = by_model.get(c64_model_id, [])
        if len(c64_rows) != 64:
            raise ValueError(
                f"{c64_model_id}: expected 64 Challenge64 rows, found {len(c64_rows)}"
            )
        asts = [
            row.get("prediction", {}).get("ast_size")
            for row in c64_rows
            if row.get("valid")
            and isinstance(row.get("prediction", {}).get("ast_size"), int)
        ]
        asts.extend(int(value) for value in new36.get("correct_formula_ast_sizes") or [])
        if len(asts) != int(c100["correct"]):
            raise ValueError(
                f'{model["id"]}: expected {c100["correct"]} train-correct AST sizes, found {len(asts)}'
            )
        holdout_rows = [
            row
            for row in holdout_by_model.get(c64_model_id, [])
            if (row.get("metadata") or {}).get("eligible_train_valid")
            and (row.get("metadata") or {}).get("holdout_available")
        ]
        holdout_correct = sum(bool(row.get("valid")) for row in holdout_rows)
        holdout_correct += int(new36.get("holdout_correct") or 0)
        holdout_evaluable = len(holdout_rows) + int(new36.get("holdout_evaluable") or 0)
        holdout = (
            f"{pct(holdout_correct, holdout_evaluable)} ({holdout_correct}/{holdout_evaluable})"
            if holdout_evaluable else "N/A"
        )
        complexity = f"{mean(asts):.1f} / {median(asts):.1f}" if asts else "N/A"
        rendered = (
            "| {name} | {evaluable} | {correct} | {holdout} | {complexity} |"
        ).format(
            name=str(model["display_name"]),
            evaluable=f'{c100["evaluable"]}/100',
            correct=f'{c100["correct"]}/100 ({pct(c100["correct"], 100)})',
            holdout=holdout,
            complexity=complexity,
        )
        rows.append((-int(c100["correct"]), -int(c100["evaluable"]), str(model["display_name"]), rendered))
    rows.sort()
    return [row[3] for row in rows]


def render_challenge64(
    *, registry: dict[str, Any], eval_path: Path, holdout_path: Path
) -> list[str]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in iter_jsonl(eval_path):
        by_model.setdefault(str(row["model_id"]), []).append(row)
    holdout_by_model: dict[str, list[dict[str, Any]]] = {}
    for row in iter_jsonl(holdout_path):
        holdout_by_model.setdefault(str(row["model_id"]), []).append(row)

    rendered: list[tuple[int, int, str, str]] = []
    for model in registry["models"]:
        model_id = str(model["id"])
        rows = by_model.get(model_id, [])
        if len(rows) != 64:
            raise ValueError(f"{model_id}: expected 64 Challenge64 rows, found {len(rows)}")
        evaluable = sum(bool(row.get("parse_ok")) for row in rows)
        correct = sum(bool(row.get("valid")) for row in rows)
        asts = [
            row.get("prediction", {}).get("ast_size")
            for row in rows
            if row.get("valid") and isinstance(row.get("prediction", {}).get("ast_size"), int)
        ]
        holdout_rows = [
            row for row in holdout_by_model.get(model_id, [])
            if (row.get("metadata") or {}).get("eligible_train_valid")
            and (row.get("metadata") or {}).get("holdout_available")
        ]
        holdout_correct = sum(bool(row.get("valid")) for row in holdout_rows)
        holdout = (
            f"{pct(holdout_correct, len(holdout_rows))} ({holdout_correct}/{len(holdout_rows)})"
            if holdout_rows else "N/A"
        )
        complexity = f"{mean(asts):.1f} / {median(asts):.1f}" if asts else "N/A"
        line = "| {name} | {evaluable} | {correct} | {holdout} | {complexity} |".format(
            name=str(model["display_name"]),
            evaluable=f"{evaluable}/64",
            correct=f"{correct}/64 ({pct(correct, 64)})",
            holdout=holdout,
            complexity=complexity,
        )
        rendered.append((-correct, -evaluable, str(model["display_name"]), line))
    rendered.sort()
    return [row[3] for row in rendered]


def render(
    *,
    challenge100_registry_path: Path,
    challenge64_registry_path: Path,
    challenge64_eval_path: Path,
    challenge64_holdout_path: Path,
) -> str:
    c100 = yaml.safe_load(challenge100_registry_path.read_text(encoding="utf-8"))
    c64 = yaml.safe_load(challenge64_registry_path.read_text(encoding="utf-8"))
    lines = [
        "# INDUCTION Challenge Leaderboards",
        "",
        "Challenge100 is the ordered union of the frozen Challenge64 benchmark and the disjoint New36 component. "
        "The Challenge100 table includes models with results on New36; the Challenge64 table remains additive and "
        "therefore includes additional models.",
        "",
        "Missing, provider-error, empty, output-limit-incomplete, and parse-invalid responses count as incorrect. "
        "A multi-formula response is evaluable if any submitted formula parses and correct if any submitted formula "
        "is train-valid. Residual cascades use parser-evaluable priority only, never correctness or holdout outcomes.",
        "",
        "## Challenge100",
        "",
        "Rows are ranked by Challenge100 Correct, then Evaluable coverage, then model name.",
        "",
        "| Model | Evaluable | Correct | Holdout Correct<br>(among train-correct) | Formula Complexity<br>(AST mean/median) |",
        "|---|---:|---:|---:|---:|",
        *render_challenge100(
            registry=c100,
            eval_path=challenge64_eval_path,
            holdout_path=challenge64_holdout_path,
        ),
        "",
        "Challenge100 formula complexity covers all train-correct direct formulas across its 100 tasks. "
        "Its generated-IID holdout diagnostic combines the frozen Challenge64 and New36 sidecars and reports only "
        "train-correct responses whose task has generated holdout worlds.",
        "",
        "## Challenge64 projection",
        "",
        "Rows are ranked by Challenge64 train-set Correct, then Evaluable coverage, then model name. "
        "Holdout is a post-selection diagnostic and is never used for prompting or selection.",
        "",
        "| Model | Evaluable | Correct | Holdout Correct<br>(among train-correct) | Formula Complexity<br>(AST mean/median) |",
        "|---|---:|---:|---:|---:|",
        *render_challenge64(
            registry=c64,
            eval_path=challenge64_eval_path,
            holdout_path=challenge64_holdout_path,
        ),
        "",
        "Formula complexity reports AST mean/median over train-correct direct formulas.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenge100-registry", type=Path, required=True)
    parser.add_argument("--challenge64-registry", type=Path, required=True)
    parser.add_argument("--challenge64-eval", type=Path, required=True)
    parser.add_argument("--challenge64-holdout", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    text = render(
        challenge100_registry_path=args.challenge100_registry,
        challenge64_registry_path=args.challenge64_registry,
        challenge64_eval_path=args.challenge64_eval,
        challenge64_holdout_path=args.challenge64_holdout,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
