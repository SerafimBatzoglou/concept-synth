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

    rows: list[tuple[int, int, int, str, str]] = []
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
        holdout = f"{pct(holdout_correct, 100)} ({holdout_correct}/100)"
        complexity = f"{mean(asts):.1f} / {median(asts):.1f}" if asts else "N/A"
        rendered = (
            "| {name} | {holdout} | {correct} | {evaluable} | {complexity} |"
        ).format(
            name=str(model["display_name"]),
            evaluable=f'{c100["evaluable"]}/100',
            correct=f'{c100["correct"]}/100 ({pct(c100["correct"], 100)})',
            holdout=holdout,
            complexity=complexity,
        )
        rows.append((-holdout_correct, -int(c100["correct"]), -int(c100["evaluable"]), str(model["display_name"]), rendered))
    rows.sort()
    return [row[4] for row in rows]


def render_challenge64(
    *, registry: dict[str, Any], eval_path: Path, holdout_path: Path
) -> list[str]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in iter_jsonl(eval_path):
        by_model.setdefault(str(row["model_id"]), []).append(row)
    holdout_by_model: dict[str, list[dict[str, Any]]] = {}
    for row in iter_jsonl(holdout_path):
        holdout_by_model.setdefault(str(row["model_id"]), []).append(row)

    rendered: list[tuple[int, int, int, str, str]] = []
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
        holdout = f"{pct(holdout_correct, 64)} ({holdout_correct}/64)"
        complexity = f"{mean(asts):.1f} / {median(asts):.1f}" if asts else "N/A"
        line = "| {name} | {holdout} | {correct} | {evaluable} | {complexity} |".format(
            name=str(model["display_name"]),
            evaluable=f"{evaluable}/64",
            correct=f"{correct}/64 ({pct(correct, 64)})",
            holdout=holdout,
            complexity=complexity,
        )
        rendered.append((-holdout_correct, -correct, -evaluable, str(model["display_name"]), line))
    rendered.sort()
    return [row[4] for row in rendered]


def render(
    *,
    challenge100_registry_path: Path,
    challenge64_registry_path: Path,
    challenge64_eval_path: Path,
    challenge64_holdout_path: Path,
) -> str:
    c100 = yaml.safe_load(challenge100_registry_path.read_text(encoding="utf-8"))
    c64 = yaml.safe_load(challenge64_registry_path.read_text(encoding="utf-8"))
    c64_models = {str(model["id"]): model for model in c64["models"]}
    if len(c64_models) != len(c64["models"]):
        raise ValueError("Duplicate Challenge64 model IDs")
    projection_ids = [str(model["challenge64"]["registry_id"]) for model in c100["models"]]
    if len(set(projection_ids)) != len(projection_ids):
        raise ValueError("Duplicate Challenge100 projection IDs")
    evals: dict[str, list[dict[str, Any]]] = {}
    for row in iter_jsonl(challenge64_eval_path):
        evals.setdefault(str(row["model_id"]), []).append(row)
    for model in c100["models"]:
        projection = model["challenge64"]
        model_id = str(projection["registry_id"])
        if model_id not in c64_models:
            raise ValueError(f"{model_id}: missing from Challenge64 registry")
        if model["display_name"] != c64_models[model_id]["display_name"]:
            raise ValueError(f"{model_id}: inconsistent display name across leaderboards")
        rows = evals.get(model_id, [])
        if len(rows) != 64:
            raise ValueError(f"{model_id}: expected 64 Challenge64 rows, found {len(rows)}")
        actual = {
            "evaluable": sum(bool(row.get("parse_ok")) for row in rows),
            "correct": sum(bool(row.get("valid")) for row in rows),
        }
        if any(actual[key] != int(projection[key]) for key in actual):
            raise ValueError(f"{model_id}: Challenge64 projection counts disagree with evaluation cache")
    lines = [
        "# INDUCTION Challenge Leaderboards",
        "",
        "Challenge100 is the ordered union of the frozen Challenge64 benchmark and the disjoint New36 component. "
        f"All {len(c100['models'])} Challenge100 models appear in the Challenge64 table, alongside "
        f"{len(c64['models']) - len(c100['models'])} additional models with Challenge64 results. "
        "Each table is ranked independently by Holdout Correct % over all problems in its task set, so model order differs.",
        "",
        "Missing, provider-error, empty, output-limit-incomplete, and parse-invalid responses count as incorrect. "
        "A multi-formula response is evaluable if any submitted formula parses and correct if any submitted formula "
        "is train-valid. Residual cascades use parser-evaluable priority only, never correctness or holdout outcomes.",
        "",
        "Holdout Correct % is the number of train-correct formulas verified correct on all available generated holdout worlds, "
        "divided by the total number of problems (100 or 64), not by the number of available holdout evaluations. "
        "Tasks without verified holdout success contribute no credit, including tasks missing holdout worlds; "
        "missing holdout outcomes remain unknown in the underlying records, not asserted failures. "
        "This changes leaderboard reporting and ranking only, not model training, evaluation, or response selection.",
        "",
        "## Challenge100",
        "",
        "Rows are ranked by Holdout Correct % (out of 100), then Train Correct, Evaluable coverage, and model name.",
        "",
        "| Model | Holdout Correct %<br>(all problems) | Train Correct | Evaluable | Formula Complexity<br>(AST mean/median) |",
        "|---|---:|---:|---:|---:|",
        *render_challenge100(
            registry=c100,
            eval_path=challenge64_eval_path,
            holdout_path=challenge64_holdout_path,
        ),
        "",
        "Challenge100 formula complexity covers all train-correct direct formulas across its 100 tasks. "
        "Holdout Correct % combines verified successes from the frozen Challenge64 and New36 sidecars, divided by all 100 problems.",
        "",
        "## Challenge64 projection",
        "",
        "Rows are ranked by Holdout Correct % (out of 64), then Train Correct, Evaluable coverage, and model name. "
        "Holdout is a post-selection diagnostic and is never used for prompting or selection.",
        "",
        "| Model | Holdout Correct %<br>(all problems) | Train Correct | Evaluable | Formula Complexity<br>(AST mean/median) |",
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
