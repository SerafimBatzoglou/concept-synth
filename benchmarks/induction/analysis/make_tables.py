#!/usr/bin/env python3
"""Regenerate compact INDUCTION summary tables from an eval-cache JSONL file."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def pct(num: int, den: int) -> float:
    return 100.0 * num / den if den else 0.0


def summarize(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row.get("task", ""), row.get("model_id", ""))].append(row)

    summary_rows: List[Dict[str, Any]] = []
    for (task, model), items in sorted(buckets.items()):
        total = len(items)
        valid = sum(1 for row in items if row.get("valid"))
        parse_ok = sum(1 for row in items if row.get("parse_ok"))
        gold_plus_0 = sum(1 for row in items if row.get("gold_plus_0"))
        gold_plus_10 = sum(1 for row in items if row.get("gold_plus_10"))
        gold_plus_25 = sum(1 for row in items if row.get("gold_plus_25"))
        ast_values = [
            row.get("prediction", {}).get("ast_size")
            for row in items
            if row.get("prediction", {}).get("ast_size") is not None
        ]
        summary_rows.append(
            {
                "task": task,
                "model": model,
                "n": total,
                "parse_ok": parse_ok,
                "parse_rate": f"{pct(parse_ok, total):.1f}",
                "valid": valid,
                "valid_rate": f"{pct(valid, total):.1f}",
                "gold_plus_0": gold_plus_0,
                "gold_plus_10": gold_plus_10,
                "gold_plus_25": gold_plus_25,
                "avg_ast": f"{mean(ast_values):.1f}" if ast_values else "",
            }
        )
    return summary_rows


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task",
        "model",
        "n",
        "parse_ok",
        "parse_rate",
        "valid",
        "valid_rate",
        "gold_plus_0",
        "gold_plus_10",
        "gold_plus_25",
        "avg_ast",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_latex(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Task & Model & $n$ & Parse \% & Valid \% & Gold+10 & Avg AST \\",
        r"\midrule",
    ]
    for row in rows:
        model = str(row["model"]).replace("_", r"\_")
        lines.append(
            f"{row['task']} & {model} & {row['n']} & {row['parse_rate']} & "
            f"{row['valid_rate']} & {row['gold_plus_10']} & {row['avg_ast']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_cache", type=Path)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/induction/generated_tables"))
    args = parser.parse_args()

    rows = summarize(iter_jsonl(args.eval_cache))
    write_csv(rows, args.out / "induction_summary.csv")
    write_latex(rows, args.out / "induction_summary.tex")
    print(f"Wrote {len(rows)} task/model rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
