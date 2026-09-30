#!/usr/bin/env python3
"""Render the public Challenge100 and Challenge64 leaderboard."""

from __future__ import annotations

import argparse
import gzip
from html import escape
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import yaml


TOKEN_USAGE_PATH = Path(__file__).resolve().parents[1] / 'eval/challenge100_output_token_usage.json'
TABLE_GAP = 24  # Approximately three characters at the table's 14px font size.


def load_token_usage(path: Path, registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding='utf-8'))
    rows = data['models']
    by_id = {row['model_id']: row for row in rows}
    expected = {model['id'] for model in registry['models']}
    if len(by_id) != len(rows) or set(by_id) != expected:
        raise ValueError('Token usage must cover each Challenge100 model exactly once')
    for row in rows:
        if type(row['output_tokens']) is not int or row['output_tokens'] < 0:
            raise ValueError('Output tokens must be a nonnegative integer')
        missing = row['unreported_responses_lower_bound']
        if missing is not None and (type(missing) is not int or missing < 0):
            raise ValueError('Unreported response count must be nonnegative or unknown')
    return by_id


def token_aligned_rows(
    rows: list[str], registry: dict[str, Any], usage: dict[str, dict[str, Any]]
) -> list[list[str]]:
    by_name = {m['display_name']: usage[m['id']] for m in registry['models']}
    if len(by_name) != len(registry['models']):
        raise ValueError('Duplicate Challenge100 display names')
    result = []
    for row in rows:
        cells = [cell.strip() for cell in row.strip('|').split('|')]
        entry = by_name[cells[0]]
        result.append(cells + [f"{entry['output_tokens'] / 1_000_000:.1f}"])
    return result


def render_challenge100_svg(rows: list[list[str]]) -> str:
    """Two separately bordered tables sharing exact header and row geometry.

    GitHub strips layout CSS from Markdown HTML. An SVG keeps a true blank gap
    between the tables and prevents independent row wrapping from misaligning
    token usage. The Markdown document also includes a text-only fallback.
    """
    left_widths = [190, 168, 148, 94, 164]
    right_widths = [124]
    left_width = sum(left_widths)
    right_x = left_width + TABLE_GAP
    width = right_x + sum(right_widths)
    header_h, row_h = 56, 32
    height = header_h + len(rows) * row_h + 2
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">',
        '<title id="title">Challenge100 leaderboard and output token usage</title>',
        '<desc id="description">Two separate row-aligned tables. Output tokens are in millions, include reasoning once and all known attempts, and exclude unreported usage. An expandable text table accompanies this image.</desc>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;font-size:14px;fill:#1f2328}.header{font-weight:600;font-size:13px}.border{stroke:#d1d9e0;stroke-width:1;fill:none}.stripe{fill:#f6f8fa}.background{fill:#fff}</style>',
    ]
    headers = [
        ['Model', 'Holdout Correct %\n(all problems)', 'Train Correct', 'Evaluable', 'Formula Complexity\n(AST mean/median)'],
        ['Output tokens\n(millions)'],
    ]
    for group_id, x, widths, cols, labels in [
        ('leaderboard', 0, left_widths, range(5), headers[0]),
        ('token-usage', right_x, right_widths, range(5, 6), headers[1]),
    ]:
        table_width = sum(widths)
        parts.append(f'<g id="{group_id}" data-x="{x}" data-width="{table_width}">')
        parts.append(f'<rect class="background" x="{x}" y="1" width="{table_width}" height="{height - 2}"/>')
        for i in range(len(rows)):
            if i % 2:
                parts.append(f'<rect class="stripe" x="{x}" y="{header_h + i * row_h}" width="{table_width}" height="{row_h}"/>')
        cursor = x
        for w, label in zip(widths, labels):
            lines = label.split('\n')
            for j, line in enumerate(lines):
                y = 33 if len(lines) == 1 else 24 + j * 18
                parts.append(f'<text class="header" x="{cursor + w / 2:g}" y="{y}" text-anchor="middle">{escape(line)}</text>')
            cursor += w
        for i, row in enumerate(rows):
            cursor = x
            y = header_h + i * row_h + 21
            parts.append(f'<g class="data-row" data-row="{i}">')
            for w, col in zip(widths, cols):
                is_left = col == 0
                tx = cursor + 12 if is_left else cursor + w - 12
                anchor = 'start' if is_left else 'end'
                parts.append(f'<text class="value" x="{tx}" y="{y}" text-anchor="{anchor}">{escape(row[col])}</text>')
                cursor += w
            parts.append('</g>')
        for i in range(len(rows)):
            y = header_h + i * row_h
            parts.append(f'<path class="border" d="M {x} {y} H {x + table_width}"/>')
        cursor = x
        for w in widths[:-1]:
            cursor += w
            parts.append(f'<path class="border" d="M {cursor} 1 V {height - 1}"/>')
        parts.append(f'<rect class="border" x="{x + .5}" y="1" width="{table_width - 1}" height="{height - 2}"/>')
        parts.append('</g>')
    parts.append('</svg>')
    return '\n'.join(parts) + '\n'


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
    token_usage_path: Path = TOKEN_USAGE_PATH,
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
    usage = load_token_usage(token_usage_path, c100)
    rows100 = render_challenge100(
        registry=c100, eval_path=challenge64_eval_path, holdout_path=challenge64_holdout_path)
    aligned_rows = token_aligned_rows(rows100, c100, usage)
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
        "![Challenge100 leaderboard with a separate, row-aligned output-token table](challenge100_leaderboard.svg)",
        "",
        "Output tokens include reasoning once and all known attempts, including non-evaluable responses. "
        "Unreported usage is excluded; these figures are lower bounds where usage is missing. "
        "[Token data](../eval/challenge100_output_token_usage.json).",
        "",
        "<details>",
        "<summary>Text-only leaderboard and token usage</summary>",
        "",
        "| Model | Holdout Correct %<br>(all problems) | Train Correct | Evaluable | Formula Complexity<br>(AST mean/median) | Output tokens (M) |",
        "|---|---:|---:|---:|---:|---:|",
        *['| ' + ' | '.join(row) + ' |' for row in aligned_rows],
        "",
        "</details>",
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
    parser.add_argument("--token-usage", type=Path, default=TOKEN_USAGE_PATH)
    parser.add_argument("--svg-out", type=Path, help="Defaults to challenge100_leaderboard.svg beside --out")
    args = parser.parse_args()
    text = render(
        challenge100_registry_path=args.challenge100_registry,
        challenge64_registry_path=args.challenge64_registry,
        challenge64_eval_path=args.challenge64_eval,
        challenge64_holdout_path=args.challenge64_holdout,
        token_usage_path=args.token_usage,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text)
    svg_out = args.svg_out or (args.out.parent / 'challenge100_leaderboard.svg' if args.out else None)
    if svg_out:
        registry = yaml.safe_load(args.challenge100_registry.read_text(encoding='utf-8'))
        usage = load_token_usage(args.token_usage, registry)
        rows = render_challenge100(registry=registry, eval_path=args.challenge64_eval, holdout_path=args.challenge64_holdout)
        svg_out.parent.mkdir(parents=True, exist_ok=True)
        svg_out.write_text(render_challenge100_svg(token_aligned_rows(rows, registry, usage)), encoding='utf-8')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
