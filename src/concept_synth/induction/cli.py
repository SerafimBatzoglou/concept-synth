"""Command line tools for the public INDUCTION benchmark release."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .benchmark_io import (
    canonical_dataset_paths,
    canonical_predictions_path,
    generated_eval_path,
    get_instance_id,
    infer_task_from_dataset,
    iter_jsonl,
    load_problem_index,
    load_problem_records,
    write_jsonl,
)
from .evaluator import evaluate_prediction
from .generator import generate_task_bundle, validate_generated_records, write_records
from .prompting import build_prompt


def _select_problem(
    problems: Dict[str, Dict[str, Any]],
    instance_id: Optional[str],
    index: int,
) -> Dict[str, Any]:
    if instance_id:
        if instance_id not in problems:
            raise SystemExit(f"Unknown instance id: {instance_id}")
        return problems[instance_id]

    ordered_ids = sorted(problems)
    if index < 0 or index >= len(ordered_ids):
        raise SystemExit(f"--index {index} is out of range for {len(ordered_ids)} instances")
    return problems[ordered_ids[index]]


def _dataset_paths_from_args(args: argparse.Namespace) -> List[Path]:
    if args.dataset:
        return [Path(path) for path in args.dataset]
    canonical = canonical_dataset_paths()
    return [canonical["FullObs"], canonical["CI"], canonical["EC"]]


def command_build_prompt(args: argparse.Namespace) -> int:
    dataset_path = Path(args.dataset)
    problems = load_problem_index(dataset_path)
    record = _select_problem(problems, args.instance_id, args.index)
    prompt, system_prompt = build_prompt(record)

    text = prompt
    if args.include_system and system_prompt:
        text = f"System prompt:\n{system_prompt}\n\nUser prompt:\n{prompt}"

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


def _iter_matching_predictions(
    predictions_path: Path,
    instance_ids: set[str],
    models: Optional[set[str]],
    limit: Optional[int],
) -> Iterable[Dict[str, Any]]:
    count = 0
    for row in iter_jsonl(predictions_path):
        if row.get("instanceId") not in instance_ids:
            continue
        if models and row.get("model") not in models:
            continue
        yield row
        count += 1
        if limit is not None and count >= limit:
            return


def command_evaluate(args: argparse.Namespace) -> int:
    predictions_path = Path(args.predictions or canonical_predictions_path())
    output_path = Path(args.output or generated_eval_path())
    models = set(args.model or []) or None
    total = 0
    append = args.append

    for dataset_path in _dataset_paths_from_args(args):
        problems = load_problem_index(dataset_path)
        instance_ids = set(problems)
        task = infer_task_from_dataset(dataset_path) or "INDUCTION"
        print(f"Evaluating {task}: {dataset_path} ({len(instance_ids)} instances)", file=sys.stderr)

        def rows() -> Iterable[Dict[str, Any]]:
            for prediction in _iter_matching_predictions(
                predictions_path,
                instance_ids,
                models,
                args.limit,
            ):
                record = problems[prediction["instanceId"]]
                yield evaluate_prediction(
                    record,
                    prediction,
                    dataset_path=str(dataset_path),
                    run_id=args.run_id,
                    timeout_ms=args.timeout_ms,
                    compute_diagnostics=not args.no_diagnostics,
                )

        written = write_jsonl(rows(), output_path, append=append)
        total += written
        append = True

    print(f"Wrote {total} rows to {output_path}", file=sys.stderr)
    return 0


def command_summarize(args: argparse.Namespace) -> int:
    rows = list(iter_jsonl(args.eval_cache))
    summary: Dict[str, Dict[str, Dict[str, int]]] = {}
    for row in rows:
        task = row.get("task") or row.get("scenario") or "unknown"
        model = row.get("model_id") or "unknown"
        summary.setdefault(task, {}).setdefault(model, {"valid": 0, "total": 0})
        summary[task][model]["total"] += 1
        if row.get("valid"):
            summary[task][model]["valid"] += 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def command_generate(args: argparse.Namespace) -> int:
    records = generate_task_bundle(
        task=args.task,
        n=args.n,
        seed=args.seed,
        worlds=args.worlds,
        yes_worlds=args.yes_worlds,
        no_worlds=args.no_worlds,
        domain_min=args.domain_min,
        domain_max=args.domain_max,
        unknown_rate=args.unknown_rate,
    )
    if args.validate:
        errors = validate_generated_records(records, timeout_ms=args.timeout_ms)
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            return 2
    write_records(records, args.output)
    print(f"Wrote {len(records)} generated records to {args.output}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Public INDUCTION benchmark tools")
    subparsers = parser.add_subparsers(dest="command")

    prompt_parser = subparsers.add_parser("build-prompt", help="Render a released prompt")
    prompt_parser.add_argument(
        "--dataset",
        default=str(canonical_dataset_paths()["FullObs"]),
        help="Released dataset YAML(.gz)",
    )
    prompt_parser.add_argument("--instance-id", help="Instance id to render")
    prompt_parser.add_argument("--index", type=int, default=0, help="Sorted instance index")
    prompt_parser.add_argument("--output", help="Write prompt to this path instead of stdout")
    prompt_parser.add_argument("--include-system", action="store_true", help="Include system prompt if present")
    prompt_parser.set_defaults(func=command_build_prompt)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate prediction rows")
    eval_parser.add_argument("--dataset", action="append", help="Dataset YAML(.gz); repeatable. Defaults to all canonical datasets.")
    eval_parser.add_argument("--predictions", default=str(canonical_predictions_path()), help="Prediction JSONL(.gz)")
    eval_parser.add_argument("--output", default=str(generated_eval_path()), help="Output eval-cache JSONL")
    eval_parser.add_argument("--append", action="store_true", help="Append to output instead of replacing it")
    eval_parser.add_argument("--model", action="append", help="Restrict to one model; repeatable")
    eval_parser.add_argument("--limit", type=int, help="Maximum matching predictions per dataset")
    eval_parser.add_argument("--run-id", default="public_rebuild", help="Run id stored in output rows")
    eval_parser.add_argument("--timeout-ms", type=int, default=120000, help="Per-world Z3 timeout for EC")
    eval_parser.add_argument("--no-diagnostics", action="store_true", help="Skip expensive EC failure diagnostics")
    eval_parser.set_defaults(func=command_evaluate)

    summary_parser = subparsers.add_parser("summarize", help="Summarize an eval-cache JSONL file")
    summary_parser.add_argument("eval_cache", help="Eval-cache JSONL path")
    summary_parser.set_defaults(func=command_summarize)

    generate_parser = subparsers.add_parser("generate", help="Generate new public-schema INDUCTION instances")
    generate_parser.add_argument("--task", choices=["FullObs", "CI", "EC", "all"], default="FullObs")
    generate_parser.add_argument("--n", type=int, default=10, help="Number of records per selected task")
    generate_parser.add_argument("--seed", type=int, default=0, help="Deterministic RNG seed")
    generate_parser.add_argument("--worlds", type=int, default=4, help="Input worlds for FullObs/EC")
    generate_parser.add_argument("--yes-worlds", type=int, default=6, help="YES worlds for CI")
    generate_parser.add_argument("--no-worlds", type=int, default=2, help="contrastive NO worlds for CI")
    generate_parser.add_argument("--domain-min", type=int, default=5)
    generate_parser.add_argument("--domain-max", type=int, default=8)
    generate_parser.add_argument("--unknown-rate", type=float, default=0.35, help="EC unknown-atom masking rate")
    generate_parser.add_argument("--output", required=True, help="Output YAML or YAML.GZ path")
    generate_parser.add_argument("--validate", action="store_true", help="Check generated records with the public evaluator")
    generate_parser.add_argument("--timeout-ms", type=int, default=30000, help="Validation timeout for EC")
    generate_parser.set_defaults(func=command_generate)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    return args.func(args)


def prompt_main(argv: Optional[List[str]] = None) -> int:
    args = ["build-prompt", *(argv if argv is not None else sys.argv[1:])]
    return main(args)


def evaluate_main(argv: Optional[List[str]] = None) -> int:
    args = ["evaluate", *(argv if argv is not None else sys.argv[1:])]
    return main(args)


def generate_main(argv: Optional[List[str]] = None) -> int:
    args = ["generate", *(argv if argv is not None else sys.argv[1:])]
    return main(args)


if __name__ == "__main__":
    raise SystemExit(main())
