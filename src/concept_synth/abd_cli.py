"""Standalone public CLI for ABD prompt rendering and evaluation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from concept_synth.abd_b1_prompt import build_abd_b1_prompt, get_abd_b1_system_prompt
from concept_synth.benchmark_io import (
    get_instance_id,
    guess_holdout_path,
    iter_embedded_results,
    load_problem_index,
    read_jsonl,
)
from concept_synth.eval_cache import (
    EvalCacheRecord,
    EvalCacheWriter,
    HoldoutEval,
    Prediction,
    Timing,
    TrainEval,
    generate_run_id,
)
from concept_synth.evaluate_abd_b1 import (
    evaluate_abd_b1,
    evaluate_on_holdouts,
    extract_alpha_from_response,
    load_holdouts_from_jsonl,
)
from concept_synth.metrics import ast_size
from concept_synth.sexpr_parser import parse_sexpr_formula


def _select_problem(
    problems: Dict[str, Dict[str, Any]],
    instance_id: Optional[str],
    index: int,
) -> Dict[str, Any]:
    if instance_id:
        try:
            return problems[instance_id]
        except KeyError as exc:
            raise SystemExit(f"Unknown instance id: {instance_id}") from exc

    ordered_ids = sorted(problems)
    if index < 0 or index >= len(ordered_ids):
        raise SystemExit(f"--index {index} is out of range for {len(ordered_ids)} instances")
    return problems[ordered_ids[index]]


def _get_worlds(
    problem: Dict[str, Any],
    holdouts: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    instance_id: Optional[str] = None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    prob_data = problem.get("problem", problem)
    all_worlds = prob_data.get("trainWorlds") or prob_data.get("worlds", [])
    train_worlds = [world for world in all_worlds if not world.get("isHeldout", False)]

    holdout_worlds: List[Dict[str, Any]] = []
    if holdouts and instance_id and instance_id in holdouts:
        holdout_worlds = holdouts[instance_id]
    elif prob_data.get("heldoutWorlds"):
        holdout_worlds = prob_data["heldoutWorlds"]

    return train_worlds, holdout_worlds


def _gold_ast_size(problem: Dict[str, Any]) -> Optional[int]:
    prob_data = problem.get("problem", problem)
    gold = prob_data.get("gold", {})
    formula = gold.get("alpha") or gold.get("goldAlpha")
    if not formula:
        return None
    try:
        return ast_size(parse_sexpr_formula(formula))
    except Exception:
        return None


def _coalesce(*values: Optional[int]) -> Optional[int]:
    for value in values:
        if value is not None:
            return value
    return None


def _evaluate_result_to_record(
    problem: Dict[str, Any],
    result: Dict[str, Any],
    instance_id: str,
    model_id: str,
    run_id: str,
    timeout_ms: int,
    holdouts: Optional[Dict[str, List[Dict[str, Any]]]],
) -> EvalCacheRecord:
    start_time = time.perf_counter()
    prob_data = problem.get("problem", problem)
    desc = problem.get("problemDescription", {})
    gold = prob_data.get("gold", {})

    scenario = prob_data.get("scenario", "ABD_FULL")
    theory = prob_data.get("theoryId") or prob_data.get("theory", {}).get("theoryId", "Unknown")
    difficulty = desc.get("difficulty", "unknown")
    train_worlds, holdout_worlds = _get_worlds(problem, holdouts, instance_id)

    gold_cost_train = _coalesce(gold.get("totalGoldAlphaCost"), gold.get("goldCost"))
    gold_cost_holdout = None
    if holdout_worlds:
        costs = [world.get("goldAlphaCost") for world in holdout_worlds]
        if all(cost is not None for cost in costs):
            gold_cost_holdout = sum(costs)

    parse_start = time.perf_counter()
    alpha_sexpr = result.get("extractedFormula", "")
    parse_error = None

    if alpha_sexpr and not isinstance(alpha_sexpr, str):
        alpha_sexpr = str(alpha_sexpr)
    if not alpha_sexpr:
        response = result.get("response", result.get("rawResponse", ""))
        if response and not isinstance(response, str):
            response = str(response)
        if response:
            alpha_sexpr, parse_error = extract_alpha_from_response(response)
        else:
            parse_error = "No response found in result"

    timing = Timing(parse_ms=(time.perf_counter() - parse_start) * 1000.0)
    raw_text = result.get("response") or result.get("rawResponse", "")
    if raw_text and not isinstance(raw_text, str):
        raw_text = str(raw_text)

    prediction = Prediction(
        raw_text=raw_text[:1000] if raw_text else None,
        formula=alpha_sexpr,
        parse_ok=alpha_sexpr is not None and parse_error is None,
        parse_error=parse_error,
    )
    train_eval = TrainEval()
    holdout_eval = HoldoutEval()
    notes = None

    if alpha_sexpr and not parse_error:
        train_start = time.perf_counter()
        train_result = evaluate_abd_b1(problem, alpha_sexpr, timeout_ms)
        timing.train_eval_ms = (time.perf_counter() - train_start) * 1000.0
        evaluated_formula = train_result.alpha_sexpr or alpha_sexpr

        prediction.formula = evaluated_formula
        prediction.original_formula = train_result.alpha_original_sexpr
        prediction.parse_ok = train_result.parse_error is None
        prediction.parse_error = train_result.parse_error
        prediction.auto_closed_parens = train_result.trailing_parens_added

        if prediction.parse_ok and evaluated_formula:
            try:
                prediction.ast_size = ast_size(parse_sexpr_formula(evaluated_formula))
            except Exception:
                prediction.ast_size = None

        if prediction.auto_closed_parens > 0:
            noun = "parenthesis" if prediction.auto_closed_parens == 1 else "parentheses"
            notes = f"Auto-closed {prediction.auto_closed_parens} trailing {noun}"

        train_eval.train_all_valid = train_result.valid
        if train_result.valid and train_result.total_cost is not None:
            train_eval.pred_cost_train_sum = train_result.total_cost
            if train_result.total_opt_cost is not None:
                train_eval.gap_vs_opt_train_sum = train_result.total_cost - train_result.total_opt_cost
                if train_worlds:
                    train_eval.gap_vs_opt_train_norm = train_eval.gap_vs_opt_train_sum / len(train_worlds)
            if gold_cost_train is not None:
                train_eval.gap_vs_gold_train_sum = train_result.total_cost - gold_cost_train
                if train_worlds:
                    train_eval.gap_vs_gold_train_norm = train_eval.gap_vs_gold_train_sum / len(train_worlds)
        elif train_result.per_world:
            train_eval.train_invalid_worlds = [
                row.get("worldId", str(index))
                for index, row in enumerate(train_result.per_world)
                if not row.get("valid", False)
            ]

        if holdout_worlds and prediction.parse_ok:
            holdout_start = time.perf_counter()
            holdout_result = evaluate_on_holdouts(
                problem,
                evaluated_formula,
                holdout_worlds,
                timeout_ms,
            )
            timing.holdout_eval_ms = (time.perf_counter() - holdout_start) * 1000.0
            holdout_eval.holdout_all_valid = holdout_result.holdout_valid
            if holdout_result.holdout_valid and holdout_result.holdout_total_cost is not None:
                holdout_eval.pred_cost_holdout_sum = holdout_result.holdout_total_cost
                if holdout_result.holdout_total_opt_cost is not None:
                    holdout_eval.gap_vs_opt_holdout_sum = (
                        holdout_result.holdout_total_cost - holdout_result.holdout_total_opt_cost
                    )
                    if holdout_worlds:
                        holdout_eval.gap_vs_opt_holdout_norm = (
                            holdout_eval.gap_vs_opt_holdout_sum / len(holdout_worlds)
                        )
                if gold_cost_holdout is not None:
                    holdout_eval.gap_vs_gold_holdout_sum = holdout_result.holdout_total_cost - gold_cost_holdout
                    if holdout_worlds:
                        holdout_eval.gap_vs_gold_holdout_norm = (
                            holdout_eval.gap_vs_gold_holdout_sum / len(holdout_worlds)
                        )
            elif holdout_result.per_holdout:
                holdout_eval.holdout_invalid_worlds = [
                    row.get("worldId", str(index))
                    for index, row in enumerate(holdout_result.per_holdout)
                    if not row.get("valid", False)
                ]

    timing.total_ms = (time.perf_counter() - start_time) * 1000.0
    return EvalCacheRecord(
        instance_id=instance_id,
        scenario=scenario,
        theory=theory,
        difficulty=difficulty,
        num_train_worlds=len(train_worlds),
        num_holdout_worlds=len(holdout_worlds),
        gold_formula=gold.get("alpha") or gold.get("goldAlpha"),
        gold_ast=_gold_ast_size(problem),
        gold_cost_train_sum=gold_cost_train,
        gold_cost_holdout_sum=gold_cost_holdout,
        opt_cost_train_sum=sum(world.get("optCost", 0) for world in train_worlds),
        opt_cost_holdout_sum=sum(world.get("optCost", 0) for world in holdout_worlds) if holdout_worlds else None,
        model_id=model_id,
        run_id=run_id,
        prediction=prediction,
        train_eval=train_eval,
        holdout_eval=holdout_eval,
        timing=timing,
        notes=notes,
    )


def _load_results(
    problems: Dict[str, Dict[str, Any]],
    predictions_path: Optional[str],
    models: Optional[set[str]],
) -> List[Dict[str, Any]]:
    if predictions_path:
        rows = read_jsonl(predictions_path)
        if models:
            rows = [row for row in rows if row.get("model") in models]
        return rows
    return iter_embedded_results(problems, models=models)


def build_prompt_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="concept-synth-abd-build-prompt",
        description="Render a released ABD prompt from the public benchmark bundle.",
    )
    parser.add_argument(
        "--dataset",
        default="benchmarks/abduction/data/abd_combined_v1.yaml.gz",
        help="Path to the benchmark bundle (.yaml or .yaml.gz).",
    )
    parser.add_argument("--instance-id", help="Benchmark instance id to render.")
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Fallback sorted-record index when --instance-id is omitted.",
    )
    parser.add_argument("--output", help="Write the rendered prompt to this file.")
    parser.add_argument(
        "--include-system-prompt",
        action="store_true",
        help="Emit a JSON object with separate system_prompt and user_prompt fields.",
    )
    return parser


def build_evaluate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="concept-synth-abd-evaluate",
        description="Evaluate embedded or external ABD predictions into eval-cache JSONL.",
    )
    parser.add_argument(
        "--dataset",
        default="benchmarks/abduction/data/abd_combined_v1.yaml.gz",
        help="Path to the benchmark bundle (.yaml or .yaml.gz).",
    )
    parser.add_argument(
        "--holdouts",
        help="Optional holdout sidecar JSONL. Defaults to the adjacent released sidecar when unique.",
    )
    parser.add_argument(
        "--predictions",
        help="Optional JSONL file with instanceId/model/extractedFormula or response fields.",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Filter to one or more model ids. When omitted, evaluate all embedded models.",
    )
    parser.add_argument("--output", required=True, help="Output eval-cache JSONL path.")
    parser.add_argument("--run-id")
    parser.add_argument("--timeout-ms", type=int, default=10000)
    parser.add_argument("--limit", type=int, help="Optional cap on the number of prediction rows.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output JSONL instead of resuming.",
    )
    return parser


def prompt_main(argv: Optional[List[str]] = None) -> int:
    args = build_prompt_parser().parse_args(argv)
    problems = load_problem_index(args.dataset)
    problem = _select_problem(problems, args.instance_id, args.index)
    prompt = build_abd_b1_prompt(problem)

    if args.include_system_prompt:
        rendered = json.dumps(
            {
                "system_prompt": get_abd_b1_system_prompt(),
                "user_prompt": prompt,
            },
            indent=2,
        )
    else:
        rendered = prompt

    if args.output:
        Path(args.output).write_text(rendered + ("\n" if not rendered.endswith("\n") else ""), encoding="utf-8")
    else:
        sys.stdout.write(rendered)
        if not rendered.endswith("\n"):
            sys.stdout.write("\n")
    return 0


def evaluate_main(argv: Optional[List[str]] = None) -> int:
    args = build_evaluate_parser().parse_args(argv)
    if not args.run_id:
        args.run_id = generate_run_id("public_eval")
    problems = load_problem_index(args.dataset)
    models = set(args.models) if args.models else None
    holdout_path = args.holdouts or guess_holdout_path(args.dataset)
    holdouts = load_holdouts_from_jsonl(holdout_path) if holdout_path else {}
    results = _load_results(problems, args.predictions, models)

    if args.limit is not None:
        results = results[: args.limit]

    if not results:
        raise SystemExit("No prediction rows matched the requested inputs.")

    written = 0
    skipped = 0
    seen_models = set()
    with EvalCacheWriter(
        output_path=args.output,
        dataset_path=args.dataset,
        holdout_path=holdout_path,
        run_id=args.run_id,
        resume=not args.overwrite,
    ) as writer:
        for result in results:
            instance_id = result.get("instanceId") or get_instance_id(result)
            if not instance_id:
                raise SystemExit("Prediction row is missing instanceId.")
            if instance_id not in problems:
                raise SystemExit(f"Prediction references unknown instanceId: {instance_id}")

            model_id = result.get("model", "external")
            seen_models.add(model_id)
            if writer.should_skip(instance_id, model_id):
                skipped += 1
                continue

            record = _evaluate_result_to_record(
                problem=problems[instance_id],
                result=result,
                instance_id=instance_id,
                model_id=model_id,
                run_id=args.run_id,
                timeout_ms=args.timeout_ms,
                holdouts=holdouts,
            )
            writer.write_record(record)
            written += 1

        writer.write_metadata(
            models=sorted(seen_models),
            command_line=" ".join(sys.argv),
        )

    print(f"Wrote {written} eval-cache records to {args.output}")
    if skipped:
        print(f"Skipped {skipped} existing instance/model pairs")
    if holdout_path:
        print(f"Using holdouts: {holdout_path}")
    else:
        print("No holdout sidecar found; holdout metrics will be empty")
    return 0


def build_main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="concept-synth-abd")
    parser.add_argument("command", choices=["build-prompt", "evaluate"])
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_main_parser()
    if not argv:
        parser.print_help()
        return 2
    args = parser.parse_args(argv[:1])
    remaining = argv[1:]
    if args.command == "build-prompt":
        return prompt_main(remaining)
    return evaluate_main(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
