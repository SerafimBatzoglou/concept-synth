"""Exact public evaluator for INDUCTION formulas and prediction rows."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

from concept_synth.fol.formulas import FOFormula
from concept_synth.fol.model import FiniteModel
from concept_synth.metrics import ast_size, quantifier_depth
from concept_synth.predicate_format import get_binary_extension, get_unary_extension, parse_binary_pair
from concept_synth.sexpr_parser import parse_sexpr_formula
from concept_synth.sexpr_printer import to_sexpr
from concept_synth.signature_utils import (
    get_allowed_induction_predicates,
    get_world_predicate_arities,
    split_predicates_by_arity,
)
from concept_synth.target import compute_target_extension

from .benchmark_io import dataset_name_from_path
from .prompting import normalize_scenario, normalize_task

try:
    from concept_synth.e_completion_z3 import ESemantics, Z3_AVAILABLE, check_e_scenario
except Exception:  # pragma: no cover - exercised when z3 is unavailable
    ESemantics = None
    Z3_AVAILABLE = False
    check_e_scenario = None


def build_model_from_world(world: Dict[str, Any]) -> FiniteModel:
    domain = world.get("domain", [])
    model = FiniteModel(len(domain), const_names=list(domain))
    const_to_idx = {name: idx for idx, name in enumerate(domain)}
    predicates = world.get("predicates", {})
    predicate_arities = get_world_predicate_arities(world)
    unary_preds, binary_preds = split_predicates_by_arity(predicate_arities)

    for pred_name in unary_preds:
        for const in get_unary_extension(predicates.get(pred_name)):
            if const in const_to_idx:
                model.set_unary(pred_name, const_to_idx[const], True)

    for pred_name in binary_preds:
        for pair in get_binary_extension(predicates.get(pred_name)):
            try:
                left, right = parse_binary_pair(pair)
            except ValueError:
                continue
            if left in const_to_idx and right in const_to_idx:
                model.set_binary(pred_name, const_to_idx[left], const_to_idx[right], True)

    return model


def match_world(world: Dict[str, Any], formula: FOFormula) -> Tuple[bool, Optional[str], Optional[str]]:
    model = build_model_from_world(world)
    target = world.get("targetExtension", {})
    if not target:
        return False, f"World {world.get('worldId', 'unknown')} has no targetExtension", None

    target_true = set(target.get("T_true", []))
    computed_true = set(compute_target_extension(model, formula).T_true)
    if computed_true == target_true:
        return True, None, None

    false_positives = sorted(computed_true - target_true)
    false_negatives = sorted(target_true - computed_true)
    if false_positives:
        elem = false_positives[0]
        return False, f"phi({elem})=TRUE but target is FALSE", elem
    if false_negatives:
        elem = false_negatives[0]
        return False, f"phi({elem})=FALSE but target is TRUE", elem
    return False, "Unknown mismatch", None


def _train_worlds(problem: Dict[str, Any]) -> List[Dict[str, Any]]:
    task = problem.get("task", {})
    train_ids = set(task.get("trainWorldIds", []))
    worlds = problem.get("worlds", [])
    if not train_ids:
        return worlds
    return [world for world in worlds if world.get("worldId") in train_ids]


def _ci_worlds(problem: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    task = problem.get("task", {})
    worlds = problem.get("worlds", [])
    yes_ids = set(task.get("yesWorldIds", []))
    no_ids = set(task.get("noWorldIds", []))
    yes_worlds = [world for world in worlds if world.get("worldId") in yes_ids]
    no_worlds = [world for world in worlds if world.get("worldId") in no_ids]

    if not yes_worlds and not no_worlds:
        yes_worlds = [world for world in worlds if world.get("splitLabel") is True]
        no_worlds = [world for world in worlds if world.get("splitLabel") is False]
    return yes_worlds, no_worlds


def evaluate_fullobs(problem: Dict[str, Any], formula: FOFormula) -> Dict[str, Any]:
    worlds = _train_worlds(problem)
    for world in worlds:
        ok, explanation, elem = match_world(world, formula)
        if not ok:
            return {
                "correct": False,
                "failed_world": world.get("worldId"),
                "failed_elements": [elem] if elem else None,
                "failure_explanation": explanation,
                "num_worlds": len(worlds),
            }
    return {
        "correct": True,
        "failed_world": None,
        "failed_elements": None,
        "failure_explanation": None,
        "num_worlds": len(worlds),
    }


def evaluate_ci(problem: Dict[str, Any], formula: FOFormula) -> Dict[str, Any]:
    yes_worlds, no_worlds = _ci_worlds(problem)
    if not yes_worlds or not no_worlds:
        return {
            "correct": False,
            "failed_world": None,
            "failed_elements": None,
            "failure_explanation": "CI requires at least one YES world and one contrastive NO world",
            "c_yes_count": len(yes_worlds),
            "c_no_count": len(no_worlds),
            "c_failed_world_type": None,
            "c_counterexample_element": None,
        }

    for world in yes_worlds:
        ok, explanation, elem = match_world(world, formula)
        if not ok:
            return {
                "correct": False,
                "failed_world": world.get("worldId"),
                "failed_elements": [elem] if elem else None,
                "failure_explanation": f"YES world {world.get('worldId')}: {explanation}",
                "c_yes_count": len(yes_worlds),
                "c_no_count": len(no_worlds),
                "c_failed_world_type": "YES",
                "c_counterexample_element": elem,
            }

    for world in no_worlds:
        ok, _explanation, _elem = match_world(world, formula)
        if ok:
            return {
                "correct": False,
                "failed_world": world.get("worldId"),
                "failed_elements": None,
                "failure_explanation": f"NO world {world.get('worldId')}: formula incorrectly matches exactly",
                "c_yes_count": len(yes_worlds),
                "c_no_count": len(no_worlds),
                "c_failed_world_type": "NO",
                "c_counterexample_element": None,
            }

    return {
        "correct": True,
        "failed_world": None,
        "failed_elements": None,
        "failure_explanation": None,
        "c_yes_count": len(yes_worlds),
        "c_no_count": len(no_worlds),
        "c_failed_world_type": None,
        "c_counterexample_element": None,
    }


def evaluate_ec(
    problem: Dict[str, Any],
    formula: FOFormula,
    *,
    timeout_ms: int = 120000,
    compute_diagnostics: bool = True,
) -> Dict[str, Any]:
    worlds = _train_worlds(problem)
    if not Z3_AVAILABLE or check_e_scenario is None or ESemantics is None:
        return {
            "correct": False,
            "failed_world": None,
            "failed_elements": None,
            "failure_explanation": "z3-solver is required for exact EC evaluation",
            "e_semantics_used": "exact_exists",
            "z3_status": "unavailable",
        }

    try:
        ok, meta = check_e_scenario(
            worlds,
            formula,
            ESemantics.EXACT_EXISTS,
            timeout_ms=timeout_ms,
            compute_diagnostics=compute_diagnostics,
        )
    except ValueError as exc:
        return {
            "correct": False,
            "failed_world": None,
            "failed_elements": None,
            "failure_explanation": f"Formula error: {exc}",
            "e_semantics_used": "exact_exists",
            "z3_status": "error",
        }

    explanation = None
    if not ok:
        explanation = meta.get("failure_reason", "No valid completion exists")
        if meta.get("failure_classification"):
            explanation = f"{explanation} ({meta['failure_classification']})"

    return {
        "correct": bool(ok),
        "failed_world": meta.get("first_failing_world"),
        "failed_elements": None,
        "failure_explanation": explanation,
        "e_semantics_used": meta.get("e_semantics_used", "exact_exists"),
        "e_sat": meta.get("e_sat"),
        "e_failure_classification": meta.get("failure_classification"),
        "e_local_unsat_elements": meta.get("local_unsat_elements"),
        "e_completion_sensitive": meta.get("completion_sensitive"),
        "e_sensitive_elements_count": meta.get("sensitive_elements_count"),
        "e_pos_unachievable_count": meta.get("pos_unachievable_count"),
        "e_neg_unachievable_count": meta.get("neg_unachievable_count"),
        "e_pos_unachievable_elements": meta.get("pos_unachievable_elements"),
        "e_neg_unachievable_elements": meta.get("neg_unachievable_elements"),
        "e_flips_f_to_t": meta.get("flips_f_to_t"),
        "e_flips_t_to_f": meta.get("flips_t_to_f"),
        "e_nonmonotone": meta.get("nonmonotone"),
        "e_num_pos_labels": meta.get("num_pos_labels"),
        "e_num_neg_labels": meta.get("num_neg_labels"),
        "e_num_local_unsat": meta.get("num_local_unsat"),
        "z3_status": "exact_exists",
    }


def _extract_json_formula(text: str) -> Tuple[Optional[str], Optional[str]]:
    candidates: List[str] = []
    stripped = text.strip()
    if stripped:
        candidates.append(stripped)
    for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE):
        candidates.append(match.group(1).strip())

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        if isinstance(payload, dict):
            formula = payload.get("formula")
            description = payload.get("description")
            if isinstance(formula, str):
                return formula, description if isinstance(description, str) else None
    return None, None


def _balanced_parenthetical_spans(text: str) -> Iterable[str]:
    starts: List[int] = []
    for idx, char in enumerate(text):
        if char == "(":
            starts.append(idx)
        elif char == ")" and starts:
            start = starts.pop()
            if not starts:
                yield text[start : idx + 1]


def extract_formula_from_prediction(
    prediction: Dict[str, Any],
    allowed_predicates: Optional[set[str]] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return formula, description, and parse error from a prediction row."""
    for key in ("extractedFormula", "formula"):
        value = prediction.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip(), prediction.get("extractedDescription") or prediction.get("description"), None

    for key in ("response", "rawResponse"):
        text = prediction.get(key)
        if not isinstance(text, str) or not text.strip():
            continue

        formula, description = _extract_json_formula(text)
        if formula:
            return formula.strip(), description, None

        last_error = None
        for candidate in _balanced_parenthetical_spans(text):
            try:
                parse_sexpr_formula(candidate, allowed_predicates)
                return candidate.strip(), None, None
            except Exception as exc:
                last_error = str(exc)
        if last_error:
            return None, None, last_error

    existing_error = prediction.get("parseError")
    return None, None, str(existing_error) if existing_error else "No formula found"


def _gold_info(problem: Dict[str, Any], problem_desc: Dict[str, Any]) -> Tuple[str, Optional[int]]:
    hidden = problem_desc.get("hiddenTarget", {})
    formula = hidden.get("formula") or problem_desc.get("gold_formula") or problem_desc.get("goldFormula") or ""
    ast = hidden.get("astSize") or problem_desc.get("gold_ast") or problem_desc.get("goldAst")
    if ast is None and formula:
        try:
            ast = ast_size(parse_sexpr_formula(formula, get_allowed_induction_predicates(problem)))
        except Exception:
            ast = None
    return formula, ast


def _metadata(problem_desc: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "benchmark_name",
        "benchmark_version",
        "gold_family_id",
        "gold_subfamily_key",
        "gold_is_lift_hard",
        "ad_band",
        "c_band",
        "e_band",
    ]
    metadata = {key: problem_desc[key] for key in keys if key in problem_desc}
    band = problem_desc.get("ad_band") or problem_desc.get("c_band") or problem_desc.get("e_band")
    if band is not None:
        metadata["band"] = band
    return metadata


def evaluate_prediction(
    record: Dict[str, Any],
    prediction: Dict[str, Any],
    *,
    dataset_path: Optional[str] = None,
    run_id: str = "public_rebuild",
    timeout_ms: int = 120000,
    compute_diagnostics: bool = True,
) -> Dict[str, Any]:
    """Evaluate one prediction row against one released benchmark record."""
    started = time.perf_counter()
    problem = record.get("problem", record)
    problem_desc = record.get("problemDescription", {})
    task = normalize_task(record)
    scenario = normalize_scenario(record)
    instance_id = record.get("instanceId") or problem.get("instanceId") or prediction.get("instanceId")
    model_id = prediction.get("model") or prediction.get("modelId") or "unknown"
    allowed_predicates = get_allowed_induction_predicates(problem)
    formula_text, description, extraction_error = extract_formula_from_prediction(prediction, allowed_predicates)
    gold_formula, gold_ast = _gold_info(problem, problem_desc)

    parsed_formula = None
    parsed_formula_text = None
    parse_error = extraction_error
    formula_ast = None
    formula_qd = None
    formula_parsed = False

    if formula_text:
        try:
            parsed_formula = parse_sexpr_formula(formula_text, allowed_predicates)
            free_vars = parsed_formula.free_vars()
            if free_vars != {"x"}:
                raise ValueError(f"Formula must have exactly one free variable x; found {sorted(free_vars)}")
            parsed_formula_text = to_sexpr(parsed_formula)
            formula_ast = ast_size(parsed_formula)
            formula_qd = quantifier_depth(parsed_formula)
            formula_parsed = True
            parse_error = None
        except Exception as exc:
            parse_error = str(exc)

    task_eval: Dict[str, Any]
    if not formula_parsed or parsed_formula is None:
        task_eval = {
            "correct": False,
            "failed_world": None,
            "failed_elements": None,
            "failure_explanation": f"Parse error: {parse_error}" if parse_error else "No formula parsed",
        }
    else:
        try:
            if scenario == "AD":
                task_eval = evaluate_fullobs(problem, parsed_formula)
            elif scenario == "C":
                task_eval = evaluate_ci(problem, parsed_formula)
            elif scenario == "E":
                task_eval = evaluate_ec(
                    problem,
                    parsed_formula,
                    timeout_ms=timeout_ms,
                    compute_diagnostics=compute_diagnostics,
                )
            else:
                task_eval = {
                    "correct": False,
                    "failed_world": None,
                    "failed_elements": None,
                    "failure_explanation": f"Unsupported scenario: {scenario}",
                }
        except Exception as exc:
            task_eval = {
                "correct": False,
                "failed_world": None,
                "failed_elements": None,
                "failure_explanation": f"Formula error: {exc}",
            }

    if formula_ast is not None and gold_ast is not None:
        ast_delta = formula_ast - int(gold_ast)
        if formula_ast < int(gold_ast):
            complexity = "simpler"
        elif formula_ast == int(gold_ast):
            complexity = "equal"
        else:
            complexity = "more_complex"
    else:
        ast_delta = None
        complexity = "unknown"

    evaluation = {
        "correctFormula": bool(task_eval.get("correct")),
        "formulaParsed": formula_parsed,
        "parsedFormula": parsed_formula_text or formula_text,
        "targetFormula": gold_formula,
        "llmAstSize": formula_ast,
        "llmQuantifierDepth": formula_qd,
        "targetAstSize": gold_ast,
        "complexityComparison": complexity,
        "failureExplanation": task_eval.get("failure_explanation"),
        "failedWorld": task_eval.get("failed_world"),
        "failedElements": task_eval.get("failed_elements"),
        "cYesCount": task_eval.get("c_yes_count"),
        "cNoCount": task_eval.get("c_no_count"),
        "cFailedWorldType": task_eval.get("c_failed_world_type"),
        "cCounterexampleElement": task_eval.get("c_counterexample_element"),
        "e_semantics_used": task_eval.get("e_semantics_used"),
        "eSat": task_eval.get("e_sat"),
        "eFailureClassification": task_eval.get("e_failure_classification"),
        "eLocalUnsatElements": task_eval.get("e_local_unsat_elements"),
        "eCompletionSensitive": task_eval.get("e_completion_sensitive"),
        "eSensitiveElementsCount": task_eval.get("e_sensitive_elements_count"),
        "ePosUnachievableCount": task_eval.get("e_pos_unachievable_count"),
        "eNegUnachievableCount": task_eval.get("e_neg_unachievable_count"),
        "ePosUnachievableElements": task_eval.get("e_pos_unachievable_elements"),
        "eNegUnachievableElements": task_eval.get("e_neg_unachievable_elements"),
        "eFlipsFtoT": task_eval.get("e_flips_f_to_t"),
        "eFlipsTtoF": task_eval.get("e_flips_t_to_f"),
        "eNonmonotone": task_eval.get("e_nonmonotone"),
        "eNumPosLabels": task_eval.get("e_num_pos_labels"),
        "eNumNegLabels": task_eval.get("e_num_neg_labels"),
        "eNumLocalUnsat": task_eval.get("e_num_local_unsat"),
        "z3Status": task_eval.get("z3_status", ""),
        "evalMs": (time.perf_counter() - started) * 1000.0,
    }

    eval_record = {
        "schema_version": "induction_eval_v1",
        "dataset_name": dataset_name_from_path(dataset_path) if dataset_path else None,
        "dataset_path": dataset_path,
        "instance_id": instance_id,
        "task": task,
        "scenario": scenario,
        "model_id": model_id,
        "run_id": run_id,
        "completed": prediction.get("success") is not False,
        "valid": bool(task_eval.get("correct")),
        "parse_ok": formula_parsed,
        "prediction": {
            "formula": formula_text,
            "description": description,
            "parse_error": parse_error,
            "ast_size": formula_ast,
            "quantifier_depth": formula_qd,
        },
        "gold_formula": gold_formula,
        "gold_ast": gold_ast,
        "ast_delta": ast_delta,
        "gold_plus_0": bool(task_eval.get("correct")) and ast_delta is not None and ast_delta <= 0,
        "gold_plus_10": bool(task_eval.get("correct")) and ast_delta is not None and ast_delta <= 10,
        "gold_plus_25": bool(task_eval.get("correct")) and ast_delta is not None and ast_delta <= 25,
        "evaluation": evaluation,
        "metadata": _metadata(problem_desc),
    }
    return eval_record
