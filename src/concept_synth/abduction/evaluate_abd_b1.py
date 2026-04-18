"""
evaluate_abd_b1.py - Evaluation for ABD-B1 Abduction Tasks

Evaluates LLM-generated alpha formulas for ABD-Full, ABD-Partial, and ABD-Skeptical scenarios.

Evaluation metrics:
- validity: whether alpha satisfies axioms in all worlds
- cost: total number of abnormal elements
  - ABD_FULL / ABD_PARTIAL: best-case cost
  - ABD_SKEPTICAL: worst-case cost
- gap: cost - optimal cost
- per-world breakdown
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    from concept_synth.bootstrap import add_repo_root
except ModuleNotFoundError:
    import os as _os
    import sys as _sys

    _path = _os.path.abspath(__file__)
    while True:
        parent = _os.path.dirname(_path)
        if _os.path.basename(_path) == "concept_synth":
            if parent not in _sys.path:
                _sys.path.insert(0, parent)
            break
        if parent == _path:
            break
        _path = parent
    from concept_synth.bootstrap import add_repo_root
add_repo_root(__file__)
from .abd_formula_utils import get_used_predicates
from .abd_b1_theory_library import TheorySpec, get_theory
from .abd_b1_z3_checker import (
    AbdValidityResult,
    check_abd_full_all_worlds,
    check_abd_full_validity,
    check_abd_partial_all_worlds,
    check_abd_partial_validity,
    check_abd_skeptical_all_worlds,
    check_abd_skeptical_validity,
    compute_abd_full_gold_opt_cost,
    compute_abd_partial_alpha_best_cost,
    compute_abd_partial_gold_opt_cost,
    compute_abd_skeptical_alpha_worst_cost,
    parse_axioms,
)
from concept_synth.sexpr_parser import SExprParseError, parse_sexpr_formula
from concept_synth.sexpr_printer import to_sexpr

# =============================================================================
# Result Data Structures
# =============================================================================


@dataclass
class AbdEvalResult:
    """Result of evaluating an alpha formula."""

    valid: bool
    parse_error: Optional[str] = None

    # Per-world results
    per_world: Optional[List[Dict[str, Any]]] = None

    # Aggregate metrics
    total_cost: Optional[int] = None
    total_opt_cost: Optional[int] = None
    total_gap: Optional[int] = None
    avg_gap: Optional[float] = None

    # Comparison to gold
    gold_alpha: Optional[str] = None
    gold_total_cost: Optional[int] = None
    cost_vs_gold: Optional[int] = None  # total_cost - gold_total_cost

    # Formula info
    alpha_sexpr: Optional[str] = None
    alpha_original_sexpr: Optional[str] = None
    alpha_normalized: Optional[str] = None
    trailing_parens_added: int = 0

    # Predicate scoping violation
    forbidden_preds_used: Optional[List[str]] = None


@dataclass(frozen=True)
class ParsedAlphaFormula:
    """Parsed alpha formula, with optional trailing-paren repair metadata."""

    ast: Any
    alpha_sexpr: str
    alpha_normalized: str
    alpha_original_sexpr: Optional[str] = None
    trailing_parens_added: int = 0


# =============================================================================
# Predicate Scoping Validation
# =============================================================================


ALLOWED_ALPHA_PREDICATES = {"P", "Q", "R", "S", "Ab"}


def _count_missing_trailing_parens(alpha_sexpr: str) -> int:
    """Count how many trailing ')' would be needed to balance the formula."""
    balance = 0
    for ch in alpha_sexpr:
        if ch == "(":
            balance += 1
        elif ch == ")":
            balance -= 1
            if balance < 0:
                return 0
    return balance if balance > 0 else 0


def parse_alpha_formula_with_suffix_repair(
    alpha_sexpr: str,
    allowed_predicates=ALLOWED_ALPHA_PREDICATES,
) -> ParsedAlphaFormula:
    """Parse an alpha formula, optionally auto-closing missing trailing parens."""
    normalized_input = alpha_sexpr if isinstance(alpha_sexpr, str) else str(alpha_sexpr)

    try:
        alpha_ast = parse_sexpr_formula(
            normalized_input,
            allowed_predicates=allowed_predicates,
        )
        return ParsedAlphaFormula(
            ast=alpha_ast,
            alpha_sexpr=normalized_input,
            alpha_normalized=to_sexpr(alpha_ast),
        )
    except SExprParseError as e:
        error_text = str(e).lower()
        missing_closers = _count_missing_trailing_parens(normalized_input)
        if missing_closers <= 0 or "end of input" not in error_text:
            raise

        repaired = normalized_input.rstrip() + (")" * missing_closers)
        alpha_ast = parse_sexpr_formula(
            repaired,
            allowed_predicates=allowed_predicates,
        )
        return ParsedAlphaFormula(
            ast=alpha_ast,
            alpha_sexpr=repaired,
            alpha_normalized=to_sexpr(alpha_ast),
            alpha_original_sexpr=normalized_input,
            trailing_parens_added=missing_closers,
        )


def _parse_alpha_for_evaluation(alpha_sexpr: str) -> Tuple[Optional[ParsedAlphaFormula], Optional[AbdEvalResult]]:
    """Parse alpha with trailing-paren repair, returning an eval result on failure."""
    try:
        return parse_alpha_formula_with_suffix_repair(alpha_sexpr), None
    except SExprParseError as e:
        return None, AbdEvalResult(valid=False, parse_error=str(e), alpha_sexpr=alpha_sexpr)
    except Exception as e:
        return None, AbdEvalResult(
            valid=False,
            parse_error=f"Parse error: {e}",
            alpha_sexpr=alpha_sexpr,
        )


def _alpha_result_kwargs(parsed_alpha: ParsedAlphaFormula) -> Dict[str, Any]:
    """Shared alpha metadata stored with evaluation results."""
    return {
        "alpha_sexpr": parsed_alpha.alpha_sexpr,
        "alpha_original_sexpr": parsed_alpha.alpha_original_sexpr,
        "alpha_normalized": parsed_alpha.alpha_normalized,
        "trailing_parens_added": parsed_alpha.trailing_parens_added,
    }


def validate_alpha_predicate_scoping(
    alpha_ast, theory_id: str
) -> Tuple[bool, Optional[str], Optional[List[str]]]:
    """
    Validate that an alpha formula respects the theory's predicate scoping.

    Args:
        alpha_ast: The parsed alpha formula AST
        theory_id: The theory ID to get predicate constraints from

    Returns:
        Tuple of (is_valid, error_message, forbidden_preds_used)
    """
    try:
        theory = get_theory(theory_id)
    except KeyError:
        # Unknown theory - allow all predicates (fallback)
        return True, None, None

    used_preds = get_used_predicates(alpha_ast)
    allowed = theory.get_effective_allowed_preds()
    forbidden = theory.get_forbidden_preds()

    # Check for Ab predicate (always forbidden)
    if "Ab" in used_preds:
        return False, "Formula uses forbidden predicate 'Ab' (circular definition)", ["Ab"]

    # Check for forbidden predicates
    forbidden_used = used_preds & forbidden
    if forbidden_used:
        return (
            False,
            f"Formula uses forbidden predicate(s): {sorted(forbidden_used)}",
            sorted(forbidden_used),
        )

    # Check that all used predicates are in allowed set
    disallowed = used_preds - allowed
    if disallowed:
        return (
            False,
            f"Formula uses disallowed predicate(s): {sorted(disallowed)} (allowed: {sorted(allowed)})",
            sorted(disallowed),
        )

    return True, None, None


# =============================================================================
# Formula Extraction
# =============================================================================


def extract_alpha_from_response(response: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract alpha formula from LLM response.

    Looks for JSON with "formula" field.

    Returns:
        Tuple of (formula_string, error_message)
    """
    import json
    import re

    # Try to find JSON in response
    # Look for {...} pattern
    json_match = re.search(r'\{[^{}]*"formula"[^{}]*\}', response, re.DOTALL)

    if json_match:
        try:
            data = json.loads(json_match.group())
            formula = data.get("formula", "")
            if formula:
                return formula, None
        except json.JSONDecodeError:
            pass

    # Try to find formula directly in S-expr format
    sexpr_match = re.search(r"\((?:and|or|not|exists|forall|P|Q|R|S)[^)]*\)", response)
    if sexpr_match:
        return sexpr_match.group(), None

    return None, "Could not extract formula from response"


# =============================================================================
# Evaluation Functions
# =============================================================================


def evaluate_abd_full(
    problem: Dict[str, Any],
    alpha_sexpr: str,
    timeout_ms: int = 5000,
    enforce_predicate_scoping: bool = True,
) -> AbdEvalResult:
    """
    Evaluate an alpha formula for ABD-Full scenario.

    Args:
        problem: Problem dictionary
        alpha_sexpr: Alpha formula in S-expression format
        timeout_ms: Z3 timeout
        enforce_predicate_scoping: If True, reject formulas using forbidden predicates

    Returns:
        AbdEvalResult with evaluation metrics
    """
    # Parse alpha
    parsed_alpha, parse_result = _parse_alpha_for_evaluation(alpha_sexpr)
    if parse_result is not None:
        return parse_result
    assert parsed_alpha is not None
    alpha_ast = parsed_alpha.ast

    # Get problem data (handle both old and new schema)
    prob_data = problem.get("problem", problem)

    # Handle axioms - may be in 'theory.axioms' (old) or 'axioms' (new)
    theory = prob_data.get("theory", {})
    axioms = theory.get("axioms", []) or prob_data.get("axioms", [])

    # Handle worlds - may be in 'worlds' (old) or 'trainWorlds' (new)
    worlds = prob_data.get("worlds", []) or prob_data.get("trainWorlds", [])

    gold = prob_data.get("gold", {})

    # Enforce predicate scoping if enabled
    if enforce_predicate_scoping:
        # Handle theoryId - may be in 'theory.theoryId' (old) or 'theoryId' (new)
        theory_id = theory.get("theoryId", "") or prob_data.get("theoryId", "")
        if theory_id:
            scoping_valid, scoping_error, forbidden_used = validate_alpha_predicate_scoping(
                alpha_ast, theory_id
            )
            if not scoping_valid:
                return AbdEvalResult(
                    valid=False,
                    parse_error=scoping_error,
                    **_alpha_result_kwargs(parsed_alpha),
                    forbidden_preds_used=forbidden_used,
                )

    # Parse axioms
    try:
        axioms_ast = parse_axioms(axioms)
    except Exception as e:
        return AbdEvalResult(
            valid=False,
            parse_error=f"Axiom parse error: {e}",
            **_alpha_result_kwargs(parsed_alpha),
        )

    # Get training worlds only
    train_worlds = [w for w in worlds if not w.get("isHeldout", False)]

    # Check validity on all training worlds
    try:
        all_valid, per_world_results, total_cost = check_abd_full_all_worlds(
            train_worlds, axioms_ast, alpha_ast, timeout_ms
        )
    except ValueError as e:
        # Catch errors like "Unbound variable" from malformed formulas
        return AbdEvalResult(
            valid=False,
            parse_error=f"Formula evaluation error: {e}",
            **_alpha_result_kwargs(parsed_alpha),
        )
    except Exception as e:
        return AbdEvalResult(
            valid=False,
            parse_error=f"Evaluation error: {type(e).__name__}: {e}",
            **_alpha_result_kwargs(parsed_alpha),
        )

    # Build per-world breakdown
    per_world = []
    for i, (world, result) in enumerate(zip(train_worlds, per_world_results)):
        world_id = world.get("worldId", f"W{i}")
        opt_cost = world.get("optCost", 0)

        per_world.append(
            {
                "worldId": world_id,
                "valid": result.valid,
                "cost": result.cost if result.valid else None,
                "optCost": opt_cost,
                "gap": (
                    (result.cost - opt_cost) if result.valid and result.cost is not None else None
                ),
                "reason": result.reason if not result.valid else None,
            }
        )

    # Compute aggregates
    total_opt_cost = sum(w.get("optCost", 0) for w in train_worlds)
    total_gap = (total_cost - total_opt_cost) if all_valid else None
    avg_gap = (total_gap / len(train_worlds)) if all_valid and len(train_worlds) > 0 else None

    # Compare to gold
    gold_alpha = gold.get("alpha", "")
    gold_total_cost = gold.get("totalGoldAlphaCost", 0)
    cost_vs_gold = (total_cost - gold_total_cost) if all_valid else None

    return AbdEvalResult(
        valid=all_valid,
        per_world=per_world,
        total_cost=total_cost if all_valid else None,
        total_opt_cost=total_opt_cost,
        total_gap=total_gap,
        avg_gap=avg_gap,
        gold_alpha=gold_alpha,
        gold_total_cost=gold_total_cost,
        cost_vs_gold=cost_vs_gold,
        **_alpha_result_kwargs(parsed_alpha),
    )


def evaluate_abd_partial(
    problem: Dict[str, Any],
    alpha_sexpr: str,
    timeout_ms: int = 5000,
    enforce_predicate_scoping: bool = True,
) -> AbdEvalResult:
    """
    Evaluate an alpha formula for ABD-Partial scenario.

    Args:
        problem: Problem dictionary
        alpha_sexpr: Alpha formula in S-expression format
        timeout_ms: Z3 timeout
        enforce_predicate_scoping: If True, reject formulas using forbidden predicates

    Returns:
        AbdEvalResult with evaluation metrics
    """
    # Parse alpha
    parsed_alpha, parse_result = _parse_alpha_for_evaluation(alpha_sexpr)
    if parse_result is not None:
        return parse_result
    assert parsed_alpha is not None
    alpha_ast = parsed_alpha.ast

    # Get problem data (handle both old and new schema)
    prob_data = problem.get("problem", problem)

    # Handle axioms - may be in 'theory.axioms' (old) or 'axioms' (new)
    theory = prob_data.get("theory", {})
    axioms = theory.get("axioms", []) or prob_data.get("axioms", [])

    # Handle worlds - may be in 'worlds' (old) or 'trainWorlds' (new)
    worlds = prob_data.get("worlds", []) or prob_data.get("trainWorlds", [])

    gold = prob_data.get("gold", {})

    # Enforce predicate scoping if enabled
    if enforce_predicate_scoping:
        # Handle theoryId - may be in 'theory.theoryId' (old) or 'theoryId' (new)
        theory_id = theory.get("theoryId", "") or prob_data.get("theoryId", "")
        if theory_id:
            scoping_valid, scoping_error, forbidden_used = validate_alpha_predicate_scoping(
                alpha_ast, theory_id
            )
            if not scoping_valid:
                return AbdEvalResult(
                    valid=False,
                    parse_error=scoping_error,
                    **_alpha_result_kwargs(parsed_alpha),
                    forbidden_preds_used=forbidden_used,
                )

    # Parse axioms
    try:
        axioms_ast = parse_axioms(axioms)
    except Exception as e:
        return AbdEvalResult(
            valid=False,
            parse_error=f"Axiom parse error: {e}",
            **_alpha_result_kwargs(parsed_alpha),
        )

    # Get training worlds only
    train_worlds = [w for w in worlds if not w.get("isHeldout", False)]

    # Check validity on all training worlds
    try:
        all_valid, per_world_results, total_cost = check_abd_partial_all_worlds(
            train_worlds, axioms_ast, alpha_ast, timeout_ms
        )
    except ValueError as e:
        # Catch errors like "Unbound variable" from malformed formulas
        return AbdEvalResult(
            valid=False,
            parse_error=f"Formula evaluation error: {e}",
            **_alpha_result_kwargs(parsed_alpha),
        )
    except Exception as e:
        return AbdEvalResult(
            valid=False,
            parse_error=f"Evaluation error: {type(e).__name__}: {e}",
            **_alpha_result_kwargs(parsed_alpha),
        )

    # Build per-world breakdown
    per_world = []
    for i, (world, result) in enumerate(zip(train_worlds, per_world_results)):
        world_id = world.get("worldId", f"W{i}")
        opt_cost = world.get("optCost", 0)

        per_world.append(
            {
                "worldId": world_id,
                "valid": result.valid,
                "cost": result.cost if result.valid else None,
                "optCost": opt_cost,
                "gap": (
                    (result.cost - opt_cost) if result.valid and result.cost is not None else None
                ),
                "reason": result.reason if not result.valid else None,
            }
        )

    # Compute aggregates
    total_opt_cost = sum(w.get("optCost", 0) for w in train_worlds)
    total_gap = (total_cost - total_opt_cost) if all_valid else None
    avg_gap = (total_gap / len(train_worlds)) if all_valid and len(train_worlds) > 0 else None

    # Compare to gold
    gold_alpha = gold.get("alpha", "")
    gold_total_cost = gold.get("totalGoldAlphaCost", 0)
    cost_vs_gold = (total_cost - gold_total_cost) if all_valid else None

    return AbdEvalResult(
        valid=all_valid,
        per_world=per_world,
        total_cost=total_cost if all_valid else None,
        total_opt_cost=total_opt_cost,
        total_gap=total_gap,
        avg_gap=avg_gap,
        gold_alpha=gold_alpha,
        gold_total_cost=gold_total_cost,
        cost_vs_gold=cost_vs_gold,
        **_alpha_result_kwargs(parsed_alpha),
    )


def evaluate_abd_skeptical(
    problem: Dict[str, Any],
    alpha_sexpr: str,
    timeout_ms: int = 5000,
    enforce_predicate_scoping: bool = True,
) -> AbdEvalResult:
    """
    Evaluate an alpha formula for ABD-Skeptical scenario.

    ABD-Skeptical uses universal/forall completion:
    - Alpha is valid iff for ALL completions of unknown atoms, the axioms hold
    - Cost is the WORST-CASE (maximum) number of abnormal objects

    Args:
        problem: Problem dictionary
        alpha_sexpr: Alpha formula in S-expression format
        timeout_ms: Z3 timeout
        enforce_predicate_scoping: If True, reject formulas using forbidden predicates

    Returns:
        AbdEvalResult with evaluation metrics
    """
    # Parse alpha
    parsed_alpha, parse_result = _parse_alpha_for_evaluation(alpha_sexpr)
    if parse_result is not None:
        return parse_result
    assert parsed_alpha is not None
    alpha_ast = parsed_alpha.ast

    # Get problem data (handle both old and new schema)
    prob_data = problem.get("problem", problem)

    # Handle axioms - may be in 'theory.axioms' (old) or 'axioms' (new)
    theory = prob_data.get("theory", {})
    axioms = theory.get("axioms", []) or prob_data.get("axioms", [])

    # Handle worlds - may be in 'worlds' (old) or 'trainWorlds' (new)
    worlds = prob_data.get("worlds", []) or prob_data.get("trainWorlds", [])

    gold = prob_data.get("gold", {})

    # Enforce predicate scoping if enabled
    if enforce_predicate_scoping:
        # Handle theoryId - may be in 'theory.theoryId' (old) or 'theoryId' (new)
        theory_id = theory.get("theoryId", "") or prob_data.get("theoryId", "")
        if theory_id:
            scoping_valid, scoping_error, forbidden_used = validate_alpha_predicate_scoping(
                alpha_ast, theory_id
            )
            if not scoping_valid:
                return AbdEvalResult(
                    valid=False,
                    parse_error=scoping_error,
                    **_alpha_result_kwargs(parsed_alpha),
                    forbidden_preds_used=forbidden_used,
                )

    # Parse axioms
    try:
        axioms_ast = parse_axioms(axioms)
    except Exception as e:
        return AbdEvalResult(
            valid=False,
            parse_error=f"Axiom parse error: {e}",
            **_alpha_result_kwargs(parsed_alpha),
        )

    # Get training worlds only
    train_worlds = [w for w in worlds if not w.get("isHeldout", False)]

    # Check validity on all training worlds (skeptical semantics)
    try:
        all_valid, per_world_results, total_worst_cost = check_abd_skeptical_all_worlds(
            train_worlds, axioms_ast, alpha_ast, timeout_ms
        )
    except ValueError as e:
        # Catch errors like "Unbound variable" from malformed formulas
        return AbdEvalResult(
            valid=False,
            parse_error=f"Formula evaluation error: {e}",
            **_alpha_result_kwargs(parsed_alpha),
        )
    except Exception as e:
        return AbdEvalResult(
            valid=False,
            parse_error=f"Evaluation error: {type(e).__name__}: {e}",
            **_alpha_result_kwargs(parsed_alpha),
        )

    # Build per-world breakdown
    # For skeptical, per_world_results contains worst-case costs
    per_world = []
    for i, (world, result) in enumerate(zip(train_worlds, per_world_results)):
        world_id = world.get("worldId", f"W{i}")
        # For skeptical, optCost might be stored as optWorstCost or just optCost
        opt_worst_cost = world.get("optWorstCost", world.get("optCost", 0))

        per_world.append(
            {
                "worldId": world_id,
                "valid": result.valid,
                "worstCost": result.cost if result.valid else None,
                "optWorstCost": opt_worst_cost,
                "gap": (
                    (result.cost - opt_worst_cost)
                    if result.valid and result.cost is not None
                    else None
                ),
                "reason": result.reason if not result.valid else None,
            }
        )

    # Compute aggregates
    # For skeptical, we use worst-case costs
    total_opt_worst_cost = sum(
        w.get("optWorstCost", w.get("optCost", 0)) for w in train_worlds
    )
    total_gap = (total_worst_cost - total_opt_worst_cost) if all_valid else None
    avg_gap = (total_gap / len(train_worlds)) if all_valid and len(train_worlds) > 0 else None

    # Compare to gold
    # For skeptical, gold may have goldWorstCost_skeptical
    gold_alpha = gold.get("alpha", "")
    gold_total_worst_cost = gold.get("totalGoldWorstCost_skeptical", gold.get("totalGoldAlphaCost", 0))
    cost_vs_gold = (total_worst_cost - gold_total_worst_cost) if all_valid else None

    return AbdEvalResult(
        valid=all_valid,
        per_world=per_world,
        total_cost=total_worst_cost if all_valid else None,  # This is worst-case cost
        total_opt_cost=total_opt_worst_cost,
        total_gap=total_gap,
        avg_gap=avg_gap,
        gold_alpha=gold_alpha,
        gold_total_cost=gold_total_worst_cost,
        cost_vs_gold=cost_vs_gold,
        **_alpha_result_kwargs(parsed_alpha),
    )


def evaluate_abd_b1(
    problem: Dict[str, Any],
    alpha_sexpr: str,
    timeout_ms: int = 5000,
    enforce_predicate_scoping: bool = True,
) -> AbdEvalResult:
    """
    Evaluate an alpha formula (auto-detect scenario).

    Args:
        problem: Problem dictionary
        alpha_sexpr: Alpha formula in S-expression format
        timeout_ms: Z3 timeout
        enforce_predicate_scoping: If True, reject formulas using forbidden predicates

    Returns:
        AbdEvalResult with evaluation metrics
    """
    prob_data = problem.get("problem", problem)
    scenario = prob_data.get("scenario", "ABD_FULL")

    if scenario == "ABD_PARTIAL":
        return evaluate_abd_partial(problem, alpha_sexpr, timeout_ms, enforce_predicate_scoping)
    elif scenario == "ABD_SKEPTICAL":
        return evaluate_abd_skeptical(problem, alpha_sexpr, timeout_ms, enforce_predicate_scoping)
    else:
        return evaluate_abd_full(problem, alpha_sexpr, timeout_ms, enforce_predicate_scoping)


# =============================================================================
# Result Formatting
# =============================================================================


def format_eval_result(result: AbdEvalResult) -> Dict[str, Any]:
    """Convert AbdEvalResult to a dictionary for storage."""
    d = {
        "valid": result.valid,
        "parseError": result.parse_error,
        "perWorld": result.per_world,
        "totalCost": result.total_cost,
        "totalOptCost": result.total_opt_cost,
        "totalGap": result.total_gap,
        "avgGap": result.avg_gap,
        "goldAlpha": result.gold_alpha,
        "goldTotalCost": result.gold_total_cost,
        "costVsGold": result.cost_vs_gold,
        "alphaSexpr": result.alpha_sexpr,
        "alphaNormalized": result.alpha_normalized,
    }
    if result.alpha_original_sexpr is not None:
        d["alphaOriginalSexpr"] = result.alpha_original_sexpr
    if result.trailing_parens_added > 0:
        d["trailingParensAdded"] = result.trailing_parens_added
    if result.forbidden_preds_used:
        d["forbiddenPredsUsed"] = result.forbidden_preds_used
    return d


def evaluate_abd_b1_result(
    problem: Dict[str, Any], result: Dict[str, Any], timeout_ms: int = 5000
) -> Dict[str, Any]:
    """
    Evaluate a single LLM result and return evaluation dictionary.

    Args:
        problem: Problem dictionary
        result: LLM result dictionary with 'response' or 'extractedFormula'
        timeout_ms: Z3 timeout

    Returns:
        Evaluation dictionary to be stored in result['evaluation']
    """
    # Extract formula
    if "extractedFormula" in result and result["extractedFormula"]:
        alpha_sexpr = result["extractedFormula"]
    elif "response" in result:
        alpha_sexpr, error = extract_alpha_from_response(result["response"])
        if error:
            return {
                "valid": False,
                "parseError": error,
                "extractionError": True,
            }
    else:
        return {
            "valid": False,
            "parseError": "No response or formula found",
            "extractionError": True,
        }

    # Evaluate
    eval_result = evaluate_abd_b1(problem, alpha_sexpr, timeout_ms)

    return format_eval_result(eval_result)


# =============================================================================
# Holdout Evaluation
# =============================================================================

@dataclass
class HoldoutEvalResult:
    """Result of evaluating an alpha formula on holdout worlds."""

    # Number of holdout worlds evaluated
    num_holdouts: int = 0

    # Validity on holdouts
    holdout_valid: bool = False  # All holdouts valid
    holdout_valid_count: int = 0  # Number of valid holdouts
    holdout_valid_rate: float = 0.0  # Proportion of valid holdouts

    # Cost metrics (only for valid holdouts)
    holdout_total_cost: Optional[int] = None
    holdout_total_opt_cost: Optional[int] = None
    holdout_total_gap: Optional[int] = None
    holdout_avg_gap: Optional[float] = None

    # Per-holdout breakdown
    per_holdout: Optional[List[Dict[str, Any]]] = None


def load_holdouts_from_jsonl(holdout_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load holdout worlds from JSONL sidecar file.

    Args:
        holdout_path: Path to holdout JSONL file

    Returns:
        Dict mapping instanceId to list of holdout world dicts
    """
    import json

    holdouts: Dict[str, List[Dict[str, Any]]] = {}

    if not os.path.exists(holdout_path):
        return holdouts

    from .benchmark_io import open_text

    with open_text(holdout_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                instance_id = record.get('instanceId', '')
                world = record.get('world', {})
                meta = record.get('meta', {})

                # Skip failed/timeout holdouts
                if meta.get('status') != 'success' or not world:
                    continue

                # Add metadata to world
                world['holdoutIdx'] = record.get('holdoutIdx', 0)
                world['scenario'] = record.get('scenario', 'ABD_FULL')
                world['theoryId'] = record.get('theoryId', '')

                if instance_id not in holdouts:
                    holdouts[instance_id] = []
                holdouts[instance_id].append(world)

            except Exception:
                continue

    return holdouts


def evaluate_on_holdouts(
    problem: Dict[str, Any],
    alpha_sexpr: str,
    holdout_worlds: List[Dict[str, Any]],
    timeout_ms: int = 5000,
) -> HoldoutEvalResult:
    """
    Evaluate an alpha formula on holdout worlds.

    Args:
        problem: Problem dictionary (for axioms)
        alpha_sexpr: Alpha formula in S-expression format
        holdout_worlds: List of holdout world dicts
        timeout_ms: Z3 timeout

    Returns:
        HoldoutEvalResult with holdout evaluation metrics
    """
    if not holdout_worlds:
        return HoldoutEvalResult(num_holdouts=0)

    # Parse alpha
    try:
        parsed_alpha = parse_alpha_formula_with_suffix_repair(alpha_sexpr)
        alpha_ast = parsed_alpha.ast
    except Exception:
        return HoldoutEvalResult(num_holdouts=len(holdout_worlds), holdout_valid=False)

    # Get axioms from problem
    prob_data = problem.get("problem", problem)
    theory = prob_data.get("theory", {})
    axioms = theory.get("axioms", []) or prob_data.get("axioms", [])
    scenario = prob_data.get("scenario", "ABD_FULL")

    try:
        axioms_ast = parse_axioms(axioms)
    except Exception:
        return HoldoutEvalResult(num_holdouts=len(holdout_worlds), holdout_valid=False)

    # Evaluate on each holdout world
    per_holdout = []
    valid_count = 0
    total_cost = 0
    total_opt_cost = 0

    for world in holdout_worlds:
        world_id = world.get("worldId", world.get("holdoutIdx", "H?"))
        opt_cost = world.get("optCost", world.get("optCost_full", 0))

        try:
            # Evaluate based on scenario
            if scenario == "ABD_PARTIAL":
                result = check_abd_partial_validity(world, axioms_ast, alpha_ast, timeout_ms)
            elif scenario == "ABD_SKEPTICAL":
                result = check_abd_skeptical_validity(world, axioms_ast, alpha_ast, timeout_ms)
            else:
                result = check_abd_full_validity(world, axioms_ast, alpha_ast, timeout_ms)

            if result.valid:
                valid_count += 1
                cost = result.cost if result.cost is not None else 0
                total_cost += cost
                total_opt_cost += opt_cost

                per_holdout.append({
                    "worldId": world_id,
                    "holdoutIdx": world.get("holdoutIdx"),
                    "valid": True,
                    "cost": cost,
                    "optCost": opt_cost,
                    "gap": cost - opt_cost,
                })
            else:
                per_holdout.append({
                    "worldId": world_id,
                    "holdoutIdx": world.get("holdoutIdx"),
                    "valid": False,
                    "reason": result.reason,
                })

        except Exception as e:
            per_holdout.append({
                "worldId": world_id,
                "holdoutIdx": world.get("holdoutIdx"),
                "valid": False,
                "reason": f"Evaluation error: {e}",
            })

    # Compute aggregates
    num_holdouts = len(holdout_worlds)
    all_valid = valid_count == num_holdouts
    valid_rate = valid_count / num_holdouts if num_holdouts > 0 else 0.0

    total_gap = (total_cost - total_opt_cost) if valid_count > 0 else None
    avg_gap = total_gap / valid_count if valid_count > 0 and total_gap is not None else None

    return HoldoutEvalResult(
        num_holdouts=num_holdouts,
        holdout_valid=all_valid,
        holdout_valid_count=valid_count,
        holdout_valid_rate=valid_rate,
        holdout_total_cost=total_cost if valid_count > 0 else None,
        holdout_total_opt_cost=total_opt_cost if valid_count > 0 else None,
        holdout_total_gap=total_gap,
        holdout_avg_gap=avg_gap,
        per_holdout=per_holdout,
    )


def format_holdout_result(result: HoldoutEvalResult) -> Dict[str, Any]:
    """Convert HoldoutEvalResult to a dictionary for storage."""
    return {
        "numHoldouts": result.num_holdouts,
        "holdoutValid": result.holdout_valid,
        "holdoutValidCount": result.holdout_valid_count,
        "holdoutValidRate": result.holdout_valid_rate,
        "holdoutTotalCost": result.holdout_total_cost,
        "holdoutTotalOptCost": result.holdout_total_opt_cost,
        "holdoutTotalGap": result.holdout_total_gap,
        "holdoutAvgGap": result.holdout_avg_gap,
        "perHoldout": result.per_holdout,
    }
