"""
Helpers for induction predicate signatures and renamed-symbol experiments.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


RENAMED_PROMPT_VARIANT = "renamed_v1"

DEFAULT_INDUCTION_SIGNATURE: Dict[str, int] = {
    "P": 1,
    "Q": 1,
    "R": 2,
    "S": 2,
    "T": 1,
}

RENAMED_INDUCTION_SIGNATURE: Dict[str, int] = {
    "Foo": 1,
    "Bar": 1,
    "Blorp": 2,
    "Wump": 2,
    "T": 1,
}

DEFAULT_ALLOWED_INDUCTION_PREDICATES: Set[str] = (
    set(DEFAULT_INDUCTION_SIGNATURE) | set(RENAMED_INDUCTION_SIGNATURE)
)


def get_problem_prompt_variant(problem: Dict[str, Any]) -> Optional[str]:
    """Return the prompt variant marker stored on a problem, if any."""
    prob = problem.get("problem", problem)
    return prob.get("promptVariant")


def get_problem_signature(problem: Dict[str, Any]) -> Dict[str, int]:
    """Return predicate arities from a problem signature, preserving order."""
    prob = problem.get("problem", problem)
    signature = {}
    predicates = prob.get("signature", {}).get("predicates", [])
    for pred in predicates:
        name = pred.get("name")
        arity = pred.get("arity")
        if name and isinstance(arity, int):
            signature[str(name)] = int(arity)
    if "T" not in signature:
        signature["T"] = 1
    return signature


def get_allowed_induction_predicates(problem: Optional[Dict[str, Any]] = None) -> Set[str]:
    """Return allowed predicate names for parsing induction formulas."""
    if problem:
        signature = get_problem_signature(problem)
        if signature:
            return set(signature.keys())
    return set(DEFAULT_ALLOWED_INDUCTION_PREDICATES)


def _parse_predicate_item_arity(item: Any) -> int:
    """Infer arity from a predicate item like 'a0' or '(a0, a1)'."""
    item_str = str(item).strip()
    if item_str.startswith("(") and item_str.endswith(")"):
        inner = item_str[1:-1]
        parts = [p.strip() for p in inner.split(",") if p.strip()]
        if len(parts) >= 2:
            return len(parts)
    return 1


def _iter_predicate_items(source: Any) -> Iterable[Any]:
    """Yield items from predicate/unknown structures."""
    if isinstance(source, dict):
        # Legacy predicate format stores observed extensions as
        # {"true": [...], "false": [...]}. Infer arity from the atoms, not
        # from the literal keys "true"/"false".
        lower_keys = {str(k).lower() for k in source.keys()}
        if lower_keys and lower_keys.issubset({"true", "false"}):
            for key in source.keys():
                values = source.get(key, [])
                if isinstance(values, (list, tuple, set)):
                    for item in values:
                        yield item
            return
        for key in source.keys():
            yield key
        return
    if isinstance(source, (list, tuple, set)):
        for item in source:
            yield item


def get_world_predicate_arities(
    world: Dict[str, Any], problem_signature: Optional[Dict[str, int]] = None
) -> Dict[str, int]:
    """
    Return predicate arities for a world.

    Priority:
    1. world.predicateArities (used by renamed benchmark files)
    2. problem signature passed in by caller
    3. infer from world data, with fallback for legacy P/Q/R/S names
    """
    world_arities = world.get("predicateArities")
    if isinstance(world_arities, dict) and world_arities:
        return {str(name): int(arity) for name, arity in world_arities.items()}

    if problem_signature:
        filtered = {name: arity for name, arity in problem_signature.items() if name != "T"}
        if filtered:
            return filtered

    arities: Dict[str, int] = {}
    for source_name in ("predicates", "fullPredicates", "unknownAtoms"):
        source = world.get(source_name, {})
        if not isinstance(source, dict):
            continue
        for pred_name, pred_data in source.items():
            if pred_name in arities:
                continue
            items = list(_iter_predicate_items(pred_data))
            if items:
                arities[pred_name] = _parse_predicate_item_arity(items[0])

    for pred_name, arity in DEFAULT_INDUCTION_SIGNATURE.items():
        if pred_name != "T" and pred_name not in arities and pred_name in world.get("predicates", {}):
            arities[pred_name] = arity

    return arities


def split_predicates_by_arity(predicate_arities: Dict[str, int]) -> Tuple[List[str], List[str]]:
    """Split predicates into unary and binary lists, preserving input order."""
    unary = [name for name, arity in predicate_arities.items() if arity == 1]
    binary = [name for name, arity in predicate_arities.items() if arity == 2]
    return unary, binary
