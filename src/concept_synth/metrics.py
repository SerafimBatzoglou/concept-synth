"""
metrics.py - Formula Complexity Metrics

Provides functions to compute various complexity metrics for FOL formulas:
- AST size (number of nodes)
- Quantifier depth
- Quantifier alternation count
- Variables used
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Set

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
from concept_synth.fol.formulas import (
    Constant,
    Eq,
    Exists,
    FOAnd,
    FOBiconditional,
    FOFormula,
    FOImplies,
    FONot,
    FOOr,
    Forall,
    FOTerm,
    Pred,
    Var,
)


def ast_size(formula: FOFormula) -> int:
    """
    Compute the AST size (number of nodes) of a formula.

    Each operator, predicate, variable, and constant counts as 1 node.
    """
    if isinstance(formula, Pred):
        # Predicate node + argument nodes
        return 1 + sum(_term_size(a) for a in formula.args)

    elif isinstance(formula, Eq):
        return 1 + _term_size(formula.left) + _term_size(formula.right)

    elif isinstance(formula, FONot):
        return 1 + ast_size(formula.child)

    elif isinstance(formula, (FOAnd, FOOr, FOImplies, FOBiconditional)):
        return 1 + ast_size(formula.left) + ast_size(formula.right)

    elif isinstance(formula, (Forall, Exists)):
        # Quantifier + bound variable + body
        return 1 + 1 + ast_size(formula.body)

    else:
        raise ValueError(f"Unknown formula type: {type(formula)}")


def _term_size(term: FOTerm) -> int:
    """Compute the size of a term."""
    if isinstance(term, (Var, Constant)):
        return 1
    else:
        return 1  # Unknown term type, count as 1


def quantifier_depth(formula: FOFormula) -> int:
    """
    Compute the maximum quantifier nesting depth.

    Examples:
        (P x) -> 0
        (forall y (P y)) -> 1
        (forall y (exists z (R y z))) -> 2
    """
    if isinstance(formula, Pred):
        return 0

    elif isinstance(formula, Eq):
        return 0

    elif isinstance(formula, FONot):
        return quantifier_depth(formula.child)

    elif isinstance(formula, (FOAnd, FOOr, FOImplies, FOBiconditional)):
        return max(quantifier_depth(formula.left), quantifier_depth(formula.right))

    elif isinstance(formula, (Forall, Exists)):
        return 1 + quantifier_depth(formula.body)

    else:
        raise ValueError(f"Unknown formula type: {type(formula)}")


def alternation_count(formula: FOFormula) -> int:
    """
    Count the number of quantifier alternations.

    An alternation occurs when we switch from ∀ to ∃ or vice versa
    along a path from root to leaf.

    Examples:
        (forall y (P y)) -> 0
        (forall y (exists z (R y z))) -> 1
        (exists x (forall y (exists z (R x y z)))) -> 2
    """
    return _alternation_helper(formula, None)


def _alternation_helper(formula: FOFormula, last_quant: str) -> int:
    """Helper for counting alternations."""
    if isinstance(formula, Pred):
        return 0

    elif isinstance(formula, Eq):
        return 0

    elif isinstance(formula, FONot):
        return _alternation_helper(formula.child, last_quant)

    elif isinstance(formula, (FOAnd, FOOr, FOImplies, FOBiconditional)):
        left_alt = _alternation_helper(formula.left, last_quant)
        right_alt = _alternation_helper(formula.right, last_quant)
        return max(left_alt, right_alt)

    elif isinstance(formula, Forall):
        if last_quant == "exists":
            return 1 + _alternation_helper(formula.body, "forall")
        else:
            return _alternation_helper(formula.body, "forall")

    elif isinstance(formula, Exists):
        if last_quant == "forall":
            return 1 + _alternation_helper(formula.body, "exists")
        else:
            return _alternation_helper(formula.body, "exists")

    else:
        raise ValueError(f"Unknown formula type: {type(formula)}")


def vars_used(formula: FOFormula) -> Set[str]:
    """
    Get the set of all variable names used in the formula
    (both free and bound).
    """
    result = set()
    _collect_vars(formula, result)
    return result


def _collect_vars(formula: FOFormula, result: Set[str]):
    """Helper to collect variables."""
    if isinstance(formula, Pred):
        for arg in formula.args:
            if isinstance(arg, Var):
                result.add(arg.name)

    elif isinstance(formula, Eq):
        if isinstance(formula.left, Var):
            result.add(formula.left.name)
        if isinstance(formula.right, Var):
            result.add(formula.right.name)

    elif isinstance(formula, FONot):
        _collect_vars(formula.child, result)

    elif isinstance(formula, (FOAnd, FOOr, FOImplies, FOBiconditional)):
        _collect_vars(formula.left, result)
        _collect_vars(formula.right, result)

    elif isinstance(formula, (Forall, Exists)):
        var_name = formula.var.name if isinstance(formula.var, Var) else str(formula.var)
        result.add(var_name)
        _collect_vars(formula.body, result)


def free_vars(formula: FOFormula) -> Set[str]:
    """
    Get the set of free variable names in the formula.
    """
    return _free_vars_helper(formula, set())


def _free_vars_helper(formula: FOFormula, bound: Set[str]) -> Set[str]:
    """Helper to collect free variables."""
    if isinstance(formula, Pred):
        result = set()
        for arg in formula.args:
            if isinstance(arg, Var) and arg.name not in bound:
                result.add(arg.name)
        return result

    elif isinstance(formula, Eq):
        result = set()
        if isinstance(formula.left, Var) and formula.left.name not in bound:
            result.add(formula.left.name)
        if isinstance(formula.right, Var) and formula.right.name not in bound:
            result.add(formula.right.name)
        return result

    elif isinstance(formula, FONot):
        return _free_vars_helper(formula.child, bound)

    elif isinstance(formula, (FOAnd, FOOr, FOImplies, FOBiconditional)):
        left_free = _free_vars_helper(formula.left, bound)
        right_free = _free_vars_helper(formula.right, bound)
        return left_free | right_free

    elif isinstance(formula, (Forall, Exists)):
        var_name = formula.var.name if isinstance(formula.var, Var) else str(formula.var)
        new_bound = bound | {var_name}
        return _free_vars_helper(formula.body, new_bound)

    else:
        return set()


def compute_all_metrics(formula: FOFormula) -> Dict[str, Any]:
    """
    Compute all metrics for a formula and return as a dictionary.
    """
    return {
        "astSize": ast_size(formula),
        "quantifierDepth": quantifier_depth(formula),
        "alternations": alternation_count(formula),
        "varsUsed": list(vars_used(formula)),
        "freeVars": list(free_vars(formula)),
    }
