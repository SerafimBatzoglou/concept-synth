"""
sexpr_printer.py - S-expression Printer for FOL Formulas

Converts FOFormula AST objects to canonical S-expression strings.
Supports alpha-normalization for comparing formulas.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Set

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


def to_sexpr(formula: FOFormula, alpha_normalize: bool = False) -> str:
    """
    Convert an FOFormula to its S-expression string representation.

    Args:
        formula: The formula to convert
        alpha_normalize: If True, rename bound variables to canonical names (y, z, w, ...)

    Returns:
        The S-expression string
    """
    if alpha_normalize:
        return _to_sexpr_alpha(formula, {}, ["y", "z", "w", "u", "v"], 0)[0]
    else:
        return _to_sexpr(formula)


def _to_sexpr(f: FOFormula) -> str:
    """Convert formula to S-expression without alpha normalization."""
    if isinstance(f, Pred):
        args_str = " ".join(_term_to_str(a) for a in f.args)
        return f"({f.name} {args_str})"

    elif isinstance(f, Eq):
        return f"(= {_term_to_str(f.left)} {_term_to_str(f.right)})"

    elif isinstance(f, FONot):
        return f"(not {_to_sexpr(f.child)})"

    elif isinstance(f, FOAnd):
        return f"(and {_to_sexpr(f.left)} {_to_sexpr(f.right)})"

    elif isinstance(f, FOOr):
        return f"(or {_to_sexpr(f.left)} {_to_sexpr(f.right)})"

    elif isinstance(f, FOImplies):
        return f"(implies {_to_sexpr(f.left)} {_to_sexpr(f.right)})"

    elif isinstance(f, FOBiconditional):
        return f"(iff {_to_sexpr(f.left)} {_to_sexpr(f.right)})"

    elif isinstance(f, Forall):
        var_name = f.var.name if isinstance(f.var, Var) else str(f.var)
        return f"(forall {var_name} {_to_sexpr(f.body)})"

    elif isinstance(f, Exists):
        var_name = f.var.name if isinstance(f.var, Var) else str(f.var)
        return f"(exists {var_name} {_to_sexpr(f.body)})"

    else:
        raise ValueError(f"Unknown formula type: {type(f)}")


def _term_to_str(t: FOTerm) -> str:
    """Convert a term to string."""
    if isinstance(t, Var):
        return t.name
    elif isinstance(t, Constant):
        return t.name
    else:
        return str(t)


def _to_sexpr_alpha(
    f: FOFormula, var_map: Dict[str, str], fresh_vars: List[str], next_idx: int
) -> tuple:
    """Convert formula to S-expression with alpha normalization."""

    if isinstance(f, Pred):
        args_str = " ".join(_term_to_str_mapped(a, var_map) for a in f.args)
        return f"({f.name} {args_str})", next_idx

    elif isinstance(f, Eq):
        return (
            f"(= {_term_to_str_mapped(f.left, var_map)} {_term_to_str_mapped(f.right, var_map)})",
            next_idx,
        )

    elif isinstance(f, FONot):
        inner, next_idx = _to_sexpr_alpha(f.child, var_map, fresh_vars, next_idx)
        return f"(not {inner})", next_idx

    elif isinstance(f, FOAnd):
        left, next_idx = _to_sexpr_alpha(f.left, var_map, fresh_vars, next_idx)
        right, next_idx = _to_sexpr_alpha(f.right, var_map, fresh_vars, next_idx)
        return f"(and {left} {right})", next_idx

    elif isinstance(f, FOOr):
        left, next_idx = _to_sexpr_alpha(f.left, var_map, fresh_vars, next_idx)
        right, next_idx = _to_sexpr_alpha(f.right, var_map, fresh_vars, next_idx)
        return f"(or {left} {right})", next_idx

    elif isinstance(f, FOImplies):
        left, next_idx = _to_sexpr_alpha(f.left, var_map, fresh_vars, next_idx)
        right, next_idx = _to_sexpr_alpha(f.right, var_map, fresh_vars, next_idx)
        return f"(implies {left} {right})", next_idx

    elif isinstance(f, FOBiconditional):
        left, next_idx = _to_sexpr_alpha(f.left, var_map, fresh_vars, next_idx)
        right, next_idx = _to_sexpr_alpha(f.right, var_map, fresh_vars, next_idx)
        return f"(iff {left} {right})", next_idx

    elif isinstance(f, Forall):
        old_var = f.var.name if isinstance(f.var, Var) else str(f.var)
        if next_idx < len(fresh_vars):
            new_var = fresh_vars[next_idx]
        else:
            new_var = f"v{next_idx}"
        new_map = dict(var_map)
        new_map[old_var] = new_var
        body, final_idx = _to_sexpr_alpha(f.body, new_map, fresh_vars, next_idx + 1)
        return f"(forall {new_var} {body})", final_idx

    elif isinstance(f, Exists):
        old_var = f.var.name if isinstance(f.var, Var) else str(f.var)
        if next_idx < len(fresh_vars):
            new_var = fresh_vars[next_idx]
        else:
            new_var = f"v{next_idx}"
        new_map = dict(var_map)
        new_map[old_var] = new_var
        body, final_idx = _to_sexpr_alpha(f.body, new_map, fresh_vars, next_idx + 1)
        return f"(exists {new_var} {body})", final_idx

    else:
        raise ValueError(f"Unknown formula type: {type(f)}")


def _term_to_str_mapped(t: FOTerm, var_map: Dict[str, str]) -> str:
    """Convert a term to string with variable mapping."""
    if isinstance(t, Var):
        return var_map.get(t.name, t.name)
    elif isinstance(t, Constant):
        return t.name
    else:
        return str(t)


def to_sexpr_canonical(formula: FOFormula) -> str:
    """
    Convert formula to canonical S-expression with alpha-normalized variables.
    This is useful for comparing formulas for equivalence.
    """
    return to_sexpr(formula, alpha_normalize=True)


def to_sexpr_nnf(formula: FOFormula) -> str:
    """
    Convert formula to S-expression, assuming it's in NNF
    (negation normal form - negations only on atoms).
    """
    return to_sexpr(formula, alpha_normalize=False)
