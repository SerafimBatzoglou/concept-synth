"""Small formula helpers needed by the public ABD evaluator."""

from __future__ import annotations

from typing import Set

from concept_synth.fol.formulas import (
    Eq,
    Exists,
    FOAnd,
    FOBiconditional,
    FOFormula,
    FOImplies,
    FONot,
    FOOr,
    Forall,
    Pred,
)


def get_used_predicates(formula_ast: FOFormula) -> Set[str]:
    """Return the predicate names referenced in a formula AST."""
    preds: Set[str] = set()

    if isinstance(formula_ast, Pred):
        preds.add(formula_ast.name)
    elif isinstance(formula_ast, Eq):
        return preds
    elif isinstance(formula_ast, FONot):
        preds |= get_used_predicates(formula_ast.child)
    elif isinstance(formula_ast, (FOAnd, FOOr, FOImplies, FOBiconditional)):
        preds |= get_used_predicates(formula_ast.left)
        preds |= get_used_predicates(formula_ast.right)
    elif isinstance(formula_ast, (Forall, Exists)):
        preds |= get_used_predicates(formula_ast.body)

    return preds

