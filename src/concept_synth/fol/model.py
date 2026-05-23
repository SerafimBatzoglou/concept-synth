"""
model.py - Finite Structure Representation and FO Evaluator

Represents finite structures with:
- A finite domain D = {a0, a1, ..., a(n-1)}
- Interpretations for unary predicates (e.g., P)
- Interpretations for binary predicates (e.g., E)

Provides evaluation of first-order formulas over finite structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .formulas import (
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


@dataclass
class FiniteModel:
    """
    A finite first-order structure.

    Attributes:
        n: Size of the domain (elements are 0, 1, ..., n-1)
        unary: Interpretations for unary predicates.
               Maps predicate name to set of indices where it's true.
        binary: Interpretations for binary predicates.
                Maps predicate name to set of (i, j) pairs where it's true.

    The domain elements are represented as integers 0..n-1 internally,
    but rendered as constants a0, a1, ..., a(n-1) in Athena.
    """

    n: int
    unary: Dict[str, Set[int]] = field(default_factory=dict)
    binary: Dict[str, Set[Tuple[int, int]]] = field(default_factory=dict)
    const_names: Optional[List[str]] = None

    def __post_init__(self):
        if self.const_names is not None and len(self.const_names) != self.n:
            raise ValueError(
                f"const_names length {len(self.const_names)} does not match domain size {self.n}"
            )

    def domain(self) -> List[int]:
        """Return the domain as a list of indices."""
        return list(range(self.n))

    def domain_constants(self) -> List[str]:
        """Return the domain as a list of constant names (a0, a1, ...)."""
        if self.const_names is not None:
            return list(self.const_names)
        return [f"a{i}" for i in range(self.n)]

    def index_to_const(self, i: int) -> str:
        """Convert a domain index to its constant name."""
        if self.const_names is not None:
            return self.const_names[i]
        return f"a{i}"

    def const_to_index(self, const: str) -> int:
        """Convert a constant name to its domain index."""
        if self.const_names is not None:
            try:
                return self.const_names.index(const)
            except ValueError as exc:
                raise ValueError(f"Invalid constant name: {const}") from exc
        if const.startswith("a") and const[1:].isdigit():
            return int(const[1:])
        raise ValueError(f"Invalid constant name: {const}")

    def eval_unary(self, pred_name: str, index: int) -> bool:
        """Evaluate a unary predicate at a domain element."""
        if pred_name not in self.unary:
            return False
        return index in self.unary[pred_name]

    def eval_binary(self, pred_name: str, i: int, j: int) -> bool:
        """Evaluate a binary predicate at a pair of domain elements."""
        if pred_name not in self.binary:
            return False
        return (i, j) in self.binary[pred_name]

    def set_unary(self, pred_name: str, index: int, value: bool):
        """Set the value of a unary predicate at a domain element."""
        if pred_name not in self.unary:
            self.unary[pred_name] = set()
        if value:
            self.unary[pred_name].add(index)
        else:
            self.unary[pred_name].discard(index)

    def set_binary(self, pred_name: str, i: int, j: int, value: bool):
        """Set the value of a binary predicate at a pair of domain elements."""
        if pred_name not in self.binary:
            self.binary[pred_name] = set()
        if value:
            self.binary[pred_name].add((i, j))
        else:
            self.binary[pred_name].discard((i, j))

    def get_all_unary_facts(self, pred_name: str) -> List[Tuple[int, bool]]:
        """Get all facts for a unary predicate (complete diagram)."""
        facts = []
        for i in range(self.n):
            facts.append((i, self.eval_unary(pred_name, i)))
        return facts

    def get_all_binary_facts(self, pred_name: str) -> List[Tuple[int, int, bool]]:
        """Get all facts for a binary predicate (complete diagram)."""
        facts = []
        for i in range(self.n):
            for j in range(self.n):
                facts.append((i, j, self.eval_binary(pred_name, i, j)))
        return facts

    def __repr__(self) -> str:
        return f"FiniteModel(n={self.n}, unary={self.unary}, binary={self.binary})"


# =============================================================================
# FO Evaluation
# =============================================================================


def evaluate(formula: FOFormula, model: FiniteModel, env: Dict[str, int] = None) -> bool:
    """
    Evaluate a first-order formula in a finite model.

    Args:
        formula: The formula to evaluate
        model: The finite structure
        env: Variable assignment (maps variable names to domain indices)

    Returns:
        True if the formula is satisfied in the model under the assignment
    """
    if env is None:
        env = {}

    if isinstance(formula, Pred):
        # Evaluate predicate application
        pred_name = formula.name
        args = formula.args

        # Resolve each argument to a domain index
        indices = []
        for arg in args:
            if isinstance(arg, Constant):
                indices.append(model.const_to_index(arg.name))
            elif isinstance(arg, Var):
                if arg.name not in env:
                    raise ValueError(f"Unbound variable: {arg.name}")
                indices.append(env[arg.name])
            else:
                raise ValueError(f"Unknown term type: {type(arg)}")

        # Evaluate based on arity
        if len(indices) == 1:
            return model.eval_unary(pred_name, indices[0])
        elif len(indices) == 2:
            return model.eval_binary(pred_name, indices[0], indices[1])
        else:
            raise ValueError(f"Unsupported predicate arity: {len(indices)}")

    elif isinstance(formula, Eq):
        # Evaluate equality
        def resolve_term(t: FOTerm) -> int:
            if isinstance(t, Constant):
                return model.const_to_index(t.name)
            elif isinstance(t, Var):
                if t.name not in env:
                    raise ValueError(f"Unbound variable: {t.name}")
                return env[t.name]
            else:
                raise ValueError(f"Unknown term type: {type(t)}")

        left_idx = resolve_term(formula.left)
        right_idx = resolve_term(formula.right)
        return left_idx == right_idx

    elif isinstance(formula, FONot):
        return not evaluate(formula.child, model, env)

    elif isinstance(formula, FOAnd):
        return evaluate(formula.left, model, env) and evaluate(formula.right, model, env)

    elif isinstance(formula, FOOr):
        return evaluate(formula.left, model, env) or evaluate(formula.right, model, env)

    elif isinstance(formula, FOImplies):
        return (not evaluate(formula.left, model, env)) or evaluate(formula.right, model, env)

    elif isinstance(formula, FOBiconditional):
        left_val = evaluate(formula.left, model, env)
        right_val = evaluate(formula.right, model, env)
        return left_val == right_val

    elif isinstance(formula, Forall):
        # Universal: true if body is true for all domain elements
        var_name = formula.var.name
        for i in range(model.n):
            new_env = env.copy()
            new_env[var_name] = i
            if not evaluate(formula.body, model, new_env):
                return False
        return True

    elif isinstance(formula, Exists):
        # Existential: true if body is true for some domain element
        var_name = formula.var.name
        for i in range(model.n):
            new_env = env.copy()
            new_env[var_name] = i
            if evaluate(formula.body, model, new_env):
                return True
        return False

    else:
        raise ValueError(f"Unknown formula type: {type(formula)}")


def count_witnesses(formula: FOFormula, model: FiniteModel, env: Dict[str, int] = None) -> int:
    """
    For an existentially quantified formula, count how many witnesses exist.
    For other formulas, returns 1 if true, 0 if false.

    This is useful for difficulty estimation - fewer witnesses = harder proofs.
    """
    if env is None:
        env = {}

    if isinstance(formula, Exists):
        var_name = formula.var.name
        count = 0
        for i in range(model.n):
            new_env = env.copy()
            new_env[var_name] = i
            if evaluate(formula.body, model, new_env):
                count += 1
        return count
    else:
        return 1 if evaluate(formula, model, env) else 0


def find_witnesses(formula: FOFormula, model: FiniteModel, env: Dict[str, int] = None) -> List[int]:
    """
    For an existentially quantified formula, find all witnesses.
    Returns list of domain indices that satisfy the formula.
    """
    if env is None:
        env = {}

    if not isinstance(formula, Exists):
        raise ValueError("find_witnesses requires an existentially quantified formula")

    var_name = formula.var.name
    witnesses = []
    for i in range(model.n):
        new_env = env.copy()
        new_env[var_name] = i
        if evaluate(formula.body, model, new_env):
            witnesses.append(i)
    return witnesses


# =============================================================================
# Model Statistics
# =============================================================================


def model_statistics(model: FiniteModel) -> Dict:
    """Compute statistics about a finite model."""
    stats = {"domain_size": model.n, "unary_predicates": {}, "binary_predicates": {}}

    for pred_name, extension in model.unary.items():
        stats["unary_predicates"][pred_name] = {
            "true_count": len(extension),
            "false_count": model.n - len(extension),
            "density": len(extension) / model.n if model.n > 0 else 0,
        }

    for pred_name, extension in model.binary.items():
        total_pairs = model.n * model.n
        stats["binary_predicates"][pred_name] = {
            "true_count": len(extension),
            "false_count": total_pairs - len(extension),
            "density": len(extension) / total_pairs if total_pairs > 0 else 0,
        }

    return stats
