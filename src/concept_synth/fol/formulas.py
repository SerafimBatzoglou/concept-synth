"""
formulas.py - First-Order Logic AST and Athena Printer

Extends the existing propositional logic formulas with first-order constructs:
- Terms: Constants and Variables
- Predicates: P(t1, ..., tn)
- Quantifiers: Forall and Exists

The printer outputs Athena concrete syntax for first-order formulas.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Set, Union

# =============================================================================
# Terms
# =============================================================================


@dataclass(frozen=True)
class FOTerm(ABC):
    """Abstract base class for first-order terms."""

    @abstractmethod
    def to_athena(self) -> str:
        """Convert term to Athena syntax."""
        pass

    @abstractmethod
    def get_vars(self) -> Set[str]:
        """Get all variable names in this term."""
        pass


@dataclass(frozen=True)
class Constant(FOTerm):
    """A constant symbol (e.g., a0, a1, ...)."""

    name: str

    def to_athena(self) -> str:
        return self.name

    def get_vars(self) -> Set[str]:
        return set()

    def __repr__(self) -> str:
        return f"Const({self.name})"


@dataclass(frozen=True)
class Var(FOTerm):
    """A variable (e.g., x, y, z)."""

    name: str

    def to_athena(self) -> str:
        # Variables in Athena are referenced by their bound name
        return self.name

    def get_vars(self) -> Set[str]:
        return {self.name}

    def __repr__(self) -> str:
        return f"Var({self.name})"


# =============================================================================
# Formulas
# =============================================================================


@dataclass
class FOFormula(ABC):
    """Abstract base class for first-order formulas."""

    @abstractmethod
    def to_athena(self) -> str:
        """Convert formula to Athena syntax."""
        pass

    @abstractmethod
    def free_vars(self) -> Set[str]:
        """Get all free variable names in this formula."""
        pass

    @abstractmethod
    def substitute(self, var_name: str, term: FOTerm) -> FOFormula:
        """Substitute a term for a variable."""
        pass


@dataclass
class Pred(FOFormula):
    """
    Predicate application: P(t1, ..., tn)

    For our benchmark:
    - P(x) is a unary predicate (arity 1)
    - E(x, y) is a binary predicate (arity 2)
    """

    name: str
    args: List[FOTerm]

    def to_athena(self) -> str:
        if len(self.args) == 0:
            return self.name
        arg_strs = " ".join(arg.to_athena() for arg in self.args)
        return f"({self.name} {arg_strs})"

    def free_vars(self) -> Set[str]:
        result = set()
        for arg in self.args:
            result |= arg.get_vars()
        return result

    def substitute(self, var_name: str, term: FOTerm) -> FOFormula:
        new_args = []
        for arg in self.args:
            if isinstance(arg, Var) and arg.name == var_name:
                new_args.append(term)
            else:
                new_args.append(arg)
        return Pred(self.name, new_args)

    def __repr__(self) -> str:
        args_str = ", ".join(repr(a) for a in self.args)
        return f"Pred({self.name}, [{args_str}])"


@dataclass
class Eq(FOFormula):
    """Equality: t1 = t2"""

    left: FOTerm
    right: FOTerm

    def to_athena(self) -> str:
        return f"(= {self.left.to_athena()} {self.right.to_athena()})"

    def free_vars(self) -> Set[str]:
        return self.left.get_vars() | self.right.get_vars()

    def substitute(self, var_name: str, term: FOTerm) -> FOFormula:
        new_left = term if isinstance(self.left, Var) and self.left.name == var_name else self.left
        new_right = (
            term if isinstance(self.right, Var) and self.right.name == var_name else self.right
        )
        return Eq(new_left, new_right)


@dataclass
class FONot(FOFormula):
    """Negation: ~φ"""

    child: FOFormula

    def to_athena(self) -> str:
        return f"(~ {self.child.to_athena()})"

    def free_vars(self) -> Set[str]:
        return self.child.free_vars()

    def substitute(self, var_name: str, term: FOTerm) -> FOFormula:
        return FONot(self.child.substitute(var_name, term))


@dataclass
class FOAnd(FOFormula):
    """Conjunction: φ & ψ"""

    left: FOFormula
    right: FOFormula

    def to_athena(self) -> str:
        return f"({self.left.to_athena()} & {self.right.to_athena()})"

    def free_vars(self) -> Set[str]:
        return self.left.free_vars() | self.right.free_vars()

    def substitute(self, var_name: str, term: FOTerm) -> FOFormula:
        return FOAnd(self.left.substitute(var_name, term), self.right.substitute(var_name, term))


@dataclass
class FOOr(FOFormula):
    """Disjunction: φ | ψ"""

    left: FOFormula
    right: FOFormula

    def to_athena(self) -> str:
        return f"({self.left.to_athena()} | {self.right.to_athena()})"

    def free_vars(self) -> Set[str]:
        return self.left.free_vars() | self.right.free_vars()

    def substitute(self, var_name: str, term: FOTerm) -> FOFormula:
        return FOOr(self.left.substitute(var_name, term), self.right.substitute(var_name, term))


@dataclass
class FOImplies(FOFormula):
    """Implication: φ ==> ψ"""

    left: FOFormula
    right: FOFormula

    def to_athena(self) -> str:
        return f"({self.left.to_athena()} ==> {self.right.to_athena()})"

    def free_vars(self) -> Set[str]:
        return self.left.free_vars() | self.right.free_vars()

    def substitute(self, var_name: str, term: FOTerm) -> FOFormula:
        return FOImplies(
            self.left.substitute(var_name, term), self.right.substitute(var_name, term)
        )


@dataclass
class FOBiconditional(FOFormula):
    """Biconditional: φ <==> ψ"""

    left: FOFormula
    right: FOFormula

    def to_athena(self) -> str:
        return f"({self.left.to_athena()} <==> {self.right.to_athena()})"

    def free_vars(self) -> Set[str]:
        return self.left.free_vars() | self.right.free_vars()

    def substitute(self, var_name: str, term: FOTerm) -> FOFormula:
        return FOBiconditional(
            self.left.substitute(var_name, term), self.right.substitute(var_name, term)
        )


@dataclass
class Forall(FOFormula):
    """
    Universal quantification: (forall x . φ)

    In Athena, we use: (forall x . body)
    where x is declared via: define [x y z ...] := [?x:D ?y:D ?z:D ...]
    """

    var: Var
    sort: str  # The sort/type of the variable (e.g., "D")
    body: FOFormula

    def to_athena(self) -> str:
        return f"(forall {self.var.name} . {self.body.to_athena()})"

    def free_vars(self) -> Set[str]:
        return self.body.free_vars() - {self.var.name}

    def substitute(self, var_name: str, term: FOTerm) -> FOFormula:
        if var_name == self.var.name:
            # Don't substitute bound variables
            return self
        return Forall(self.var, self.sort, self.body.substitute(var_name, term))


@dataclass
class Exists(FOFormula):
    """
    Existential quantification: (exists x . φ)

    In Athena, we use: (exists x . body)
    where x is declared via: define [x y z ...] := [?x:D ?y:D ?z:D ...]
    """

    var: Var
    sort: str  # The sort/type of the variable (e.g., "D")
    body: FOFormula

    def to_athena(self) -> str:
        return f"(exists {self.var.name} . {self.body.to_athena()})"

    def free_vars(self) -> Set[str]:
        return self.body.free_vars() - {self.var.name}

    def substitute(self, var_name: str, term: FOTerm) -> FOFormula:
        if var_name == self.var.name:
            # Don't substitute bound variables
            return self
        return Exists(self.var, self.sort, self.body.substitute(var_name, term))


# =============================================================================
# Helper Functions
# =============================================================================


def fo_to_athena_string(formula: FOFormula) -> str:
    """Convert a first-order formula to Athena syntax string."""
    return formula.to_athena()


def free_vars(formula: FOFormula) -> Set[str]:
    """Get all free variables in a formula."""
    return formula.free_vars()


def bound_vars(formula: FOFormula) -> Set[str]:
    """Get all bound variables in a formula."""
    result = set()

    def collect(f: FOFormula):
        if isinstance(f, Pred):
            pass
        elif isinstance(f, Eq):
            pass
        elif isinstance(f, FONot):
            collect(f.child)
        elif isinstance(f, (FOAnd, FOOr, FOImplies, FOBiconditional)):
            collect(f.left)
            collect(f.right)
        elif isinstance(f, (Forall, Exists)):
            result.add(f.var.name)
            collect(f.body)

    collect(formula)
    return result


# Standard variable names for Athena (matching the define block)
STANDARD_VAR_NAMES = ["x", "y", "z", "u", "v", "w"]


def rename_vars_standard(formula: FOFormula, sort: str = "D") -> FOFormula:
    """
    Rename all bound variables to use standard names from STANDARD_VAR_NAMES.
    This ensures formulas match the Athena define block:
        define [x y z u v w] := [?x:D ?y:D ?z:D ?u:D ?v:D ?w:D]

    Returns a new formula with standardized variable names.
    """
    used_names: List[str] = []
    name_mapping: dict = {}

    def get_fresh_name() -> str:
        for name in STANDARD_VAR_NAMES:
            if name not in used_names:
                used_names.append(name)
                return name
        raise ValueError(f"Too many bound variables (max {len(STANDARD_VAR_NAMES)})")

    def rename(f: FOFormula) -> FOFormula:
        if isinstance(f, Pred):
            new_args = []
            for arg in f.args:
                if isinstance(arg, Var) and arg.name in name_mapping:
                    new_args.append(Var(name_mapping[arg.name]))
                else:
                    new_args.append(arg)
            return Pred(f.name, new_args)

        elif isinstance(f, Eq):
            new_left = (
                Var(name_mapping[f.left.name])
                if isinstance(f.left, Var) and f.left.name in name_mapping
                else f.left
            )
            new_right = (
                Var(name_mapping[f.right.name])
                if isinstance(f.right, Var) and f.right.name in name_mapping
                else f.right
            )
            return Eq(new_left, new_right)

        elif isinstance(f, FONot):
            return FONot(rename(f.child))

        elif isinstance(f, FOAnd):
            return FOAnd(rename(f.left), rename(f.right))

        elif isinstance(f, FOOr):
            return FOOr(rename(f.left), rename(f.right))

        elif isinstance(f, FOImplies):
            return FOImplies(rename(f.left), rename(f.right))

        elif isinstance(f, FOBiconditional):
            return FOBiconditional(rename(f.left), rename(f.right))

        elif isinstance(f, Forall):
            old_name = f.var.name
            new_name = get_fresh_name()
            name_mapping[old_name] = new_name
            new_body = rename(f.body)
            del name_mapping[old_name]  # Remove from scope after processing
            return Forall(Var(new_name), sort, new_body)

        elif isinstance(f, Exists):
            old_name = f.var.name
            new_name = get_fresh_name()
            name_mapping[old_name] = new_name
            new_body = rename(f.body)
            del name_mapping[old_name]  # Remove from scope after processing
            return Exists(Var(new_name), sort, new_body)

        else:
            raise ValueError(f"Unknown formula type: {type(f)}")

    return rename(formula)


def get_all_vars_used(formula: FOFormula) -> Set[str]:
    """Get all variable names used in a formula (both free and bound)."""
    return free_vars(formula) | bound_vars(formula)


def formula_depth(formula: FOFormula) -> int:
    """Calculate the depth of a formula (maximum nesting of quantifiers/connectives)."""
    if isinstance(formula, (Pred, Eq)):
        return 1
    elif isinstance(formula, FONot):
        return 1 + formula_depth(formula.child)
    elif isinstance(formula, (FOAnd, FOOr, FOImplies, FOBiconditional)):
        return 1 + max(formula_depth(formula.left), formula_depth(formula.right))
    elif isinstance(formula, (Forall, Exists)):
        return 1 + formula_depth(formula.body)
    else:
        return 0


def quantifier_depth(formula: FOFormula) -> int:
    """Calculate the quantifier depth (maximum nesting of quantifiers only)."""
    if isinstance(formula, (Pred, Eq)):
        return 0
    elif isinstance(formula, FONot):
        return quantifier_depth(formula.child)
    elif isinstance(formula, (FOAnd, FOOr, FOImplies, FOBiconditional)):
        return max(quantifier_depth(formula.left), quantifier_depth(formula.right))
    elif isinstance(formula, (Forall, Exists)):
        return 1 + quantifier_depth(formula.body)
    else:
        return 0


def count_quantifiers(formula: FOFormula) -> dict:
    """Count the number of each type of quantifier in a formula."""
    counts = {"forall": 0, "exists": 0}

    def count(f: FOFormula):
        if isinstance(f, Pred) or isinstance(f, Eq):
            pass
        elif isinstance(f, FONot):
            count(f.child)
        elif isinstance(f, (FOAnd, FOOr, FOImplies, FOBiconditional)):
            count(f.left)
            count(f.right)
        elif isinstance(f, Forall):
            counts["forall"] += 1
            count(f.body)
        elif isinstance(f, Exists):
            counts["exists"] += 1
            count(f.body)

    count(formula)
    return counts
