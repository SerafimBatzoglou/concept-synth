"""
abd_b1_theory_library.py - Theory Library for ABD-B1 Abduction Tasks

Defines a library of default/exception theories (Theta) using generic predicate names:
- Observed predicates: P(x), Q(x), R(x,y), S(x,y)
- Abducible predicate: Ab(x)

Each theory specifies:
- theory_id: unique identifier
- description: human-readable description
- axioms: list of S-expr formula strings
- allowed_alpha_preds: which predicates alpha(x) may use
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set


@dataclass
class TheorySpec:
    """Specification of a default/exception theory."""

    theory_id: str
    description: str
    axioms: List[str]  # S-expr formula strings
    allowed_alpha_preds: Set[str]  # Predicates that alpha may reference
    repaired_predicates: Set[str] = None  # Predicates appearing in consequent (symptom shortcuts)
    allow_shortcut: bool = False  # If True, allow repaired predicates in alpha (easy mode)
    enabled_for_generation: bool = True  # If False, excluded from benchmark generation
    notes: str = ""  # Optional notes about the theory

    def __post_init__(self):
        # Ensure allowed_alpha_preds is a set
        if isinstance(self.allowed_alpha_preds, list):
            self.allowed_alpha_preds = set(self.allowed_alpha_preds)

        # Ensure repaired_predicates is a set
        if self.repaired_predicates is None:
            self.repaired_predicates = set()
        elif isinstance(self.repaired_predicates, list):
            self.repaired_predicates = set(self.repaired_predicates)

    def get_effective_allowed_preds(self) -> Set[str]:
        """
        Get the effective set of allowed predicates for alpha.

        If allow_shortcut is False, excludes repaired predicates to prevent
        trivial solutions like alpha = (not (Q x)).

        Returns:
            Set of predicate names that alpha may use
        """
        if self.allow_shortcut:
            return self.allowed_alpha_preds
        else:
            return self.allowed_alpha_preds - self.repaired_predicates

    def get_forbidden_preds(self) -> Set[str]:
        """
        Get predicates that are forbidden in alpha.

        Always includes 'Ab'. If allow_shortcut is False, also includes
        repaired predicates.

        Returns:
            Set of forbidden predicate names
        """
        forbidden = {"Ab"}
        if not self.allow_shortcut:
            forbidden |= self.repaired_predicates
        return forbidden


# =============================================================================
# Theory Definitions
# =============================================================================

THEORY_LIBRARY: Dict[str, TheorySpec] = {}


def _register_theory(spec: TheorySpec):
    """Register a theory in the library."""
    THEORY_LIBRARY[spec.theory_id] = spec


# TH1: Simple default rule P(x) ∧ ¬Ab(x) → Q(x)
# "If x has property P and is not abnormal, then x has property Q"
# Repaired: Q (consequent) - alpha=(not Q) would be a shortcut
# Allowed: P, R, S (NOT Q) - enables relational formulas without symptom shortcuts
_register_theory(
    TheorySpec(
        theory_id="TH1",
        description="P(x) ∧ ¬Ab(x) → Q(x): Default rule with unary predicates only",
        axioms=["(forall x (implies (and (P x) (not (Ab x))) (Q x)))"],
        allowed_alpha_preds={"P", "R", "S"},
        repaired_predicates={"Q"},
        enabled_for_generation=True,
        notes="Core theory: simple unary default",
    )
)


# TH2: Default rule with existential antecedent
# "If x is related to some P-element and is not abnormal, then x has Q"
# Repaired: Q (consequent)
_register_theory(
    TheorySpec(
        theory_id="TH2",
        description="∃y(R(x,y) ∧ P(y)) ∧ ¬Ab(x) → Q(x): Default with relational antecedent",
        axioms=["(forall x (implies (and (exists y (and (R x y) (P y))) (not (Ab x))) (Q x)))"],
        allowed_alpha_preds={"P", "R", "S"},  # S added for more template variety
        repaired_predicates={"Q"},
        enabled_for_generation=True,
        notes="Core theory: relational antecedent",
    )
)


# TH3: Default rule with existential consequent
# "If x has P and is not abnormal, then x is R-related to something"
# Repaired: R (consequent involves R)
# Allowed: P, Q, S (NOT R) - enables relational formulas without symptom shortcuts
_register_theory(
    TheorySpec(
        theory_id="TH3",
        description="P(x) ∧ ¬Ab(x) → ∃y R(x,y): Default with existential consequent",
        axioms=["(forall x (implies (and (P x) (not (Ab x))) (exists y (R x y))))"],
        allowed_alpha_preds={"P", "Q", "S"},
        repaired_predicates={"R"},
        enabled_for_generation=True,
        notes="Core theory: existential consequent",
    )
)


# TH4: Same as TH3 (duplicate - DISABLED)
# Kept for backwards compatibility but excluded from generation
_register_theory(
    TheorySpec(
        theory_id="TH4",
        description="P(x) ∧ ¬Ab(x) → ∃y R(x,y): Default with existential consequent (duplicate of TH3)",
        axioms=["(forall x (implies (and (P x) (not (Ab x))) (exists y (R x y))))"],
        allowed_alpha_preds={"P", "Q", "S"},
        repaired_predicates={"R"},
        enabled_for_generation=False,  # DISABLED: duplicate of TH3
        notes="Duplicate of TH3 - excluded from benchmark v0.2+",
    )
)


# TH5: Default rule with nested universal consequent
# "If x has P and is not abnormal, then all R-successors of x have Q"
# Repaired: Q (consequent)
_register_theory(
    TheorySpec(
        theory_id="TH5",
        description="P(x) ∧ ¬Ab(x) → ∀y(R(x,y) → Q(y)): Default with universal consequent",
        axioms=["(forall x (implies (and (P x) (not (Ab x))) (forall y (implies (R x y) (Q y)))))"],
        allowed_alpha_preds={"P", "R", "S"},  # S added for more template variety
        repaired_predicates={"Q"},
        enabled_for_generation=True,
        notes="Core theory: universal consequent",
    )
)


# TH6: Two interacting default rules
# "P(x) ∧ ¬Ab(x) → Q(x)" and "Q(x) ∧ ¬Ab(x) → ∃y S(x,y)"
# Repaired: Q (first consequent), S (second consequent)
# DISABLED: effective_allowed collapses to {P} only when allow_shortcut=False
_register_theory(
    TheorySpec(
        theory_id="TH6",
        description="Two chained defaults: P→Q and Q→∃S",
        axioms=[
            "(forall x (implies (and (P x) (not (Ab x))) (Q x)))",
            "(forall x (implies (and (Q x) (not (Ab x))) (exists y (S x y))))",
        ],
        allowed_alpha_preds={"P", "R"},  # Added R to give effective_allowed={P,R}
        repaired_predicates={"Q", "S"},
        enabled_for_generation=False,  # DISABLED: degenerate alpha space
        notes="Degenerate: effective_allowed too small without R. May re-enable in v0.3.",
    )
)


# TH7: Default with binary predicate in antecedent and consequent
# "If x R-relates to some y with P(y), and x is not abnormal, then x S-relates to some z with Q(z)"
# Repaired: S, Q (consequent)
_register_theory(
    TheorySpec(
        theory_id="TH7",
        description="∃y(R(x,y)∧P(y)) ∧ ¬Ab(x) → ∃z(S(x,z)∧Q(z)): Relational default",
        axioms=[
            "(forall x (implies (and (exists y (and (R x y) (P y))) (not (Ab x))) (exists z (and (S x z) (Q z)))))"
        ],
        allowed_alpha_preds={"P", "R"},  # Only P and R after excluding repaired S,Q
        repaired_predicates={"S", "Q"},
        enabled_for_generation=True,
        notes="Core theory: relational antecedent and consequent",
    )
)


# TH8: Contrapositive-style: if no Q, then Ab or no P
# Equivalent to TH1 but stated differently for variety
# Repaired: Q (same as TH1)
# DISABLED: effective_allowed collapses to {P} only
_register_theory(
    TheorySpec(
        theory_id="TH8",
        description="¬Q(x) → Ab(x) ∨ ¬P(x): Contrapositive of TH1",
        axioms=["(forall x (implies (not (Q x)) (or (Ab x) (not (P x)))))"],
        allowed_alpha_preds={"P", "R", "S"},  # Added R,S but still degenerate logically
        repaired_predicates={"Q"},
        enabled_for_generation=False,  # DISABLED: logically equivalent to TH1, adds no variety
        notes="Logically equivalent to TH1 - excluded from benchmark v0.2+",
    )
)


# TH9: Relational antecedent with universal consequent
# "If x R-relates to some y with P(y), and x is not abnormal, then all S-successors of x have Q"
# Repaired: Q (consequent)
# Allowed: P, R, S (after excluding Q)
# DISABLED in v0.6.1: all models collapse to the same Tier1 shortcut (exists y (and (R x y) (P y)))
_register_theory(
    TheorySpec(
        theory_id="TH9",
        description="∃y(R(x,y)∧P(y)) ∧ ¬Ab(x) → ∀z(S(x,z)→Q(z)): Relational antecedent + universal consequent",
        axioms=[
            "(forall x (implies (and (exists y (and (R x y) (P y))) (not (Ab x))) (forall z (implies (S x z) (Q z)))))"
        ],
        allowed_alpha_preds={"P", "R", "S"},  # Q is repaired (consequent)
        repaired_predicates={"Q"},
        enabled_for_generation=False,  # DISABLED v0.6.1: collapses to single Tier1 shortcut
        notes="v0.6+: DISABLED - all models find same shortcut (exists y (and (R x y) (P y)))",
    )
)


# TH10: S-based relational antecedent with R-based existential consequent
# "If x S-relates to some y with P(y), and x is not abnormal, then x R-relates to some z with Q(z)"
# Repaired: R, Q (consequent uses both R and Q)
# Allowed: P, S (after excluding R, Q)
_register_theory(
    TheorySpec(
        theory_id="TH10",
        description="∃y(S(x,y)∧P(y)) ∧ ¬Ab(x) → ∃z(R(x,z)∧Q(z)): S-antecedent, R-consequent (variant of TH7)",
        axioms=[
            "(forall x (implies (and (exists y (and (S x y) (P y))) (not (Ab x))) (exists z (and (R x z) (Q z)))))"
        ],
        allowed_alpha_preds={"P", "S"},  # R and Q are repaired (consequent)
        repaired_predicates={"R", "Q"},
        enabled_for_generation=True,
        notes="v0.6+: variant of TH7 with S in antecedent and R in consequent",
    )
)


# TH11: Relational antecedent with nested universal consequent
# "If x R-relates to some y with P(y), and x is not abnormal, then x S-relates to some z
#  such that all R-successors of z have P"
# This is TH7-like but with a nested universal in the consequent, making it harder.
# Repaired: S (consequent uses S to find z)
# The nested structure (forall w (implies (R z w) (P w))) is part of the consequent constraint.
# Allowed: P, Q, R (after excluding S)
_register_theory(
    TheorySpec(
        theory_id="TH11",
        description="∃y(R(x,y)∧P(y)) ∧ ¬Ab(x) → ∃z(S(x,z)∧∀w(R(z,w)→P(w))): Nested universal in consequent",
        axioms=[
            "(forall x (implies (and (exists y (and (R x y) (P y))) (not (Ab x))) (exists z (and (S x z) (forall w (implies (R z w) (P w)))))))"
        ],
        allowed_alpha_preds={"P", "Q", "R"},  # S is repaired (consequent uses S)
        repaired_predicates={"S"},
        enabled_for_generation=True,
        notes="v0.6.1+: TH7-like but with nested universal in consequent - harder than TH7",
    )
)


# TH12: Relational antecedent with universal consequent (v0.6.2)
# "If x R-relates to some y with P(y), and x is not abnormal, then all S-successors of x have Q"
# Similar to TH9 but with different structure to avoid the shortcut collapse.
# Repaired: Q (consequent)
# Allowed: P, R, S (after excluding Q)
_register_theory(
    TheorySpec(
        theory_id="TH12",
        description="∃y(R(x,y)∧P(y)) ∧ ¬Ab(x) → ∀z(S(x,z)→Q(z)): Relational antecedent + universal consequent",
        axioms=[
            "(forall x (implies (and (exists y (and (R x y) (P y))) (not (Ab x))) (forall z (implies (S x z) (Q z)))))"
        ],
        allowed_alpha_preds={"P", "R", "S"},  # Q is repaired (consequent)
        repaired_predicates={"Q"},
        enabled_for_generation=True,
        notes="v0.6.2+: relational antecedent with universal consequent - eligible for fast checking",
    )
)


# =============================================================================
# API Functions
# =============================================================================


def get_theory(theory_id: str) -> TheorySpec:
    """
    Get a theory specification by ID.

    Args:
        theory_id: The theory identifier (e.g., "TH1")

    Returns:
        TheorySpec for the requested theory

    Raises:
        KeyError: If theory_id is not found
    """
    if theory_id not in THEORY_LIBRARY:
        raise KeyError(f"Unknown theory: {theory_id}. Available: {list_theories()}")
    return THEORY_LIBRARY[theory_id]


def list_theories() -> List[str]:
    """
    List all available theory IDs.

    Returns:
        List of theory ID strings
    """
    return list(THEORY_LIBRARY.keys())


def list_enabled_theories() -> List[str]:
    """
    List theory IDs that are enabled for generation.

    Returns:
        List of enabled theory ID strings
    """
    return [tid for tid, spec in THEORY_LIBRARY.items() if spec.enabled_for_generation]


def get_enabled_theories() -> Dict[str, TheorySpec]:
    """
    Get all enabled theories as a dictionary.

    Returns:
        Dictionary mapping theory_id to TheorySpec for enabled theories only
    """
    return {tid: spec for tid, spec in THEORY_LIBRARY.items() if spec.enabled_for_generation}


def get_all_theories() -> Dict[str, TheorySpec]:
    """
    Get all theories as a dictionary.

    Returns:
        Dictionary mapping theory_id to TheorySpec
    """
    return THEORY_LIBRARY.copy()


def get_theories_by_complexity(max_axioms: int = 1, max_qd: int = 2) -> List[str]:
    """
    Get theories filtered by complexity.

    Args:
        max_axioms: Maximum number of axioms
        max_qd: Maximum quantifier depth (approximate)

    Returns:
        List of theory IDs matching criteria
    """

    # Simple heuristic: count nesting depth by counting 'forall'/'exists'
    def estimate_qd(axiom: str) -> int:
        return axiom.count("forall") + axiom.count("exists")

    result = []
    for tid, spec in THEORY_LIBRARY.items():
        if len(spec.axioms) <= max_axioms:
            max_axiom_qd = max(estimate_qd(ax) for ax in spec.axioms) if spec.axioms else 0
            if max_axiom_qd <= max_qd:
                result.append(tid)
    return result


# =============================================================================
# Predicate Sets
# =============================================================================

# Standard predicates used in the benchmark
UNARY_PREDICATES = {"P", "Q"}
BINARY_PREDICATES = {"R", "S"}
OBSERVED_PREDICATES = UNARY_PREDICATES | BINARY_PREDICATES
ABDUCIBLE_PREDICATE = "Ab"
ALL_PREDICATES = OBSERVED_PREDICATES | {ABDUCIBLE_PREDICATE}
