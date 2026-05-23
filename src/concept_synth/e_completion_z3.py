"""
e_completion_z3.py - Exact E Completion Checker using Z3 SAT

Implements correct EXISTS-completion semantics for Setup E:
For each world W, there exists a single completion assignment to W's unknown atoms
such that for all domain elements a, φ(a) matches the given T labels.

Completion is PER WORLD (not shared across worlds).

This module uses the shared grounding engine from fo_grounding_z3.py for:
- Unknown variable creation
- Atom semantics (CWA-with-unknown-exceptions)
- Formula grounding (finite quantifier unrolling)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

# Add parent to path for imports
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
try:
    import z3

    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    z3 = None

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
    Pred,
    Var,
)

# Import shared grounding engine
from .fo_grounding_z3 import GroundingContext, build_element_constraint
from .fo_grounding_z3 import build_known_atoms as _build_known_atoms_shared
from .fo_grounding_z3 import (
    build_label_match_constraints,
    build_match_constraints_from_world,
    build_unknown_atoms_set,
)
from .fo_grounding_z3 import build_unknown_vars as _build_unknown_vars_shared
from .fo_grounding_z3 import check_z3_available, create_grounding_context, create_solver
from .fo_grounding_z3 import eval_formula_direct as _eval_formula_direct_shared
from .fo_grounding_z3 import eval_formula_under_completion as _eval_formula_under_completion_shared
from .fo_grounding_z3 import ground_axioms_to_z3
from .fo_grounding_z3 import ground_formula_to_z3 as _ground_formula_shared

# Aliases for consistency with other code
Const = Constant
Atom = Pred
Not = FONot
And = FOAnd
Or = FOOr
Implies = FOImplies


@dataclass
class CompletionResult:
    """Result of checking E completion."""

    satisfiable: bool
    model: Optional[Any] = None  # Z3 model if SAT
    reason: Optional[str] = None  # Explanation if UNSAT

    # Local feasibility diagnostics (Task C)
    local_feasibility: Optional[Dict[str, bool]] = None  # element -> SAT for element alone
    local_unsat_elements: Optional[List[str]] = None  # Elements with UNSAT constraints alone
    failure_classification: Optional[str] = None  # "local_impossible" or "global_inconsistent"
    unsat_core_elements: Optional[List[str]] = None  # Elements in unsat core (if computed)

    # Completion sensitivity diagnostics
    completion_sensitive: Optional[bool] = None  # True if formula differs under extreme completions
    sensitive_elements_count: Optional[int] = None  # Number of elements that differ

    # Forced truth diagnostics for local_impossible failures
    pos_unachievable_count: Optional[int] = None  # Elements where label=True but cannot make true
    neg_unachievable_count: Optional[int] = None  # Elements where label=False but cannot make false
    pos_unachievable_elements: Optional[List[str]] = None  # Which elements are pos_unachievable
    neg_unachievable_elements: Optional[List[str]] = None  # Which elements are neg_unachievable

    # Directional flip profile
    flips_f_to_t: Optional[int] = None  # Elements flipping False->True
    flips_t_to_f: Optional[int] = None  # Elements flipping True->False
    nonmonotone: Optional[bool] = None  # True if any T->F flips

    # Label counts for normalization
    num_pos_labels: Optional[int] = None  # Number of elements with label=True
    num_neg_labels: Optional[int] = None  # Number of elements with label=False
    num_local_unsat: Optional[int] = None  # Number of locally UNSAT elements


def build_unknown_vars(
    world: Dict[str, Any], world_id: str
) -> Dict[Tuple[str, Tuple[str, ...]], Any]:
    """
    Build Z3 boolean variables for unknown atoms in a world.

    This is a thin wrapper around the shared grounding module.

    Args:
        world: World dictionary with 'unknownAtoms' field
        world_id: Unique identifier for this world (for variable naming)

    Returns:
        Dictionary mapping (pred_name, args_tuple) to Z3 BoolRef
    """
    return _build_unknown_vars_shared(world, world_id)


def build_known_atoms(
    world: Dict[str, Any]
) -> Tuple[Set[Tuple[str, Tuple[str, ...]]], Set[Tuple[str, Tuple[str, ...]]]]:
    """
    Build sets of known true and known false atoms from world predicates.

    This is a thin wrapper around the shared grounding module.

    Returns:
        Tuple of (known_true_set, known_false_set)
    """
    return _build_known_atoms_shared(world)


def atom_to_z3(
    pred_name: str,
    args: Tuple[str, ...],
    known_true: Set[Tuple[str, Tuple[str, ...]]],
    known_false: Set[Tuple[str, Tuple[str, ...]]],
    unknown_vars: Dict[Tuple[str, Tuple[str, ...]], Any],
) -> Any:
    """
    Convert an atom to Z3 expression using CWA-with-unknown-exceptions.

    This is a compatibility wrapper that creates a temporary GroundingContext
    and delegates to the shared module. For new code, use the shared module
    directly with a GroundingContext.

    Truth rules:
    - If atom in known_true: BoolVal(True)
    - If atom in known_false: BoolVal(False)
    - If atom in unknown set: unknown_vars[(pred, args)]
    - Else (CWA): BoolVal(False)
    """
    if not Z3_AVAILABLE:
        raise RuntimeError("Z3 is not installed")

    # Create a minimal context for the shared function
    from .fo_grounding_z3 import atom_to_z3 as _atom_to_z3_shared

    ctx = GroundingContext(
        domain=[],  # Not needed for atom evaluation
        known_true=known_true,
        known_false=known_false,
        unknown_atoms={},  # Not needed
        unknown_vars=unknown_vars,
        pred_bindings={},
        cache={},
    )
    return _atom_to_z3_shared(pred_name, args, ctx)


def eval_formula_to_z3(
    node: FOFormula,
    env: Dict[str, str],  # Variable name -> domain constant
    domain: List[str],
    known_true: Set[Tuple[str, Tuple[str, ...]]],
    known_false: Set[Tuple[str, Tuple[str, ...]]],
    unknown_vars: Dict[Tuple[str, Tuple[str, ...]], Any],
    cache: Optional[Dict[Tuple[int, FrozenSet], Any]] = None,
) -> Any:
    """
    Recursively convert FOL formula to Z3 expression.

    This is a compatibility wrapper that creates a temporary GroundingContext
    and delegates to the shared module. For new code, use ground_formula_to_z3
    from fo_grounding_z3 directly with a GroundingContext.

    Supports: atoms, (= x y), not, and, or, implies, forall, exists
    Uses finite unrolling for quantifiers over the domain.

    Args:
        node: FOL formula AST node
        env: Current variable bindings (var_name -> domain_constant)
        domain: List of domain constants
        known_true: Set of known true atoms
        known_false: Set of known false atoms
        unknown_vars: Z3 variables for unknown atoms
        cache: Memoization cache keyed by (node_id, frozen_env)

    Returns:
        Z3 BoolRef expression
    """
    if not Z3_AVAILABLE:
        raise RuntimeError("Z3 is not installed")

    # Create a GroundingContext for the shared function
    ctx = GroundingContext(
        domain=domain,
        known_true=known_true,
        known_false=known_false,
        unknown_atoms={},  # Not needed for grounding
        unknown_vars=unknown_vars,
        pred_bindings={},
        cache=cache if cache is not None else {},
    )

    return _ground_formula_shared(node, env, ctx)


def build_match_constraints(
    world: Dict[str, Any],
    formula_ast: FOFormula,
    unknown_vars: Dict[Tuple[str, Tuple[str, ...]], Any],
) -> Any:
    """
    Build Z3 constraints requiring formula to match target extension.

    This is a compatibility wrapper that creates a GroundingContext and
    delegates to the shared module.

    For each element a in domain:
    - If a in T_true: require φ(a) = True
    - If a in T_false (or not in T_true): require φ(a) = False

    Returns:
        Z3 And expression of all constraints
    """
    if not Z3_AVAILABLE:
        raise RuntimeError("Z3 is not installed")

    domain = world.get("domain", [])
    known_true, known_false = build_known_atoms(world)

    # Create grounding context
    ctx = GroundingContext(
        domain=domain,
        known_true=known_true,
        known_false=known_false,
        unknown_atoms=build_unknown_atoms_set(world),
        unknown_vars=unknown_vars,
        pred_bindings={},
        cache={},
    )

    return build_match_constraints_from_world(world, formula_ast, ctx)


def check_element_feasibility(
    world: Dict[str, Any],
    formula_ast: FOFormula,
    element: str,
    unknown_vars: Dict[Tuple[str, Tuple[str, ...]], Any],
    timeout_ms: int = 1000,
) -> bool:
    """
    Check if a single element's constraint is satisfiable alone.

    This tests whether there exists ANY completion making the formula
    match the target label for just this one element.

    Args:
        world: World dictionary
        formula_ast: Parsed formula AST
        element: Domain element to check
        unknown_vars: Z3 variables for unknown atoms
        timeout_ms: Timeout

    Returns:
        True if element constraint alone is SAT
    """
    if not Z3_AVAILABLE:
        return True  # Assume SAT if Z3 not available

    domain = world.get("domain", [])
    target = world.get("targetExtension", {})
    t_true_set = set(target.get("T_true", []))

    known_true, known_false = build_known_atoms(world)

    # Build constraint for just this element
    env = {"x": element}
    cache = {}

    formula_expr = eval_formula_to_z3(
        formula_ast, env, domain, known_true, known_false, unknown_vars, cache
    )

    if element in t_true_set:
        constraint = formula_expr
    else:
        constraint = z3.Not(formula_expr)

    solver = z3.Solver()
    solver.set("timeout", timeout_ms)
    solver.add(constraint)

    result = solver.check()
    return result == z3.sat


def check_local_feasibility(
    world: Dict[str, Any],
    formula_ast: FOFormula,
    world_id: Optional[str] = None,
    timeout_ms: int = 1000,
) -> Tuple[Dict[str, bool], List[str]]:
    """
    Check local feasibility for each element separately.

    Returns:
        Tuple of (element_to_sat_dict, list_of_unsat_elements)
    """
    if not Z3_AVAILABLE:
        return {}, []

    if world_id is None:
        world_id = world.get("worldId", "W")

    unknown_vars = build_unknown_vars(world, world_id)
    domain = world.get("domain", [])

    local_sat = {}
    unsat_elements = []

    for elem in domain:
        sat = check_element_feasibility(world, formula_ast, elem, unknown_vars, timeout_ms)
        local_sat[elem] = sat
        if not sat:
            unsat_elements.append(elem)

    return local_sat, unsat_elements


def eval_formula_under_fixed_completion(
    world: Dict[str, Any], formula_ast: FOFormula, all_unknowns_true: bool
) -> Dict[str, bool]:
    """
    Evaluate formula at each domain element under a fixed completion.

    This is a thin wrapper around the shared grounding module.

    This does NOT use Z3 - it's a direct evaluation where all unknown atoms
    are assigned the same truth value.

    Args:
        world: World dictionary
        formula_ast: Parsed formula AST
        all_unknowns_true: If True, all unknown atoms = True; else all = False

    Returns:
        Dictionary mapping element -> truth value of formula at that element
    """
    return _eval_formula_under_completion_shared(world, formula_ast, all_unknowns_true)


def _eval_formula_direct(
    node: FOFormula,
    env: Dict[str, str],
    domain: List[str],
    known_true: Set[Tuple[str, Tuple[str, ...]]],
    known_false: Set[Tuple[str, Tuple[str, ...]]],
) -> bool:
    """
    Direct evaluation of formula without Z3.

    This is a thin wrapper around the shared grounding module.
    Uses closed-world assumption: atoms not in known_true are False.
    """
    return _eval_formula_direct_shared(node, env, domain, known_true, known_false)


def compute_completion_sensitivity(
    world: Dict[str, Any], formula_ast: FOFormula
) -> Tuple[bool, int]:
    """
    Check if formula is sensitive to completion assignment.

    Evaluates formula under two extreme completions:
    - completion0: all unknown atoms = False
    - completion1: all unknown atoms = True

    Returns:
        Tuple of (is_sensitive, count_of_differing_elements)
    """
    # Evaluate under both extreme completions
    eval_false = eval_formula_under_fixed_completion(world, formula_ast, all_unknowns_true=False)
    eval_true = eval_formula_under_fixed_completion(world, formula_ast, all_unknowns_true=True)

    # Count elements that differ
    differing_count = 0
    for elem in eval_false:
        if eval_false.get(elem) != eval_true.get(elem):
            differing_count += 1

    return (differing_count > 0, differing_count)


@dataclass
class FlipProfile:
    """Directional flip profile for completion sensitivity."""

    flips_f_to_t: int  # Elements that flip False->True when unknowns go False->True
    flips_t_to_f: int  # Elements that flip True->False when unknowns go False->True
    nonmonotone: bool  # True if any T->F flips exist
    total_elements: int


def compute_flip_profile(world: Dict[str, Any], formula_ast: FOFormula) -> FlipProfile:
    """
    Compute directional flip profile for completion sensitivity.

    Evaluates formula under two extreme completions:
    - observed: all unknown atoms = False
    - optimistic: all unknown atoms = True

    For each element, tracks direction of flip:
    - F->T: formula goes from False to True when unknowns become True
    - T->F: formula goes from True to False when unknowns become True

    A formula is "nonmonotone" if any T->F flips exist (formula becomes
    less permissive when unknowns are added).

    Returns:
        FlipProfile with directional counts
    """
    # Evaluate under observed (all unknowns False)
    eval_observed = eval_formula_under_fixed_completion(world, formula_ast, all_unknowns_true=False)
    # Evaluate under optimistic (all unknowns True)
    eval_optimistic = eval_formula_under_fixed_completion(
        world, formula_ast, all_unknowns_true=True
    )

    flips_f_to_t = 0
    flips_t_to_f = 0

    for elem in eval_observed:
        obs = eval_observed.get(elem, False)
        opt = eval_optimistic.get(elem, False)

        if not obs and opt:
            flips_f_to_t += 1
        elif obs and not opt:
            flips_t_to_f += 1

    return FlipProfile(
        flips_f_to_t=flips_f_to_t,
        flips_t_to_f=flips_t_to_f,
        nonmonotone=(flips_t_to_f > 0),
        total_elements=len(eval_observed),
    )


@dataclass
class FormulaStructure:
    """Structural analysis of a formula."""

    ast_size: int
    quantifier_depth: int
    num_forall: int
    num_exists: int
    num_alternations: int
    has_forall: bool
    has_forall_exists_pattern: bool  # forall with exists in body
    x_dependent: bool  # x occurs in predicate position


def analyze_formula_structure(formula_ast: FOFormula) -> FormulaStructure:
    """
    Analyze structural properties of a formula.

    Returns:
        FormulaStructure with counts and flags
    """

    # Count AST size
    def count_nodes(node):
        if isinstance(node, (Pred, Eq)):
            return 1
        elif isinstance(node, FONot):
            return 1 + count_nodes(node.child)
        elif isinstance(node, (FOAnd, FOOr, FOImplies, FOBiconditional)):
            return 1 + count_nodes(node.left) + count_nodes(node.right)
        elif isinstance(node, (Forall, Exists)):
            return 1 + count_nodes(node.body)
        return 1

    # Count quantifier depth
    def max_qd(node, depth=0):
        if isinstance(node, (Pred, Eq)):
            return depth
        elif isinstance(node, FONot):
            return max_qd(node.child, depth)
        elif isinstance(node, (FOAnd, FOOr, FOImplies, FOBiconditional)):
            return max(max_qd(node.left, depth), max_qd(node.right, depth))
        elif isinstance(node, (Forall, Exists)):
            return max_qd(node.body, depth + 1)
        return depth

    # Count forall/exists
    def count_quantifiers(node):
        forall_count = 0
        exists_count = 0

        if isinstance(node, Forall):
            forall_count = 1
            sub_f, sub_e = count_quantifiers(node.body)
            return forall_count + sub_f, sub_e
        elif isinstance(node, Exists):
            exists_count = 1
            sub_f, sub_e = count_quantifiers(node.body)
            return sub_f, exists_count + sub_e
        elif isinstance(node, FONot):
            return count_quantifiers(node.child)
        elif isinstance(node, (FOAnd, FOOr, FOImplies, FOBiconditional)):
            f1, e1 = count_quantifiers(node.left)
            f2, e2 = count_quantifiers(node.right)
            return f1 + f2, e1 + e2
        return 0, 0

    # Count alternations
    def count_alternations(node, last_quant=None):
        if isinstance(node, Forall):
            alt = 1 if last_quant == "exists" else 0
            return alt + count_alternations(node.body, "forall")
        elif isinstance(node, Exists):
            alt = 1 if last_quant == "forall" else 0
            return alt + count_alternations(node.body, "exists")
        elif isinstance(node, FONot):
            return count_alternations(node.child, last_quant)
        elif isinstance(node, (FOAnd, FOOr, FOImplies, FOBiconditional)):
            return max(
                count_alternations(node.left, last_quant),
                count_alternations(node.right, last_quant),
            )
        return 0

    # Check for forall-exists pattern
    def has_forall_exists(node, in_forall=False):
        if isinstance(node, Forall):
            return has_forall_exists(node.body, True)
        elif isinstance(node, Exists):
            if in_forall:
                return True
            return has_forall_exists(node.body, False)
        elif isinstance(node, FONot):
            return has_forall_exists(node.child, in_forall)
        elif isinstance(node, (FOAnd, FOOr, FOImplies, FOBiconditional)):
            return has_forall_exists(node.left, in_forall) or has_forall_exists(
                node.right, in_forall
            )
        return False

    # Check if x is used in predicate position
    def uses_x(node):
        if isinstance(node, Pred):
            for arg in node.args:
                if isinstance(arg, Var) and arg.name == "x":
                    return True
            return False
        elif isinstance(node, Eq):
            if isinstance(node.left, Var) and node.left.name == "x":
                return True
            if isinstance(node.right, Var) and node.right.name == "x":
                return True
            return False
        elif isinstance(node, FONot):
            return uses_x(node.child)
        elif isinstance(node, (FOAnd, FOOr, FOImplies, FOBiconditional)):
            return uses_x(node.left) or uses_x(node.right)
        elif isinstance(node, (Forall, Exists)):
            return uses_x(node.body)
        return False

    num_forall, num_exists = count_quantifiers(formula_ast)

    return FormulaStructure(
        ast_size=count_nodes(formula_ast),
        quantifier_depth=max_qd(formula_ast),
        num_forall=num_forall,
        num_exists=num_exists,
        num_alternations=count_alternations(formula_ast),
        has_forall=(num_forall > 0),
        has_forall_exists_pattern=has_forall_exists(formula_ast),
        x_dependent=uses_x(formula_ast),
    )


def check_can_make_true_false(
    world: Dict[str, Any],
    formula_ast: FOFormula,
    element: str,
    world_id: Optional[str] = None,
    timeout_ms: int = 1000,
) -> Tuple[bool, bool]:
    """
    Check if formula can be made true and/or false at an element.

    Uses Z3 to check:
    - SAT(base ∧ φ(element)) - can the formula be made true?
    - SAT(base ∧ ¬φ(element)) - can the formula be made false?

    Args:
        world: World dictionary
        formula_ast: Parsed formula AST
        element: Domain element to check
        world_id: Optional world identifier
        timeout_ms: Timeout

    Returns:
        Tuple of (can_be_true, can_be_false)
    """
    if not Z3_AVAILABLE:
        return (True, True)  # Assume both possible if Z3 not available

    if world_id is None:
        world_id = world.get("worldId", "W")

    domain = world.get("domain", [])
    unknown_vars = build_unknown_vars(world, world_id)
    known_true, known_false = build_known_atoms(world)

    # Build formula expression for this element
    env = {"x": element}
    cache = {}
    formula_expr = eval_formula_to_z3(
        formula_ast, env, domain, known_true, known_false, unknown_vars, cache
    )

    # Check if can be true
    solver_true = z3.Solver()
    solver_true.set("timeout", timeout_ms)
    solver_true.add(formula_expr)
    can_be_true = solver_true.check() == z3.sat

    # Check if can be false
    solver_false = z3.Solver()
    solver_false.set("timeout", timeout_ms)
    solver_false.add(z3.Not(formula_expr))
    can_be_false = solver_false.check() == z3.sat

    return (can_be_true, can_be_false)


def compute_forced_truth_diagnostics(
    world: Dict[str, Any],
    formula_ast: FOFormula,
    local_unsat_elements: List[str],
    world_id: Optional[str] = None,
    max_elements: int = 10,
    timeout_ms: int = 1000,
) -> Tuple[int, int, List[str], List[str]]:
    """
    For local_impossible failures, classify why each element is UNSAT.

    For each locally UNSAT element:
    - If label=True and cannot make true: pos_unachievable
    - If label=False and cannot make false: neg_unachievable

    Args:
        world: World dictionary
        formula_ast: Parsed formula AST
        local_unsat_elements: List of elements with UNSAT constraints
        world_id: Optional world identifier
        max_elements: Max elements to check (for performance)
        timeout_ms: Timeout per check

    Returns:
        Tuple of (pos_unachievable_count, neg_unachievable_count,
                  pos_unachievable_elements, neg_unachievable_elements)
    """
    target = world.get("targetExtension", {})
    t_true_set = set(target.get("T_true", []))

    pos_unachievable = []
    neg_unachievable = []

    # Cap the number of elements to check
    elements_to_check = local_unsat_elements[:max_elements]

    for elem in elements_to_check:
        can_be_true, can_be_false = check_can_make_true_false(
            world, formula_ast, elem, world_id, timeout_ms
        )

        target_label = elem in t_true_set

        if target_label:  # Label is True
            if not can_be_true:
                pos_unachievable.append(elem)
        else:  # Label is False
            if not can_be_false:
                neg_unachievable.append(elem)

    return (len(pos_unachievable), len(neg_unachievable), pos_unachievable, neg_unachievable)


def check_world_exists_completion(
    world: Dict[str, Any],
    formula_ast: FOFormula,
    world_id: Optional[str] = None,
    axioms_ast_list: Optional[List[FOFormula]] = None,
    timeout_ms: int = 5000,
    compute_diagnostics: bool = False,
) -> CompletionResult:
    """
    Check if there exists a completion of unknown atoms making formula match target.

    This implements exact EXISTS-completion semantics for a single world.

    Args:
        world: World dictionary
        formula_ast: Parsed formula AST
        world_id: Optional world identifier (for variable naming)
        axioms_ast_list: Optional list of axiom formulas to also satisfy
        timeout_ms: Z3 solver timeout in milliseconds

    Returns:
        CompletionResult with satisfiable flag and optional model
    """
    if not Z3_AVAILABLE:
        return CompletionResult(satisfiable=False, reason="Z3 not available")

    # Use world_id from world dict if not provided
    if world_id is None:
        world_id = world.get("worldId", "W")

    # Build unknown variables
    unknown_vars = build_unknown_vars(world, world_id)

    # Build match constraints
    match_constraints = build_match_constraints(world, formula_ast, unknown_vars)

    # Create solver
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)

    # Add match constraints
    solver.add(match_constraints)

    # Add axiom constraints if provided
    if axioms_ast_list:
        domain = world.get("domain", [])
        known_true, known_false = build_known_atoms(world)
        cache = {}

        for axiom in axioms_ast_list:
            # Axioms are typically closed formulas (no free variables)
            # But if they have free x, we require them for all elements
            axiom_expr = eval_formula_to_z3(
                axiom, {}, domain, known_true, known_false, unknown_vars, cache
            )
            solver.add(axiom_expr)

    # Check satisfiability
    result = solver.check()

    if result == z3.sat:
        # Compute completion sensitivity even for SAT cases if diagnostics requested
        completion_sensitive = None
        sensitive_elements_count = None
        flips_f_to_t = None
        flips_t_to_f = None
        nonmonotone = None
        num_pos_labels = None
        num_neg_labels = None

        if compute_diagnostics:
            try:
                completion_sensitive, sensitive_elements_count = compute_completion_sensitivity(
                    world, formula_ast
                )
                # Compute flip profile
                flip_profile = compute_flip_profile(world, formula_ast)
                flips_f_to_t = flip_profile.flips_f_to_t
                flips_t_to_f = flip_profile.flips_t_to_f
                nonmonotone = flip_profile.nonmonotone

                # Count labels
                target = world.get("targetExtension", {})
                t_true_set = set(target.get("T_true", []))
                domain = world.get("domain", [])
                num_pos_labels = len(t_true_set)
                num_neg_labels = len(domain) - num_pos_labels
            except Exception:
                pass  # Skip on error

        return CompletionResult(
            satisfiable=True,
            model=solver.model(),
            completion_sensitive=completion_sensitive,
            sensitive_elements_count=sensitive_elements_count,
            flips_f_to_t=flips_f_to_t,
            flips_t_to_f=flips_t_to_f,
            nonmonotone=nonmonotone,
            num_pos_labels=num_pos_labels,
            num_neg_labels=num_neg_labels,
        )
    elif result == z3.unsat:
        # Compute diagnostics if requested
        local_feasibility = None
        local_unsat_elements = None
        failure_classification = None
        completion_sensitive = None
        sensitive_elements_count = None
        pos_unachievable_count = None
        neg_unachievable_count = None
        pos_unachievable_elements = None
        neg_unachievable_elements = None
        flips_f_to_t = None
        flips_t_to_f = None
        nonmonotone = None
        num_pos_labels = None
        num_neg_labels = None
        num_local_unsat = None

        if compute_diagnostics:
            # Count labels
            target = world.get("targetExtension", {})
            t_true_set = set(target.get("T_true", []))
            domain = world.get("domain", [])
            num_pos_labels = len(t_true_set)
            num_neg_labels = len(domain) - num_pos_labels

            # Compute completion sensitivity and flip profile
            try:
                completion_sensitive, sensitive_elements_count = compute_completion_sensitivity(
                    world, formula_ast
                )
                # Compute flip profile
                flip_profile = compute_flip_profile(world, formula_ast)
                flips_f_to_t = flip_profile.flips_f_to_t
                flips_t_to_f = flip_profile.flips_t_to_f
                nonmonotone = flip_profile.nonmonotone
            except Exception:
                pass

            # Check local feasibility
            local_feasibility, local_unsat_elements = check_local_feasibility(
                world, formula_ast, world_id, timeout_ms=1000
            )

            if local_unsat_elements:
                failure_classification = "local_impossible"
                num_local_unsat = len(local_unsat_elements)

                # Compute forced truth diagnostics
                try:
                    (
                        pos_unachievable_count,
                        neg_unachievable_count,
                        pos_unachievable_elements,
                        neg_unachievable_elements,
                    ) = compute_forced_truth_diagnostics(
                        world,
                        formula_ast,
                        local_unsat_elements,
                        world_id,
                        max_elements=10,
                        timeout_ms=1000,
                    )
                except Exception:
                    pass
            else:
                failure_classification = "global_inconsistent"
                num_local_unsat = 0

        return CompletionResult(
            satisfiable=False,
            reason="No valid completion exists",
            local_feasibility=local_feasibility,
            local_unsat_elements=local_unsat_elements,
            failure_classification=failure_classification,
            completion_sensitive=completion_sensitive,
            sensitive_elements_count=sensitive_elements_count,
            pos_unachievable_count=pos_unachievable_count,
            neg_unachievable_count=neg_unachievable_count,
            pos_unachievable_elements=pos_unachievable_elements,
            neg_unachievable_elements=neg_unachievable_elements,
            flips_f_to_t=flips_f_to_t,
            flips_t_to_f=flips_t_to_f,
            nonmonotone=nonmonotone,
            num_pos_labels=num_pos_labels,
            num_neg_labels=num_neg_labels,
            num_local_unsat=num_local_unsat,
        )
    else:
        # Timeout or unknown
        return CompletionResult(satisfiable=False, reason=f"Solver returned: {result}")


def check_instance_exists_completion(
    worlds: List[Dict[str, Any]],
    formula_ast: FOFormula,
    axioms_ast_list: Optional[List[FOFormula]] = None,
    timeout_ms: int = 5000,
) -> Tuple[bool, List[CompletionResult]]:
    """
    Check EXISTS-completion for all worlds in an instance.

    Completion is PER WORLD - each world can have a different completion.
    Returns True iff all worlds have a valid completion.

    Args:
        worlds: List of world dictionaries
        formula_ast: Parsed formula AST
        axioms_ast_list: Optional axioms
        timeout_ms: Timeout per world

    Returns:
        Tuple of (all_satisfied, list_of_per_world_results)
    """
    results = []
    all_satisfied = True

    for world in worlds:
        world_id = world.get("worldId", f"W{len(results)}")
        result = check_world_exists_completion(
            world, formula_ast, world_id, axioms_ast_list, timeout_ms
        )
        results.append(result)

        if not result.satisfiable:
            all_satisfied = False

    return all_satisfied, results


# =============================================================================
# Convenience functions for generation/CEGIS
# =============================================================================


def formula_matches_target_e_exact(
    formula_ast: FOFormula, world: Dict[str, Any], timeout_ms: int = 5000
) -> bool:
    """
    Check if a formula matches target labels under exact E semantics.

    This is the E-scenario equivalent of formula_matches_target() used in AD/C.
    Use this in CEGIS and identifiability checks for E scenario.

    Args:
        formula_ast: Formula to check
        world: World dictionary with unknownAtoms and targetExtension
        timeout_ms: Timeout for Z3

    Returns:
        True if there exists a completion making formula match target
    """
    result = check_world_exists_completion(world, formula_ast, timeout_ms=timeout_ms)
    return result.satisfiable


def check_e_distractor_broken(
    distractor_formula: FOFormula, world: Dict[str, Any], timeout_ms: int = 5000
) -> bool:
    """
    Check if a distractor is broken (doesn't match target) under E semantics.

    A distractor is "broken" if there is NO completion making it match the target.

    Returns:
        True if distractor is broken (UNSAT)
    """
    return not formula_matches_target_e_exact(distractor_formula, world, timeout_ms)


def check_e_mutant_separated(
    gold_formula: FOFormula,
    mutant_formula: FOFormula,
    world: Dict[str, Any],
    timeout_ms: int = 5000,
) -> bool:
    """
    Check if a mutant is separated from gold under E semantics.

    Separation means: gold matches target but mutant doesn't (or vice versa).

    Returns:
        True if mutant is separated from gold in this world
    """
    gold_matches = formula_matches_target_e_exact(gold_formula, world, timeout_ms)
    mutant_matches = formula_matches_target_e_exact(mutant_formula, world, timeout_ms)
    return gold_matches != mutant_matches


# =============================================================================
# Legacy approximation method (for comparison/debugging)
# =============================================================================


def check_world_approx_two_extremes(
    world: Dict[str, Any], formula_ast: FOFormula
) -> Tuple[bool, Optional[str], Optional[List[str]]]:
    """
    Legacy two-extremes approximation for E completion.

    This is the OLD method that checks:
    - OBSERVED model (unknowns = false)
    - OPTIMISTIC model (unknowns = true)

    For each element:
    - If T=true: formula must be true in at least one model
    - If T=false: formula must be false in at least one model

    This is UNSOUND (can pass when exact fails) and INCOMPLETE (can fail when exact passes).
    Kept for comparison and debugging only.
    """
    from .evaluate_results import (
        build_model_from_world,
        build_optimistic_model_from_world,
        compute_target_extension,
    )
    from .target import compute_target_extension

    domain = world.get("domain", [])
    target = world.get("targetExtension", {})
    t_true_set = set(target.get("T_true", []))
    t_false_set = set(domain) - t_true_set

    # Build both extreme models
    observed_model = build_model_from_world(world, use_full_predicates=False)
    optimistic_model = build_optimistic_model_from_world(world)

    # Evaluate formula on both
    observed_result = compute_target_extension(observed_model, formula_ast)
    optimistic_result = compute_target_extension(optimistic_model, formula_ast)

    observed_true = set(observed_result.T_true)
    optimistic_true = set(optimistic_result.T_true)

    false_negatives = []
    false_positives = []

    for elem in t_true_set:
        if elem not in observed_true and elem not in optimistic_true:
            false_negatives.append(elem)

    for elem in t_false_set:
        if elem in observed_true and elem in optimistic_true:
            false_positives.append(elem)

    if not false_negatives and not false_positives:
        return True, None, None

    failed_elements = false_positives + false_negatives
    explanation_parts = []
    if false_positives:
        explanation_parts.append(f"Formula always true for T=false: {false_positives}")
    if false_negatives:
        explanation_parts.append(f"Formula always false for T=true: {false_negatives}")

    return False, "; ".join(explanation_parts), failed_elements


# =============================================================================
# Semantics mode enum
# =============================================================================


class ESemantics:
    """E scenario semantics modes."""

    EXACT_EXISTS = "exact_exists"  # Default: exact Z3 EXISTS-completion
    APPROX_TWO_EXTREMES = "approx_two_extremes"  # Legacy approximation
    ALL_COMPLETIONS = "all_completions"  # Robust: require ALL completions work


def check_e_scenario(
    worlds: List[Dict[str, Any]],
    formula_ast: FOFormula,
    semantics: str = ESemantics.EXACT_EXISTS,
    axioms_ast_list: Optional[List[FOFormula]] = None,
    timeout_ms: int = 5000,
    compute_diagnostics: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Check E scenario with specified semantics.

    Args:
        worlds: List of world dictionaries
        formula_ast: Parsed formula AST
        semantics: One of ESemantics values
        axioms_ast_list: Optional axioms
        timeout_ms: Z3 timeout per world
        compute_diagnostics: If True, compute local feasibility diagnostics for failures

    Returns:
        Tuple of (is_correct, metadata_dict)
    """
    meta = {"e_semantics_used": semantics, "worlds_checked": len(worlds)}

    if semantics == ESemantics.EXACT_EXISTS:
        if not Z3_AVAILABLE:
            # Fall back to approximation with warning
            meta["fallback_reason"] = "Z3 not available"
            semantics = ESemantics.APPROX_TWO_EXTREMES
            meta["e_semantics_used"] = semantics

    if semantics == ESemantics.EXACT_EXISTS:
        # Check each world with optional diagnostics
        results = []
        all_sat = True
        worlds_timed_out = 0

        for world in worlds:
            world_id = world.get("worldId", f"W{len(results)}")
            # Only compute diagnostics for failing worlds
            result = check_world_exists_completion(
                world,
                formula_ast,
                world_id,
                axioms_ast_list,
                timeout_ms,
                compute_diagnostics=False,  # First pass without diagnostics
            )
            results.append(result)
            if not result.satisfiable:
                all_sat = False
                # Check if this was a timeout
                if result.reason and "unknown" in result.reason.lower():
                    worlds_timed_out += 1

        meta["e_sat"] = all_sat
        meta["per_world_sat"] = [r.satisfiable for r in results]
        meta["worlds_timed_out"] = worlds_timed_out
        meta["timed_out"] = worlds_timed_out > 0

        if not all_sat:
            # Find first failing world and compute diagnostics if requested
            for i, r in enumerate(results):
                if not r.satisfiable:
                    world = worlds[i]
                    world_id = world.get("worldId", f"W{i}")
                    meta["first_failing_world"] = world_id
                    meta["failure_reason"] = r.reason

                    # Compute diagnostics for this failing world
                    if compute_diagnostics:
                        detailed_result = check_world_exists_completion(
                            world,
                            formula_ast,
                            world_id,
                            axioms_ast_list,
                            timeout_ms,
                            compute_diagnostics=True,
                        )
                        meta["failure_classification"] = detailed_result.failure_classification
                        meta["local_unsat_elements"] = detailed_result.local_unsat_elements
                        if detailed_result.local_unsat_elements:
                            meta["example_local_unsat"] = detailed_result.local_unsat_elements[0]

                        # Add completion sensitivity diagnostics
                        meta["completion_sensitive"] = detailed_result.completion_sensitive
                        meta["sensitive_elements_count"] = detailed_result.sensitive_elements_count

                        # Add forced truth diagnostics
                        meta["pos_unachievable_count"] = detailed_result.pos_unachievable_count
                        meta["neg_unachievable_count"] = detailed_result.neg_unachievable_count
                        meta["pos_unachievable_elements"] = (
                            detailed_result.pos_unachievable_elements
                        )
                        meta["neg_unachievable_elements"] = (
                            detailed_result.neg_unachievable_elements
                        )

                        # Add flip profile diagnostics
                        meta["flips_f_to_t"] = detailed_result.flips_f_to_t
                        meta["flips_t_to_f"] = detailed_result.flips_t_to_f
                        meta["nonmonotone"] = detailed_result.nonmonotone

                        # Add label counts for normalization
                        meta["num_pos_labels"] = detailed_result.num_pos_labels
                        meta["num_neg_labels"] = detailed_result.num_neg_labels
                        meta["num_local_unsat"] = detailed_result.num_local_unsat
                    break
        else:
            # Also compute completion sensitivity for successful cases if diagnostics requested
            if compute_diagnostics and worlds:
                first_world = worlds[0]
                world_id = first_world.get("worldId", "W0")
                detailed_result = check_world_exists_completion(
                    first_world,
                    formula_ast,
                    world_id,
                    axioms_ast_list,
                    timeout_ms,
                    compute_diagnostics=True,
                )
                meta["completion_sensitive"] = detailed_result.completion_sensitive
                meta["sensitive_elements_count"] = detailed_result.sensitive_elements_count

                # Add flip profile diagnostics for successful cases too
                meta["flips_f_to_t"] = detailed_result.flips_f_to_t
                meta["flips_t_to_f"] = detailed_result.flips_t_to_f
                meta["nonmonotone"] = detailed_result.nonmonotone

                # Add label counts
                meta["num_pos_labels"] = detailed_result.num_pos_labels
                meta["num_neg_labels"] = detailed_result.num_neg_labels

        return all_sat, meta

    elif semantics == ESemantics.APPROX_TWO_EXTREMES:
        # Use legacy approximation
        all_passed = True
        per_world_results = []

        for world in worlds:
            passed, explanation, failed = check_world_approx_two_extremes(world, formula_ast)
            per_world_results.append(passed)
            if not passed:
                all_passed = False
                if "first_failing_world" not in meta:
                    meta["first_failing_world"] = world.get("worldId", "unknown")
                    meta["failure_reason"] = explanation

        meta["e_sat"] = all_passed
        meta["per_world_sat"] = per_world_results
        return all_passed, meta

    elif semantics == ESemantics.ALL_COMPLETIONS:
        # Robust semantics: require formula works for ALL completions
        # This is UNSAT check of (constraints ∧ ¬match)
        # For now, not fully implemented - would need negation of match constraints
        meta["error"] = "ALL_COMPLETIONS not yet implemented"
        return False, meta

    else:
        meta["error"] = f"Unknown semantics: {semantics}"
        return False, meta
