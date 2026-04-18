"""
abd_b1_z3_checker.py - Z3 SAT/OPT Checker for ABD-B1 Abduction Tasks

Provides Z3-based checking and optimization for ABD-Full, ABD-Partial, and ABD-Skeptical scenarios:

1. Validity checking:
   - check_abd_full_validity: Full observation, SAT check with Ab ↔ alpha
   - check_abd_partial_validity: Partial observation, EXISTS-completion SAT
   - check_abd_skeptical_validity: Partial observation, FORALL-completion validity

2. Cost optimization:
   - compute_abd_full_gold_opt_cost: Minimize |Ab| with Ab free
   - compute_abd_partial_gold_opt_cost: Minimize |Ab| over completions
   - compute_abd_partial_alpha_best_cost: Minimize |alpha(a)| over completions
   - compute_abd_skeptical_alpha_worst_cost: Maximize |alpha(a)| over valid completions

Semantics comparison:
- ABD_PARTIAL (existential): alpha is valid if SOME completion makes axioms hold
- ABD_SKEPTICAL (universal): alpha is valid if ALL completions make axioms hold

Reuses the shared grounding engine from fo_grounding_z3.py.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

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

from concept_synth.fo_grounding_z3 import (
    GroundingContext,
    build_known_atoms,
    build_unknown_atoms_set,
    build_unknown_vars_from_atoms,
    check_z3_available,
    ground_formula_to_z3,
)
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
from concept_synth.sexpr_parser import parse_sexpr_formula

# =============================================================================
# Result Data Structures
# =============================================================================


@dataclass
class AbdValidityResult:
    """Result of ABD validity check."""

    valid: bool
    reason: Optional[str] = None
    model: Optional[Any] = None  # Z3 model if SAT
    cost: Optional[int] = None  # Cost of alpha on this world
    per_element_alpha: Optional[Dict[str, bool]] = None  # alpha(a) for each a


@dataclass
class AbdOptResult:
    """Result of ABD optimization."""

    opt_cost: int
    model: Optional[Any] = None
    ab_assignment: Optional[Dict[str, bool]] = None  # Ab(a) for each a
    completion: Optional[Dict[str, bool]] = None  # For partial: unknown atom values


# =============================================================================
# Ground Atom Variable Creation
# =============================================================================


def create_predicate_vars(
    domain: List[str], world_id: str, include_ab: bool = True
) -> Dict[Tuple[str, Tuple[str, ...]], Any]:
    """
    Create Z3 boolean variables for all ground atoms.

    Args:
        domain: List of domain constants
        world_id: Unique identifier for variable naming
        include_ab: Whether to include Ab(a) variables

    Returns:
        Dictionary mapping (pred_name, args_tuple) to Z3 BoolRef
    """
    if not Z3_AVAILABLE:
        raise RuntimeError("Z3 is not installed")

    pred_vars = {}

    # Unary predicates
    for pred in ["P", "Q"]:
        for a in domain:
            key = (pred, (a,))
            pred_vars[key] = z3.Bool(f"{world_id}__{pred}__{a}")

    # Binary predicates
    for pred in ["R", "S"]:
        for a in domain:
            for b in domain:
                key = (pred, (a, b))
                pred_vars[key] = z3.Bool(f"{world_id}__{pred}__{a}__{b}")

    # Abnormality predicate
    if include_ab:
        for a in domain:
            key = ("Ab", (a,))
            pred_vars[key] = z3.Bool(f"{world_id}__Ab__{a}")

    return pred_vars


def create_unknown_vars(
    unknown_atoms: Dict[str, List[Any]], world_id: str
) -> Dict[Tuple[str, Tuple[str, ...]], Any]:
    """
    Create Z3 variables for unknown atoms only.

    Delegates to the shared build_unknown_vars_from_atoms function.

    Args:
        unknown_atoms: Dict mapping pred_name to list of unknown ground atoms
        world_id: Unique identifier for variable naming

    Returns:
        Dictionary mapping (pred_name, args_tuple) to Z3 BoolRef
    """
    return build_unknown_vars_from_atoms(unknown_atoms, world_id)


# =============================================================================
# Grounding Context Creation
# =============================================================================


def create_abd_grounding_context(
    world: Dict[str, Any],
    world_id: str,
    alpha_ast: Optional[FOFormula] = None,
    ab_vars: Optional[Dict[Tuple[str, Tuple[str, ...]], Any]] = None,
    unknown_vars: Optional[Dict[Tuple[str, Tuple[str, ...]], Any]] = None,
) -> GroundingContext:
    """
    Create a grounding context for ABD checking.

    Args:
        world: World dictionary with domain, predicates, unknownAtoms
        world_id: Unique identifier
        alpha_ast: If provided, Ab(a) is substituted with alpha(a)
        ab_vars: If provided, use these Z3 vars for Ab(a) (for optimization)
        unknown_vars: Z3 vars for unknown atoms (for partial observation)

    Returns:
        GroundingContext configured for ABD grounding
    """
    domain = world.get("domain", [])
    known_true, known_false = build_known_atoms(world)
    unknown_atoms_set = build_unknown_atoms_set(world)

    # Build pred_bindings for Ab
    pred_bindings = {}

    if alpha_ast is not None:
        # Inline substitution: Ab(a) := alpha(a)
        def ab_binding(args: Tuple[str, ...]) -> Any:
            # Ground alpha at this element
            env = {"x": args[0]}
            # Create a sub-context without Ab binding to avoid recursion
            sub_ctx = GroundingContext(
                domain=domain,
                known_true=known_true,
                known_false=known_false,
                unknown_atoms=unknown_atoms_set,
                unknown_vars=unknown_vars or {},
                pred_bindings={},  # No Ab binding in sub-context
                cache={},
            )
            return ground_formula_to_z3(alpha_ast, env, sub_ctx)

        pred_bindings["Ab"] = ab_binding

    elif ab_vars is not None:
        # Use provided Ab variables
        def ab_binding(args: Tuple[str, ...]) -> Any:
            key = ("Ab", args)
            if key in ab_vars:
                return ab_vars[key]
            else:
                return z3.BoolVal(False)  # Default to not abnormal

        pred_bindings["Ab"] = ab_binding

    return GroundingContext(
        domain=domain,
        known_true=known_true,
        known_false=known_false,
        unknown_atoms=unknown_atoms_set,
        unknown_vars=unknown_vars or {},
        pred_bindings=pred_bindings,
        cache={},
    )


# =============================================================================
# Constraint Building
# =============================================================================


def build_true_atoms_set(world: Dict[str, Any]) -> Set[Tuple[str, Tuple[str, ...]]]:
    """
    Build a set of all ground atoms that are TRUE in the world.

    Args:
        world: World dictionary with predicates

    Returns:
        Set of (pred_name, args_tuple) for all true atoms
    """
    true_atoms = set()
    predicates = world.get("predicates", {})

    # Unary predicates
    for pred in ["P", "Q"]:
        if pred in predicates:
            true_list = predicates[pred].get("true", [])
            for const in true_list:
                true_atoms.add((pred, (str(const),)))

    # Binary predicates
    for pred in ["R", "S"]:
        if pred in predicates:
            true_list = predicates[pred].get("true", [])
            for pair_str in true_list:
                pair_str = str(pair_str).strip("()")
                parts = [p.strip() for p in pair_str.split(",")]
                if len(parts) == 2:
                    true_atoms.add((pred, (parts[0], parts[1])))

    return true_atoms


def assert_closed_world_facts(
    ctx: GroundingContext, domain: List[str], predicates: List[str] = None
) -> None:
    """
    Enforce closed-world assumption: atoms not in known_true are FALSE.

    This modifies the context's known_true and known_false sets to ensure
    all ground atoms are explicitly assigned.

    Args:
        ctx: GroundingContext to modify
        domain: List of domain constants
        predicates: List of predicate names to process (default: P, Q, R, S)
    """
    if predicates is None:
        predicates = ["P", "Q", "R", "S"]

    # For each predicate, ensure all ground atoms are either true or false
    for pred in predicates:
        if pred in ["P", "Q"]:
            # Unary predicate
            for a in domain:
                key = (pred, (a,))
                if key not in ctx.known_true:
                    ctx.known_false.add(key)
        elif pred in ["R", "S"]:
            # Binary predicate
            for a in domain:
                for b in domain:
                    key = (pred, (a, b))
                    if key not in ctx.known_true:
                        ctx.known_false.add(key)


def create_closed_world_grounding_context(
    world: Dict[str, Any],
    world_id: str,
    alpha_ast: Optional[FOFormula] = None,
    ab_vars: Optional[Dict[Tuple[str, Tuple[str, ...]], Any]] = None,
) -> GroundingContext:
    """
    Create a grounding context with closed-world assumption enforced.

    For ABD-Full: all atoms not listed as TRUE are FALSE.

    Args:
        world: World dictionary with domain, predicates
        world_id: Unique identifier
        alpha_ast: If provided, Ab(a) is substituted with alpha(a)
        ab_vars: If provided, use these Z3 vars for Ab(a)

    Returns:
        GroundingContext with closed-world semantics
    """
    domain = world.get("domain", [])

    # Build true atoms from world
    true_atoms = build_true_atoms_set(world)

    # Initialize known_false as empty - we'll populate it
    known_false: Set[Tuple[str, Tuple[str, ...]]] = set()

    # Build pred_bindings for Ab
    pred_bindings = {}

    if alpha_ast is not None:

        def ab_binding(args: Tuple[str, ...]) -> Any:
            env = {"x": args[0]}
            sub_ctx = GroundingContext(
                domain=domain,
                known_true=true_atoms,
                known_false=known_false,
                unknown_atoms={},
                unknown_vars={},
                pred_bindings={},
                cache={},
            )
            # Apply closed-world to sub-context
            assert_closed_world_facts(sub_ctx, domain)
            return ground_formula_to_z3(alpha_ast, env, sub_ctx)

        pred_bindings["Ab"] = ab_binding

    elif ab_vars is not None:

        def ab_binding(args: Tuple[str, ...]) -> Any:
            key = ("Ab", args)
            if key in ab_vars:
                return ab_vars[key]
            else:
                return z3.BoolVal(False)

        pred_bindings["Ab"] = ab_binding

    ctx = GroundingContext(
        domain=domain,
        known_true=true_atoms,
        known_false=known_false,
        unknown_atoms={},
        unknown_vars={},
        pred_bindings=pred_bindings,
        cache={},
    )

    # Apply closed-world assumption
    assert_closed_world_facts(ctx, domain)

    return ctx


def assert_known_facts(
    solver: Any, world: Dict[str, Any], pred_vars: Dict[Tuple[str, Tuple[str, ...]], Any]
):
    """
    Assert known facts from the world as Z3 constraints.

    For full observation: all atoms are fixed.
    For partial observation: only known atoms are fixed.
    """
    predicates = world.get("predicates", {})

    # Unary predicates
    for pred in ["P", "Q"]:
        if pred in predicates:
            true_list = predicates[pred].get("true", [])
            false_list = predicates[pred].get("false", [])

            for const in true_list:
                key = (pred, (str(const),))
                if key in pred_vars:
                    solver.add(pred_vars[key] == z3.BoolVal(True))

            for const in false_list:
                key = (pred, (str(const),))
                if key in pred_vars:
                    solver.add(pred_vars[key] == z3.BoolVal(False))

    # Binary predicates
    for pred in ["R", "S"]:
        if pred in predicates:
            true_list = predicates[pred].get("true", [])
            false_list = predicates[pred].get("false", [])

            for pair_str in true_list:
                pair_str = str(pair_str).strip("()")
                parts = [p.strip() for p in pair_str.split(",")]
                if len(parts) == 2:
                    key = (pred, (parts[0], parts[1]))
                    if key in pred_vars:
                        solver.add(pred_vars[key] == z3.BoolVal(True))

            for pair_str in false_list:
                pair_str = str(pair_str).strip("()")
                parts = [p.strip() for p in pair_str.split(",")]
                if len(parts) == 2:
                    key = (pred, (parts[0], parts[1]))
                    if key in pred_vars:
                        solver.add(pred_vars[key] == z3.BoolVal(False))


def ground_axioms(axioms_ast: List[FOFormula], ctx: GroundingContext) -> List[Any]:
    """
    Ground all axioms to Z3 expressions.

    Args:
        axioms_ast: List of axiom formula ASTs
        ctx: Grounding context

    Returns:
        List of Z3 BoolRef expressions
    """
    grounded = []
    for axiom in axioms_ast:
        # Axioms are closed formulas (no free variables)
        expr = ground_formula_to_z3(axiom, {}, ctx)
        grounded.append(expr)
    return grounded


# =============================================================================
# ABD-Full Validity Check
# =============================================================================


def check_abd_full_validity(
    world: Dict[str, Any],
    axioms_ast: List[FOFormula],
    alpha_ast: FOFormula,
    world_id: Optional[str] = None,
    timeout_ms: int = 5000,
) -> AbdValidityResult:
    """
    Check validity of alpha for ABD-Full (full observation).

    Semantics: SAT of (Theta axioms + world facts + Ab ↔ alpha)
    where Ab(a) is substituted by alpha(a) in the axioms.

    Uses closed-world assumption: atoms not listed as TRUE are FALSE.

    Args:
        world: World dictionary with full observation
        axioms_ast: List of theory axiom ASTs
        alpha_ast: Alpha formula AST
        world_id: Optional world identifier
        timeout_ms: Z3 timeout

    Returns:
        AbdValidityResult with validity status and cost
    """
    if not Z3_AVAILABLE:
        return AbdValidityResult(valid=False, reason="Z3 not available")

    if world_id is None:
        world_id = world.get("worldId", "W")

    domain = world.get("domain", [])

    # Create grounding context with closed-world semantics and alpha substitution
    ctx = create_closed_world_grounding_context(world, world_id, alpha_ast=alpha_ast)

    # Ground axioms
    grounded_axioms = ground_axioms(axioms_ast, ctx)

    # Create solver and add constraints
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)

    for axiom_expr in grounded_axioms:
        solver.add(axiom_expr)

    # Check satisfiability
    result = solver.check()

    if result == z3.sat:
        # Compute cost: count elements where alpha(a) is true
        cost = 0
        per_element = {}
        for a in domain:
            env = {"x": a}
            alpha_expr = ground_formula_to_z3(alpha_ast, env, ctx)
            # Evaluate in the model
            model = solver.model()
            try:
                val = model.eval(alpha_expr, model_completion=True)
                is_true = z3.is_true(val)
                per_element[a] = is_true
                if is_true:
                    cost += 1
            except:
                per_element[a] = False

        return AbdValidityResult(
            valid=True, model=solver.model(), cost=cost, per_element_alpha=per_element
        )
    elif result == z3.unsat:
        return AbdValidityResult(valid=False, reason="UNSAT: axioms violated")
    else:
        return AbdValidityResult(valid=False, reason=f"Solver returned: {result}")


# =============================================================================
# ABD-Partial Validity Check (Exists-Completion)
# =============================================================================


def check_abd_partial_validity(
    world: Dict[str, Any],
    axioms_ast: List[FOFormula],
    alpha_ast: FOFormula,
    world_id: Optional[str] = None,
    timeout_ms: int = 5000,
) -> AbdValidityResult:
    """
    Check validity of alpha for ABD-Partial (partial observation).

    Semantics: EXISTS completion of unknown atoms such that
    SAT(Theta + known facts + completion + Ab ↔ alpha)

    Args:
        world: World dictionary with partial observation (unknownAtoms)
        axioms_ast: List of theory axiom ASTs
        alpha_ast: Alpha formula AST
        world_id: Optional world identifier
        timeout_ms: Z3 timeout

    Returns:
        AbdValidityResult with validity status
    """
    if not Z3_AVAILABLE:
        return AbdValidityResult(valid=False, reason="Z3 not available")

    if world_id is None:
        world_id = world.get("worldId", "W")

    domain = world.get("domain", [])
    unknown_atoms = world.get("unknownAtoms", {})

    # Create Z3 vars for unknown atoms
    unknown_vars = create_unknown_vars(unknown_atoms, world_id)

    # Create grounding context with alpha substitution and unknown vars
    ctx = create_abd_grounding_context(
        world, world_id, alpha_ast=alpha_ast, unknown_vars=unknown_vars
    )

    # Ground axioms
    grounded_axioms = ground_axioms(axioms_ast, ctx)

    # Create solver and add constraints
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)

    for axiom_expr in grounded_axioms:
        solver.add(axiom_expr)

    # Check satisfiability (existential over unknown atoms)
    result = solver.check()

    if result == z3.sat:
        model = solver.model()

        # Compute best-case cost (alpha(a) may depend on unknowns)
        # For simplicity, compute cost under the found model
        cost = 0
        per_element = {}
        for a in domain:
            env = {"x": a}
            alpha_expr = ground_formula_to_z3(alpha_ast, env, ctx)
            try:
                val = model.eval(alpha_expr, model_completion=True)
                is_true = z3.is_true(val)
                per_element[a] = is_true
                if is_true:
                    cost += 1
            except:
                per_element[a] = False

        return AbdValidityResult(valid=True, model=model, cost=cost, per_element_alpha=per_element)
    elif result == z3.unsat:
        return AbdValidityResult(valid=False, reason="UNSAT: no valid completion exists")
    else:
        return AbdValidityResult(valid=False, reason=f"Solver returned: {result}")


# =============================================================================
# ABD-Skeptical Validity Check (Universal/Forall-Completion)
# =============================================================================


def check_abd_skeptical_validity(
    world: Dict[str, Any],
    axioms_ast: List[FOFormula],
    alpha_ast: FOFormula,
    world_id: Optional[str] = None,
    timeout_ms: int = 5000,
) -> AbdValidityResult:
    """
    Check validity of alpha for ABD-Skeptical (universal completion).

    Semantics: FOR ALL completions of unknown atoms,
    SAT(Theta + known facts + completion + Ab ↔ alpha)

    Implementation: Check UNSAT(NOT(AX)) where AX is the grounded axioms.
    - If SAT(NOT(AX)): counterexample completion exists → alpha is INVALID
    - If UNSAT(NOT(AX)): no counterexample exists → alpha is VALID

    Args:
        world: World dictionary with partial observation (unknownAtoms)
        axioms_ast: List of theory axiom ASTs
        alpha_ast: Alpha formula AST
        world_id: Optional world identifier
        timeout_ms: Z3 timeout

    Returns:
        AbdValidityResult with validity status
    """
    if not Z3_AVAILABLE:
        return AbdValidityResult(valid=False, reason="Z3 not available")

    if world_id is None:
        world_id = world.get("worldId", "W")

    domain = world.get("domain", [])
    unknown_atoms = world.get("unknownAtoms", {})

    # Create Z3 vars for unknown atoms (free variables for completion search)
    unknown_vars = create_unknown_vars(unknown_atoms, world_id)

    # Create grounding context with alpha substitution and unknown vars
    ctx = create_abd_grounding_context(
        world, world_id, alpha_ast=alpha_ast, unknown_vars=unknown_vars
    )

    # Ground axioms
    grounded_axioms = ground_axioms(axioms_ast, ctx)

    # Handle empty axioms case - trivially valid (no constraints to violate)
    if not grounded_axioms:
        return AbdValidityResult(
            valid=True,
            reason="No axioms to check (trivially valid)",
            cost=0,
            per_element={},
        )

    # Create solver to find a counterexample completion
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)

    # Assert the NEGATION of (all axioms hold)
    # A counterexample is a completion where some axiom fails
    axioms_conjunction = z3.And(grounded_axioms) if len(grounded_axioms) > 1 else grounded_axioms[0]
    solver.add(z3.Not(axioms_conjunction))

    # Check satisfiability of the negated axioms
    result = solver.check()

    if result == z3.sat:
        # Found a counterexample completion where axioms fail
        # This means alpha is NOT valid under skeptical semantics
        model = solver.model()
        return AbdValidityResult(
            valid=False,
            reason="Counterexample completion found: axioms violated",
            model=model,
        )
    elif result == z3.unsat:
        # No counterexample exists - alpha is valid for ALL completions
        # Compute cost under an arbitrary valid completion
        # (For skeptical, the cost that matters is worst-case, computed separately)
        cost = 0
        per_element = {}
        for a in domain:
            env = {"x": a}
            alpha_expr = ground_formula_to_z3(alpha_ast, env, ctx)
            # Without a model, evaluate alpha structurally
            # For atoms that don't depend on unknowns, this gives the value
            # For atoms that do depend on unknowns, we can't determine cost here
            try:
                simplified = z3.simplify(alpha_expr)
                if z3.is_true(simplified):
                    per_element[a] = True
                    cost += 1
                elif z3.is_false(simplified):
                    per_element[a] = False
                else:
                    # Depends on unknowns - mark as unknown for now
                    per_element[a] = None
            except:
                per_element[a] = None

        return AbdValidityResult(
            valid=True,
            cost=cost,  # This is approximate; use worst_cost for accurate value
            per_element_alpha=per_element,
            reason="Valid: no counterexample completion exists",
        )
    else:
        return AbdValidityResult(valid=False, reason=f"Solver returned: {result}")


# =============================================================================
# ABD-Skeptical Worst-Case Cost
# =============================================================================


def compute_abd_skeptical_alpha_worst_cost(
    world: Dict[str, Any],
    axioms_ast: List[FOFormula],
    alpha_ast: FOFormula,
    world_id: Optional[str] = None,
    timeout_ms: int = 10000,
) -> AbdOptResult:
    """
    Compute the worst-case cost of alpha for ABD-Skeptical.

    This returns max |{x: alpha(x)}| over all completions that satisfy the axioms.
    Used for skeptical scoring: minimize worst-case abnormality count.

    Alpha is fixed; unknown atoms are free.
    Maximize sum alpha(a) subject to SAT(Theta + known + completion + Ab↔alpha).

    Args:
        world: World dictionary with partial observation
        axioms_ast: List of theory axiom ASTs
        alpha_ast: Alpha formula AST
        world_id: Optional world identifier
        timeout_ms: Z3 timeout

    Returns:
        AbdOptResult with worst (maximum) cost
    """
    if not Z3_AVAILABLE:
        return AbdOptResult(opt_cost=-1)

    if world_id is None:
        world_id = world.get("worldId", "W")

    domain = world.get("domain", [])
    unknown_atoms = world.get("unknownAtoms", {})

    # Create unknown atom variables
    unknown_vars = create_unknown_vars(unknown_atoms, world_id)

    # Create grounding context with alpha substitution and unknown vars
    ctx = create_abd_grounding_context(
        world, world_id, alpha_ast=alpha_ast, unknown_vars=unknown_vars
    )

    # Ground axioms
    grounded_axioms = ground_axioms(axioms_ast, ctx)

    # Ground alpha for each element to build cost expression
    alpha_exprs = []
    for a in domain:
        env = {"x": a}
        alpha_expr = ground_formula_to_z3(alpha_ast, env, ctx)
        alpha_exprs.append(alpha_expr)

    # Use Optimize solver
    opt = z3.Optimize()
    opt.set("timeout", timeout_ms)

    # Assert axioms must hold (only consider valid completions)
    for axiom_expr in grounded_axioms:
        opt.add(axiom_expr)

    # MAXIMIZE sum of alpha(a) - this gives worst-case cost
    alpha_sum = z3.Sum([z3.If(expr, 1, 0) for expr in alpha_exprs])
    opt.maximize(alpha_sum)

    result = opt.check()

    if result == z3.sat:
        model = opt.model()

        # Compute worst cost
        worst_cost = 0
        ab_assignment = {}
        for i, a in enumerate(domain):
            val = model.eval(alpha_exprs[i], model_completion=True)
            is_ab = z3.is_true(val)
            ab_assignment[a] = is_ab
            if is_ab:
                worst_cost += 1

        # Extract completion
        completion = {}
        for key, var in unknown_vars.items():
            val = model.eval(var, model_completion=True)
            completion[str(key)] = z3.is_true(val)

        return AbdOptResult(
            opt_cost=worst_cost, model=model, ab_assignment=ab_assignment, completion=completion
        )
    elif result == z3.unsat:
        # No valid completion exists (should not happen if validity was checked first)
        return AbdOptResult(opt_cost=-1)
    else:
        return AbdOptResult(opt_cost=-1)


def compute_abd_skeptical_alpha_best_cost(
    world: Dict[str, Any],
    axioms_ast: List[FOFormula],
    alpha_ast: FOFormula,
    world_id: Optional[str] = None,
    timeout_ms: int = 10000,
) -> AbdOptResult:
    """
    Compute the best-case cost of alpha for ABD-Skeptical.

    This returns min |{x: alpha(x)}| over all completions that satisfy the axioms.
    Useful for diagnostics and comparing with existential semantics.

    This is equivalent to compute_abd_partial_alpha_best_cost, provided here
    for API consistency.

    Args:
        world: World dictionary with partial observation
        axioms_ast: List of theory axiom ASTs
        alpha_ast: Alpha formula AST
        world_id: Optional world identifier
        timeout_ms: Z3 timeout

    Returns:
        AbdOptResult with best (minimum) cost
    """
    # Delegate to the partial version since the computation is identical
    return compute_abd_partial_alpha_best_cost(
        world, axioms_ast, alpha_ast, world_id, timeout_ms
    )


# =============================================================================
# ABD-Skeptical Multi-World Checking
# =============================================================================
# Quick Rejection Pre-filter for Cheater Testing (v0.6.8)
# =============================================================================


def generate_random_completion(
    unknown_atoms: Dict[str, List[str]],
    rng: Any,
) -> Dict[str, Set[str]]:
    """
    Generate a random completion of unknown atoms.

    Args:
        unknown_atoms: Dict mapping predicate name to list of unknown atom strings
        rng: Random number generator

    Returns:
        Dict mapping predicate name to set of atoms that are TRUE in this completion
    """
    completion = {}
    for pred, atoms in unknown_atoms.items():
        # Randomly decide which unknown atoms are true
        completion[pred] = set(a for a in atoms if rng.random() < 0.5)
    return completion


def evaluate_axioms_with_completion(
    world: Dict[str, Any],
    axioms_ast: List[FOFormula],
    alpha_ast: FOFormula,
    completion: Dict[str, Set[str]],
) -> bool:
    """
    Evaluate if axioms hold under a specific completion with Ab ↔ alpha.

    Args:
        world: World dict with predicates and unknownAtoms
        axioms_ast: List of axiom ASTs
        alpha_ast: Alpha formula AST (defines Ab)
        completion: Dict mapping predicate to set of unknown atoms that are TRUE

    Returns:
        True if all axioms hold, False otherwise
    """
    from concept_synth.fol.formulas import (
        Pred, Var, Constant, FONot, FOAnd, FOOr, FOImplies, Forall, Exists, Eq
    )

    domain = world.get("domain", [])
    predicates = world.get("predicates", {})
    unknown_atoms = world.get("unknownAtoms", {})

    # Build complete truth set: known true + completion of unknowns
    true_atoms: Set[Tuple[str, Tuple[str, ...]]] = set()

    for pred_name, pred_data in predicates.items():
        for atom in pred_data.get("true", []):
            if isinstance(atom, str):
                if atom.startswith("(") and atom.endswith(")"):
                    inner = atom[1:-1]
                    parts = tuple(p.strip() for p in inner.split(","))
                    true_atoms.add((pred_name, parts))
                else:
                    true_atoms.add((pred_name, (atom,)))

    # Add unknown atoms that are TRUE in this completion
    for pred_name, true_unknowns in completion.items():
        for atom in true_unknowns:
            if isinstance(atom, str):
                if atom.startswith("(") and atom.endswith(")"):
                    inner = atom[1:-1]
                    parts = tuple(p.strip() for p in inner.split(","))
                    true_atoms.add((pred_name, parts))
                else:
                    true_atoms.add((pred_name, (atom,)))

    # Pre-compute alpha(a) for each domain element
    alpha_values = {}
    for a in domain:
        alpha_values[a] = _eval_formula_fast(alpha_ast, {"x": a}, domain, true_atoms)

    def eval_term(term, env: Dict[str, str]) -> str:
        if isinstance(term, Constant):
            return term.name
        elif isinstance(term, Var):
            return env.get(term.name, term.name)
        else:
            raise ValueError(f"Unknown term type: {type(term)}")

    def eval_formula(f: FOFormula, env: Dict[str, str]) -> bool:
        if isinstance(f, Pred):
            args = tuple(eval_term(t, env) for t in f.args)
            if f.name == "Ab":
                # Ab(a) ↔ alpha(a)
                return alpha_values.get(args[0], False)
            return (f.name, args) in true_atoms

        elif isinstance(f, FONot):
            return not eval_formula(f.child, env)

        elif isinstance(f, FOAnd):
            return eval_formula(f.left, env) and eval_formula(f.right, env)

        elif isinstance(f, FOOr):
            return eval_formula(f.left, env) or eval_formula(f.right, env)

        elif isinstance(f, FOImplies):
            return (not eval_formula(f.left, env)) or eval_formula(f.right, env)

        elif isinstance(f, Forall):
            for d in domain:
                new_env = env.copy()
                new_env[f.var.name] = d
                if not eval_formula(f.body, new_env):
                    return False
            return True

        elif isinstance(f, Exists):
            for d in domain:
                new_env = env.copy()
                new_env[f.var.name] = d
                if eval_formula(f.body, new_env):
                    return True
            return False

        elif isinstance(f, Eq):
            left_val = eval_term(f.left, env)
            right_val = eval_term(f.right, env)
            return left_val == right_val

        else:
            raise ValueError(f"Unsupported formula type: {type(f)}")

    # Check all axioms
    for axiom in axioms_ast:
        if not eval_formula(axiom, {}):
            return False
    return True


def _eval_formula_fast(
    formula: FOFormula,
    env: Dict[str, str],
    domain: List[str],
    true_atoms: Set[Tuple[str, Tuple[str, ...]]],
) -> bool:
    """Fast formula evaluation helper (no Ab binding)."""
    from concept_synth.fol.formulas import (
        Pred, Var, Constant, FONot, FOAnd, FOOr, FOImplies, Forall, Exists, Eq
    )

    def eval_term(term) -> str:
        if isinstance(term, Constant):
            return term.name
        elif isinstance(term, Var):
            return env.get(term.name, term.name)
        else:
            raise ValueError(f"Unknown term type: {type(term)}")

    if isinstance(formula, Pred):
        args = tuple(eval_term(t) for t in formula.args)
        return (formula.name, args) in true_atoms

    elif isinstance(formula, FONot):
        return not _eval_formula_fast(formula.child, env, domain, true_atoms)

    elif isinstance(formula, FOAnd):
        return (_eval_formula_fast(formula.left, env, domain, true_atoms) and
                _eval_formula_fast(formula.right, env, domain, true_atoms))

    elif isinstance(formula, FOOr):
        return (_eval_formula_fast(formula.left, env, domain, true_atoms) or
                _eval_formula_fast(formula.right, env, domain, true_atoms))

    elif isinstance(formula, FOImplies):
        return (not _eval_formula_fast(formula.left, env, domain, true_atoms) or
                _eval_formula_fast(formula.right, env, domain, true_atoms))

    elif isinstance(formula, Forall):
        for d in domain:
            new_env = env.copy()
            new_env[formula.var.name] = d
            if not _eval_formula_fast(formula.body, new_env, domain, true_atoms):
                return False
        return True

    elif isinstance(formula, Exists):
        for d in domain:
            new_env = env.copy()
            new_env[formula.var.name] = d
            if _eval_formula_fast(formula.body, new_env, domain, true_atoms):
                return True
        return False

    elif isinstance(formula, Eq):
        return eval_term(formula.left) == eval_term(formula.right)

    else:
        raise ValueError(f"Unsupported formula type: {type(formula)}")


def quick_reject_cheater(
    worlds: List[Dict[str, Any]],
    axioms_ast: List[FOFormula],
    cheater_ast: FOFormula,
    num_samples: int = 10,
    rng: Any = None,
) -> bool:
    """
    Quick pre-filter to reject cheaters that fail on sampled completions.

    For skeptical validity, a cheater must work for ALL completions.
    If we find ONE completion where axioms don't hold with Ab ↔ cheater,
    we can skip the expensive Z3 check.

    Args:
        worlds: List of world dicts with unknownAtoms
        axioms_ast: List of axiom ASTs
        cheater_ast: Cheater formula AST
        num_samples: Number of random completions to try per world
        rng: Random number generator (uses random.Random() if None)

    Returns:
        True if cheater definitely fails (can skip Z3), False if need Z3 to confirm
    """
    import random
    if rng is None:
        rng = random.Random()

    for world in worlds:
        unknown_atoms = world.get("unknownAtoms", {})

        # Skip worlds with no unknowns (no sampling needed)
        total_unknowns = sum(len(v) for v in unknown_atoms.values())
        if total_unknowns == 0:
            continue

        for _ in range(num_samples):
            completion = generate_random_completion(unknown_atoms, rng)

            # If axioms don't hold under this completion with Ab ↔ cheater,
            # the cheater is not skeptically valid
            if not evaluate_axioms_with_completion(world, axioms_ast, cheater_ast, completion):
                return True  # Quick reject

    return False  # Need Z3 to confirm validity


def check_abd_skeptical_all_worlds(
    worlds: List[Dict[str, Any]],
    axioms_ast: List[FOFormula],
    alpha_ast: FOFormula,
    timeout_ms: int = 5000,
) -> Tuple[bool, List[AbdValidityResult], int]:
    """
    Check alpha skeptical validity on all worlds and compute total worst-case cost.

    For skeptical semantics, we need all worlds to be valid under universal completion,
    and the cost is the sum of worst-case costs across all worlds.

    Returns:
        Tuple of (all_valid, per_world_results, total_worst_cost)
    """
    results = []
    all_valid = True
    total_worst_cost = 0

    for i, world in enumerate(worlds):
        world_id = world.get("worldId", f"W{i}")
        result = check_abd_skeptical_validity(world, axioms_ast, alpha_ast, world_id, timeout_ms)
        results.append(result)

        if not result.valid:
            all_valid = False
        else:
            # Compute worst-case cost for valid alpha
            worst_result = compute_abd_skeptical_alpha_worst_cost(
                world, axioms_ast, alpha_ast, world_id, timeout_ms
            )
            if worst_result.opt_cost >= 0:
                total_worst_cost += worst_result.opt_cost

    return all_valid, results, total_worst_cost


# =============================================================================
# ABD-Full Gold Optimum Cost
# =============================================================================


def compute_abd_full_gold_opt_cost(
    world: Dict[str, Any],
    axioms_ast: List[FOFormula],
    world_id: Optional[str] = None,
    timeout_ms: int = 10000,
) -> AbdOptResult:
    """
    Compute the optimal (minimum) |Ab| for ABD-Full.

    Ab is a free decision variable per element.
    Minimize sum Ab(a) subject to SAT(Theta + facts).

    Uses closed-world assumption: atoms not listed as TRUE are FALSE.

    Args:
        world: World dictionary with full observation
        axioms_ast: List of theory axiom ASTs
        world_id: Optional world identifier
        timeout_ms: Z3 timeout

    Returns:
        AbdOptResult with optimal cost
    """
    if not Z3_AVAILABLE:
        return AbdOptResult(opt_cost=-1)

    if world_id is None:
        world_id = world.get("worldId", "W")

    domain = world.get("domain", [])

    # Create Ab variables
    ab_vars = {}
    for a in domain:
        key = ("Ab", (a,))
        ab_vars[key] = z3.Bool(f"{world_id}__Ab__{a}")

    # Create grounding context with closed-world semantics and Ab variables
    ctx = create_closed_world_grounding_context(world, world_id, ab_vars=ab_vars)

    # Ground axioms
    grounded_axioms = ground_axioms(axioms_ast, ctx)

    # Use Optimize solver
    opt = z3.Optimize()
    opt.set("timeout", timeout_ms)

    for axiom_expr in grounded_axioms:
        opt.add(axiom_expr)

    # Minimize sum of Ab(a)
    ab_sum = z3.Sum([z3.If(ab_vars[("Ab", (a,))], 1, 0) for a in domain])
    opt.minimize(ab_sum)

    result = opt.check()

    if result == z3.sat:
        model = opt.model()

        # Extract Ab assignment
        ab_assignment = {}
        opt_cost = 0
        for a in domain:
            key = ("Ab", (a,))
            val = model.eval(ab_vars[key], model_completion=True)
            is_ab = z3.is_true(val)
            ab_assignment[a] = is_ab
            if is_ab:
                opt_cost += 1

        return AbdOptResult(opt_cost=opt_cost, model=model, ab_assignment=ab_assignment)
    else:
        return AbdOptResult(opt_cost=-1)


# =============================================================================
# ABD-Partial Gold Optimum Cost
# =============================================================================


def compute_abd_partial_gold_opt_cost(
    world: Dict[str, Any],
    axioms_ast: List[FOFormula],
    world_id: Optional[str] = None,
    timeout_ms: int = 10000,
) -> AbdOptResult:
    """
    Compute the optimal (minimum) |Ab| for ABD-Partial.

    Unknown atoms and Ab are free decision variables.
    Minimize sum Ab(a) subject to SAT(Theta + known facts + completion).

    Args:
        world: World dictionary with partial observation
        axioms_ast: List of theory axiom ASTs
        world_id: Optional world identifier
        timeout_ms: Z3 timeout

    Returns:
        AbdOptResult with optimal cost
    """
    if not Z3_AVAILABLE:
        return AbdOptResult(opt_cost=-1)

    if world_id is None:
        world_id = world.get("worldId", "W")

    domain = world.get("domain", [])
    unknown_atoms = world.get("unknownAtoms", {})

    # Create Ab variables
    ab_vars = {}
    for a in domain:
        key = ("Ab", (a,))
        ab_vars[key] = z3.Bool(f"{world_id}__Ab__{a}")

    # Create unknown atom variables
    unknown_vars = create_unknown_vars(unknown_atoms, world_id)

    # Create grounding context with Ab variables and unknown vars
    ctx = create_abd_grounding_context(world, world_id, ab_vars=ab_vars, unknown_vars=unknown_vars)

    # Ground axioms
    grounded_axioms = ground_axioms(axioms_ast, ctx)

    # Use Optimize solver
    opt = z3.Optimize()
    opt.set("timeout", timeout_ms)

    for axiom_expr in grounded_axioms:
        opt.add(axiom_expr)

    # Minimize sum of Ab(a)
    ab_sum = z3.Sum([z3.If(ab_vars[("Ab", (a,))], 1, 0) for a in domain])
    opt.minimize(ab_sum)

    result = opt.check()

    if result == z3.sat:
        model = opt.model()

        # Extract Ab assignment
        ab_assignment = {}
        opt_cost = 0
        for a in domain:
            key = ("Ab", (a,))
            val = model.eval(ab_vars[key], model_completion=True)
            is_ab = z3.is_true(val)
            ab_assignment[a] = is_ab
            if is_ab:
                opt_cost += 1

        # Extract completion
        completion = {}
        for key, var in unknown_vars.items():
            val = model.eval(var, model_completion=True)
            completion[str(key)] = z3.is_true(val)

        return AbdOptResult(
            opt_cost=opt_cost, model=model, ab_assignment=ab_assignment, completion=completion
        )
    else:
        return AbdOptResult(opt_cost=-1)


# =============================================================================
# ABD-Partial Alpha Best Cost
# =============================================================================


def compute_abd_partial_alpha_best_cost(
    world: Dict[str, Any],
    axioms_ast: List[FOFormula],
    alpha_ast: FOFormula,
    world_id: Optional[str] = None,
    timeout_ms: int = 10000,
) -> AbdOptResult:
    """
    Compute the best-case cost of alpha for ABD-Partial.

    Alpha is fixed; unknown atoms are free.
    Minimize sum alpha(a) subject to SAT(Theta + known + completion + Ab↔alpha).

    Args:
        world: World dictionary with partial observation
        axioms_ast: List of theory axiom ASTs
        alpha_ast: Alpha formula AST
        world_id: Optional world identifier
        timeout_ms: Z3 timeout

    Returns:
        AbdOptResult with best cost
    """
    if not Z3_AVAILABLE:
        return AbdOptResult(opt_cost=-1)

    if world_id is None:
        world_id = world.get("worldId", "W")

    domain = world.get("domain", [])
    unknown_atoms = world.get("unknownAtoms", {})

    # Create unknown atom variables
    unknown_vars = create_unknown_vars(unknown_atoms, world_id)

    # Create grounding context with alpha substitution and unknown vars
    ctx = create_abd_grounding_context(
        world, world_id, alpha_ast=alpha_ast, unknown_vars=unknown_vars
    )

    # Ground axioms
    grounded_axioms = ground_axioms(axioms_ast, ctx)

    # Ground alpha for each element to build cost expression
    alpha_exprs = []
    for a in domain:
        env = {"x": a}
        alpha_expr = ground_formula_to_z3(alpha_ast, env, ctx)
        alpha_exprs.append(alpha_expr)

    # Use Optimize solver
    opt = z3.Optimize()
    opt.set("timeout", timeout_ms)

    for axiom_expr in grounded_axioms:
        opt.add(axiom_expr)

    # Minimize sum of alpha(a)
    alpha_sum = z3.Sum([z3.If(expr, 1, 0) for expr in alpha_exprs])
    opt.minimize(alpha_sum)

    result = opt.check()

    if result == z3.sat:
        model = opt.model()

        # Compute cost
        best_cost = 0
        ab_assignment = {}
        for i, a in enumerate(domain):
            val = model.eval(alpha_exprs[i], model_completion=True)
            is_ab = z3.is_true(val)
            ab_assignment[a] = is_ab
            if is_ab:
                best_cost += 1

        # Extract completion
        completion = {}
        for key, var in unknown_vars.items():
            val = model.eval(var, model_completion=True)
            completion[str(key)] = z3.is_true(val)

        return AbdOptResult(
            opt_cost=best_cost, model=model, ab_assignment=ab_assignment, completion=completion
        )
    elif result == z3.unsat:
        return AbdOptResult(opt_cost=-1)  # Invalid alpha
    else:
        return AbdOptResult(opt_cost=-1)


# =============================================================================
# Multi-World Checking
# =============================================================================


def check_abd_full_all_worlds(
    worlds: List[Dict[str, Any]],
    axioms_ast: List[FOFormula],
    alpha_ast: FOFormula,
    timeout_ms: int = 5000,
) -> Tuple[bool, List[AbdValidityResult], int]:
    """
    Check alpha validity on all worlds and compute total cost.

    Returns:
        Tuple of (all_valid, per_world_results, total_cost)
    """
    results = []
    all_valid = True
    total_cost = 0

    for i, world in enumerate(worlds):
        world_id = world.get("worldId", f"W{i}")
        result = check_abd_full_validity(world, axioms_ast, alpha_ast, world_id, timeout_ms)
        results.append(result)

        if not result.valid:
            all_valid = False
        elif result.cost is not None:
            total_cost += result.cost

    return all_valid, results, total_cost


def check_abd_partial_all_worlds(
    worlds: List[Dict[str, Any]],
    axioms_ast: List[FOFormula],
    alpha_ast: FOFormula,
    timeout_ms: int = 5000,
) -> Tuple[bool, List[AbdValidityResult], int]:
    """
    Check alpha validity on all worlds (partial observation) and compute total cost.

    Returns:
        Tuple of (all_valid, per_world_results, total_cost)
    """
    results = []
    all_valid = True
    total_cost = 0

    for i, world in enumerate(worlds):
        world_id = world.get("worldId", f"W{i}")
        result = check_abd_partial_validity(world, axioms_ast, alpha_ast, world_id, timeout_ms)
        results.append(result)

        if not result.valid:
            all_valid = False
        elif result.cost is not None:
            total_cost += result.cost

    return all_valid, results, total_cost


# =============================================================================
# Utility: Parse Axioms
# =============================================================================


def parse_axioms(axiom_strs: List[str]) -> List[FOFormula]:
    """
    Parse a list of axiom S-expr strings into ASTs.

    Args:
        axiom_strs: List of S-expression strings

    Returns:
        List of FOFormula ASTs
    """
    allowed_preds = {"P", "Q", "R", "S", "Ab"}
    return [parse_sexpr_formula(s, allowed_predicates=allowed_preds) for s in axiom_strs]


# =============================================================================
# Antecedent Extraction and Formula Evaluation (v0.6.8 - ABD_SKEPTICAL hardening)
# =============================================================================


def get_default_antecedent_ast(theory_axiom_ast: FOFormula) -> Optional[FOFormula]:
    """
    Extract the default antecedent AST from a theory axiom in canonical shape.

    Canonical shape: (forall x (implies (and <Ante(x)> (not (Ab x))) <Cons(x)>))

    Args:
        theory_axiom_ast: Parsed theory axiom

    Returns:
        The <Ante(x)> subtree, or None if axiom doesn't match canonical shape
    """
    # Import formula types
    from concept_synth.fol.formulas import Forall, FOImplies, FOAnd, FONot

    # Check it's a Forall
    if not isinstance(theory_axiom_ast, Forall):
        return None

    body = theory_axiom_ast.body

    # Check body is an Implies
    if not isinstance(body, FOImplies):
        return None

    antecedent = body.left
    # consequent = body.right  # We don't need this

    # Check antecedent is an And
    if not isinstance(antecedent, FOAnd):
        return None

    # Canonical form: (and <Ante> (not (Ab x)))
    # left = <Ante(x)>, right = (not (Ab x))
    # But we should also handle (and (not (Ab x)) <Ante>) just in case
    left = antecedent.left
    right = antecedent.right

    # Check which side has the (not (Ab x)) pattern
    if isinstance(right, FONot):
        # Standard: (and <Ante> (not (Ab x)))
        return left
    elif isinstance(left, FONot):
        # Reversed: (and (not (Ab x)) <Ante>)
        return right
    else:
        # Neither side is negation - unexpected shape
        return None


def evaluate_formula_at_element(
    formula: FOFormula,
    world: Dict[str, Any],
    element: str,
    var_name: str = "x",
    alpha_binding: Optional[Callable[[str], bool]] = None,
) -> bool:
    """
    Evaluate a formula with one free variable at a specific domain element.

    Uses closed-world assumption: atoms not in true set are false.

    Args:
        formula: FOFormula with free variable var_name
        world: World dict with domain, predicates
        element: Domain element to substitute for var_name
        var_name: Name of the free variable (default "x")
        alpha_binding: Optional function to evaluate Ab(a) -> bool

    Returns:
        True if formula is satisfied at element, False otherwise
    """
    from concept_synth.fol.formulas import (
        Pred, Var, Constant, FONot, FOAnd, FOOr, FOImplies, Forall, Exists, Eq
    )

    domain = world.get("domain", [])

    # Build true atoms set for fast lookup
    true_atoms: Set[Tuple[str, Tuple[str, ...]]] = set()
    predicates = world.get("predicates", {})
    for pred_name, pred_data in predicates.items():
        for atom in pred_data.get("true", []):
            if isinstance(atom, str):
                if atom.startswith("(") and atom.endswith(")"):
                    # Binary: "(a, b)"
                    inner = atom[1:-1]
                    parts = [p.strip() for p in inner.split(",")]
                    true_atoms.add((pred_name, tuple(parts)))
                else:
                    # Unary
                    true_atoms.add((pred_name, (atom,)))

    def eval_term(term, env: Dict[str, str]) -> str:
        """Evaluate a term to a constant given variable bindings."""
        if isinstance(term, Constant):
            return term.name
        elif isinstance(term, Var):
            if term.name in env:
                return env[term.name]
            raise ValueError(f"Unbound variable: {term.name}")
        else:
            raise ValueError(f"Unknown term type: {type(term)}")

    def eval_rec(f: FOFormula, env: Dict[str, str]) -> bool:
        """Recursively evaluate formula."""
        if isinstance(f, Pred):
            # Evaluate predicate
            args = tuple(eval_term(t, env) for t in f.args)

            # Special handling for Ab predicate
            if f.name == "Ab":
                if alpha_binding is not None:
                    return alpha_binding(args[0])
                # Default: check in true_atoms
                return (f.name, args) in true_atoms

            return (f.name, args) in true_atoms

        elif isinstance(f, FONot):
            return not eval_rec(f.child, env)

        elif isinstance(f, FOAnd):
            return eval_rec(f.left, env) and eval_rec(f.right, env)

        elif isinstance(f, FOOr):
            return eval_rec(f.left, env) or eval_rec(f.right, env)

        elif isinstance(f, FOImplies):
            return (not eval_rec(f.left, env)) or eval_rec(f.right, env)

        elif isinstance(f, Forall):
            # Evaluate body for all domain elements
            bound_var = f.var.name
            for d in domain:
                new_env = env.copy()
                new_env[bound_var] = d
                if not eval_rec(f.body, new_env):
                    return False
            return True

        elif isinstance(f, Exists):
            # Evaluate body for some domain element
            bound_var = f.var.name
            for d in domain:
                new_env = env.copy()
                new_env[bound_var] = d
                if eval_rec(f.body, new_env):
                    return True
            return False

        elif isinstance(f, Eq):
            # Equality: (= t1 t2)
            left_val = eval_term(f.left, env)
            right_val = eval_term(f.right, env)
            return left_val == right_val

        else:
            raise ValueError(f"Unsupported formula type: {type(f)}")

    # Start evaluation with initial binding
    return eval_rec(formula, {var_name: element})


def count_satisfying_x(
    world: Dict[str, Any],
    formula: FOFormula,
    alpha_binding: Optional[Callable[[str], bool]] = None,
) -> int:
    """
    Count domain elements satisfying a formula with free variable x.

    Args:
        world: World dict with domain and predicates
        formula: FOFormula with free variable "x"
        alpha_binding: Optional function to evaluate Ab(a) -> bool

    Returns:
        Number of domain elements where formula evaluates to True
    """
    domain = world.get("domain", [])
    count = 0
    for elem in domain:
        if evaluate_formula_at_element(formula, world, elem, "x", alpha_binding):
            count += 1
    return count


def get_alpha_satisfying_elements(
    world: Dict[str, Any],
    alpha_ast: FOFormula,
) -> Set[str]:
    """
    Get the set of domain elements where alpha(x) is true.

    Args:
        world: World dict with domain and predicates
        alpha_ast: Alpha formula with free variable "x"

    Returns:
        Set of domain elements satisfying alpha
    """
    domain = world.get("domain", [])
    result = set()
    for elem in domain:
        if evaluate_formula_at_element(alpha_ast, world, elem, "x"):
            result.add(elem)
    return result


def get_antecedent_satisfying_elements(
    world: Dict[str, Any],
    ante_ast: FOFormula,
) -> Set[str]:
    """
    Get the set of domain elements where antecedent(x) is true.

    Args:
        world: World dict with domain and predicates
        ante_ast: Antecedent formula with free variable "x"

    Returns:
        Set of domain elements satisfying antecedent
    """
    domain = world.get("domain", [])
    result = set()
    for elem in domain:
        if evaluate_formula_at_element(ante_ast, world, elem, "x"):
            result.add(elem)
    return result


# =============================================================================
# Bounded-k SAT for Fast optCost Computation (v0.6.3)
# =============================================================================


@dataclass
class BoundedSatResult:
    """Result of bounded-k SAT optCost computation."""

    sat: bool  # True if some k <= k_max is satisfiable
    opt_cost: int  # Optimal cost found, or -1 if UNSAT for all k
    model: Optional[Any] = None  # Z3 model if SAT
    ab_assignment: Optional[Dict[str, bool]] = None  # Ab(a) for each a


def min_ab_cost_bounded_sat_full(
    world: Dict[str, Any],
    axioms_ast: List[FOFormula],
    k_max: int = 4,
    world_id: Optional[str] = None,
    timeout_ms: int = 6000,
) -> BoundedSatResult:
    """
    Compute minimum |Ab| using bounded-k SAT checks for ABD-Full.

    Instead of using Z3 Optimize, we do incremental SAT checks:
    - For k = 0, 1, 2, ..., k_max:
        Check SAT(axioms ∧ facts ∧ (Sum Ab(a) == k))
    - Return the first k that is satisfiable.

    This is faster than Optimize when k is small (expected 1-3).

    Args:
        world: World dictionary with full observation
        axioms_ast: List of theory axiom ASTs
        k_max: Maximum k to check (reject if no solution found for k <= k_max)
        world_id: Optional world identifier
        timeout_ms: Z3 timeout per check

    Returns:
        BoundedSatResult with sat status and opt_cost
    """
    if not Z3_AVAILABLE:
        return BoundedSatResult(sat=False, opt_cost=-1)

    if world_id is None:
        world_id = world.get("worldId", "W")

    domain = world.get("domain", [])
    n = len(domain)

    # Create Ab variables
    ab_vars = {}
    for a in domain:
        key = ("Ab", (a,))
        ab_vars[key] = z3.Bool(f"{world_id}__Ab__{a}")

    # Create grounding context with closed-world semantics and Ab variables
    ctx = create_closed_world_grounding_context(world, world_id, ab_vars=ab_vars)

    # Ground axioms once
    grounded_axioms = ground_axioms(axioms_ast, ctx)

    # Create base solver with axioms
    base_solver = z3.Solver()
    base_solver.set("timeout", timeout_ms)
    for axiom_expr in grounded_axioms:
        base_solver.add(axiom_expr)

    # Build Ab count expression
    ab_list = [ab_vars[("Ab", (a,))] for a in domain]

    # Try k = 0, 1, ..., k_max
    for k in range(min(k_max + 1, n + 1)):
        base_solver.push()

        # Add cardinality constraint: exactly k Ab(a) are true
        # Use PbEq for pseudo-boolean equality
        pb_constraint = z3.PbEq([(v, 1) for v in ab_list], k)
        base_solver.add(pb_constraint)

        result = base_solver.check()

        if result == z3.sat:
            model = base_solver.model()

            # Extract Ab assignment
            ab_assignment = {}
            for a in domain:
                key = ("Ab", (a,))
                val = model.eval(ab_vars[key], model_completion=True)
                ab_assignment[a] = z3.is_true(val)

            base_solver.pop()
            return BoundedSatResult(sat=True, opt_cost=k, model=model, ab_assignment=ab_assignment)

        base_solver.pop()

    # No solution found for k <= k_max
    return BoundedSatResult(sat=False, opt_cost=-1)


def min_ab_cost_bounded_sat_partial(
    world: Dict[str, Any],
    axioms_ast: List[FOFormula],
    k_max: int = 4,
    world_id: Optional[str] = None,
    timeout_ms: int = 6000,
) -> BoundedSatResult:
    """
    Compute minimum |Ab| using bounded-k SAT checks for ABD-Partial.

    For partial observation, we have unknown atoms that can be completed
    existentially. We check:
    - For k = 0, 1, 2, ..., k_max:
        Check SAT(axioms ∧ known_facts ∧ (Sum Ab(a) == k))
        where unknown atoms are free variables.

    Args:
        world: World dictionary with partial observation (unknownAtoms)
        axioms_ast: List of theory axiom ASTs
        k_max: Maximum k to check
        world_id: Optional world identifier
        timeout_ms: Z3 timeout per check

    Returns:
        BoundedSatResult with sat status and opt_cost
    """
    if not Z3_AVAILABLE:
        return BoundedSatResult(sat=False, opt_cost=-1)

    if world_id is None:
        world_id = world.get("worldId", "W")

    domain = world.get("domain", [])
    unknown_atoms = world.get("unknownAtoms", {})
    n = len(domain)

    # Create Ab variables
    ab_vars = {}
    for a in domain:
        key = ("Ab", (a,))
        ab_vars[key] = z3.Bool(f"{world_id}__Ab__{a}")

    # Create unknown vars
    unknown_vars = create_unknown_vars(unknown_atoms, world_id)

    # Create grounding context with Ab variables and unknown vars
    ctx = create_abd_grounding_context(world, world_id, ab_vars=ab_vars, unknown_vars=unknown_vars)

    # Ground axioms once
    grounded_axioms = ground_axioms(axioms_ast, ctx)

    # Create base solver with axioms
    base_solver = z3.Solver()
    base_solver.set("timeout", timeout_ms)
    for axiom_expr in grounded_axioms:
        base_solver.add(axiom_expr)

    # Build Ab count expression
    ab_list = [ab_vars[("Ab", (a,))] for a in domain]

    # Try k = 0, 1, ..., k_max
    for k in range(min(k_max + 1, n + 1)):
        base_solver.push()

        # Add cardinality constraint: exactly k Ab(a) are true
        pb_constraint = z3.PbEq([(v, 1) for v in ab_list], k)
        base_solver.add(pb_constraint)

        result = base_solver.check()

        if result == z3.sat:
            model = base_solver.model()

            # Extract Ab assignment
            ab_assignment = {}
            for a in domain:
                key = ("Ab", (a,))
                val = model.eval(ab_vars[key], model_completion=True)
                ab_assignment[a] = z3.is_true(val)

            # Extract completion for unknown atoms
            completion = {}
            for key, var in unknown_vars.items():
                val = model.eval(var, model_completion=True)
                completion[key] = z3.is_true(val)

            base_solver.pop()
            return BoundedSatResult(sat=True, opt_cost=k, model=model, ab_assignment=ab_assignment)

        base_solver.pop()

    # No solution found for k <= k_max
    return BoundedSatResult(sat=False, opt_cost=-1)


def min_ab_cost_bounded_sat_alpha_partial(
    world: Dict[str, Any],
    axioms_ast: List[FOFormula],
    alpha_ast: FOFormula,
    k_max: int = 4,
    world_id: Optional[str] = None,
    timeout_ms: int = 6000,
) -> BoundedSatResult:
    """
    Compute minimum alpha cost using bounded-k SAT for ABD-Partial.

    This computes the minimum number of elements where alpha(a) is true
    over all valid completions of unknown atoms.

    Args:
        world: World dictionary with partial observation
        axioms_ast: List of theory axiom ASTs
        alpha_ast: Alpha formula AST (Ab ↔ alpha substitution)
        k_max: Maximum k to check
        world_id: Optional world identifier
        timeout_ms: Z3 timeout per check

    Returns:
        BoundedSatResult with sat status and opt_cost (= min |{a : alpha(a)}|)
    """
    if not Z3_AVAILABLE:
        return BoundedSatResult(sat=False, opt_cost=-1)

    if world_id is None:
        world_id = world.get("worldId", "W")

    domain = world.get("domain", [])
    unknown_atoms = world.get("unknownAtoms", {})
    n = len(domain)

    # Create unknown vars
    unknown_vars = create_unknown_vars(unknown_atoms, world_id)

    # Create grounding context with alpha substitution and unknown vars
    ctx = create_abd_grounding_context(
        world, world_id, alpha_ast=alpha_ast, unknown_vars=unknown_vars
    )

    # Ground axioms once
    grounded_axioms = ground_axioms(axioms_ast, ctx)

    # Create alpha indicator variables for each element
    alpha_indicators = []
    for a in domain:
        env = {"x": a}
        alpha_expr = ground_formula_to_z3(alpha_ast, env, ctx)
        alpha_indicators.append(alpha_expr)

    # Create base solver with axioms
    base_solver = z3.Solver()
    base_solver.set("timeout", timeout_ms)
    for axiom_expr in grounded_axioms:
        base_solver.add(axiom_expr)

    # Try k = 0, 1, ..., k_max
    for k in range(min(k_max + 1, n + 1)):
        base_solver.push()

        # Add cardinality constraint: exactly k alpha(a) are true
        # Convert to pseudo-boolean: Sum If(alpha_i, 1, 0) == k
        pb_terms = [(z3.If(ai, 1, 0), 1) for ai in alpha_indicators]
        # Use AtMost + AtLeast for exact count
        base_solver.add(z3.PbLe([(ai, 1) for ai in alpha_indicators], k))
        base_solver.add(z3.PbGe([(ai, 1) for ai in alpha_indicators], k))

        result = base_solver.check()

        if result == z3.sat:
            model = base_solver.model()

            # Extract alpha assignment
            ab_assignment = {}
            for i, a in enumerate(domain):
                val = model.eval(alpha_indicators[i], model_completion=True)
                ab_assignment[a] = z3.is_true(val)

            base_solver.pop()
            return BoundedSatResult(sat=True, opt_cost=k, model=model, ab_assignment=ab_assignment)

        base_solver.pop()

    # No solution found for k <= k_max (min_ab_alpha_cost_bounded_sat_skeptical)
    return BoundedSatResult(sat=False, opt_cost=-1)


# =============================================================================
# ABD-Skeptical Optimizations v0.6.10
# =============================================================================


@dataclass
class SkepticalCheckResult:
    """Result of optimized skeptical validity check."""
    valid: bool
    reason: Optional[str] = None
    counterexample_completion: Optional[Dict[str, bool]] = None


def check_abd_skeptical_validity_fast(
    world: Dict[str, Any],
    axioms_ast: List[FOFormula],
    alpha_ast: FOFormula,
    world_id: Optional[str] = None,
    timeout_ms: int = 2000,
) -> SkepticalCheckResult:
    """
    Fast skeptical validity check via counterexample SAT.

    Semantics: alpha is skeptically valid iff NO completion violates the axioms.

    Implementation: Single SAT check for counterexample:
        UNSAT(known_facts ∧ Ab↔alpha ∧ ¬(grounded axioms))

    If SAT: found counterexample completion → alpha INVALID
    If UNSAT: no counterexample exists → alpha VALID

    Args:
        world: World dict with partial observation (unknownAtoms)
        axioms_ast: List of theory axiom ASTs
        alpha_ast: Alpha formula AST
        world_id: Optional world identifier
        timeout_ms: Z3 timeout

    Returns:
        SkepticalCheckResult with validity status
    """
    if not Z3_AVAILABLE:
        return SkepticalCheckResult(valid=False, reason="Z3 not available")

    if world_id is None:
        world_id = world.get("worldId", "W")

    unknown_atoms = world.get("unknownAtoms", {})

    # Create Z3 vars for unknown atoms (free variables for completion search)
    unknown_vars = create_unknown_vars(unknown_atoms, world_id)

    # Create grounding context with alpha substitution and unknown vars
    ctx = create_abd_grounding_context(
        world, world_id, alpha_ast=alpha_ast, unknown_vars=unknown_vars
    )

    # Ground axioms
    grounded_axioms = ground_axioms(axioms_ast, ctx)

    # Create solver to find a counterexample completion
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)

    # Assert the NEGATION of (all axioms hold)
    # A counterexample is a completion where some axiom fails
    if len(grounded_axioms) > 1:
        axioms_conjunction = z3.And(grounded_axioms)
    else:
        axioms_conjunction = grounded_axioms[0]
    solver.add(z3.Not(axioms_conjunction))

    # Check satisfiability of the negated axioms
    result = solver.check()

    if result == z3.sat:
        # Found a counterexample completion where axioms fail
        model = solver.model()
        # Extract counterexample completion
        completion = {}
        for key, var in unknown_vars.items():
            val = model.eval(var, model_completion=True)
            completion[str(key)] = z3.is_true(val)
        return SkepticalCheckResult(
            valid=False,
            reason="Counterexample completion found",
            counterexample_completion=completion,
        )
    elif result == z3.unsat:
        # No counterexample exists - alpha is valid for ALL completions
        return SkepticalCheckResult(valid=True, reason="No counterexample exists")
    else:
        return SkepticalCheckResult(valid=False, reason=f"Solver returned: {result}")


def compute_opt_cost_by_violation_count(
    world: Dict[str, Any],
    axioms_ast: List[FOFormula],
    theory_id: str,
) -> int:
    """
    Compute optimal cost for single-default theories by counting violations.

    For axiom shape: forall x ((Ante(x) & not Ab(x)) -> Cons(x))
    With Ab empty, violations are: Viol(x) := Ante(x) & not Cons(x)
    optCost = |{x : Viol(x)}| (minimum exceptions needed)

    This is O(n) fast evaluation, no Z3 needed.

    Args:
        world: Full observation world dict
        axioms_ast: List of theory axiom ASTs (expects single default axiom)
        theory_id: Theory ID for context

    Returns:
        Optimal cost (number of violations), or -1 if theory doesn't match shape
    """
    if not axioms_ast:
        return -1

    # Extract antecedent and consequent from the first axiom
    axiom = axioms_ast[0]

    # Try to extract Ante and Cons from canonical shape
    ante_ast = get_default_antecedent_ast(axiom)
    cons_ast = get_default_consequent_ast(axiom)

    if ante_ast is None or cons_ast is None:
        return -1  # Theory doesn't match canonical shape

    domain = world.get("domain", [])

    # Count violations: Ante(x) is true AND Cons(x) is false
    violation_count = 0
    for elem in domain:
        ante_val = evaluate_formula_at_element(ante_ast, world, elem, "x")
        if ante_val:
            cons_val = evaluate_formula_at_element(cons_ast, world, elem, "x")
            if not cons_val:
                violation_count += 1

    return violation_count


def get_default_consequent_ast(theory_axiom_ast: FOFormula) -> Optional[FOFormula]:
    """
    Extract the default consequent AST from a theory axiom in canonical shape.

    Canonical shape: (forall x (implies (and <Ante(x)> (not (Ab x))) <Cons(x)>))

    Args:
        theory_axiom_ast: Parsed theory axiom

    Returns:
        The <Cons(x)> subtree, or None if axiom doesn't match canonical shape
    """
    from concept_synth.fol.formulas import Forall, FOImplies

    if not isinstance(theory_axiom_ast, Forall):
        return None

    body = theory_axiom_ast.body

    if not isinstance(body, FOImplies):
        return None

    # Consequent is the right side of the implication
    return body.right


def compute_alpha_cost_fast(
    world: Dict[str, Any],
    alpha_ast: FOFormula,
) -> int:
    """
    Compute alpha cost (number of elements where alpha is true) via fast evaluation.

    No Z3 needed - pure Python evaluation using closed-world assumption.

    Args:
        world: World dict (full or completed partial)
        alpha_ast: Alpha formula AST

    Returns:
        Count of domain elements where alpha(x) is true
    """
    domain = world.get("domain", [])
    cost = 0
    for elem in domain:
        if evaluate_formula_at_element(alpha_ast, world, elem, "x"):
            cost += 1
    return cost


def compute_extreme_completion_costs_fast(
    world: Dict[str, Any],
    alpha_ast: FOFormula,
) -> Dict[str, Any]:
    """
    Compute alpha costs under extreme completions (all-false, all-true) via fast eval.

    No Z3 needed - creates completed worlds and evaluates directly.

    Args:
        world: Partial observation world dict
        alpha_ast: Alpha formula AST

    Returns:
        Dict with goldCost_allFalse, goldCost_allTrue, goldCostRobust, robustSpread
    """
    unknown_atoms = world.get("unknownAtoms", {})

    # Build all-false completion world
    world_all_false = world.copy()
    world_all_false["domain"] = list(world.get("domain", []))
    predicates_false = {}
    for pred in ["P", "Q", "R", "S"]:
        known_true = list(world["predicates"].get(pred, {}).get("true", []))
        # Unknown atoms become false (just not in true list)
        predicates_false[pred] = {"true": known_true, "false": []}
    world_all_false["predicates"] = predicates_false
    world_all_false.pop("unknownAtoms", None)

    # Build all-true completion world
    world_all_true = world.copy()
    world_all_true["domain"] = list(world.get("domain", []))
    predicates_true = {}
    for pred in ["P", "Q", "R", "S"]:
        known_true = list(world["predicates"].get(pred, {}).get("true", []))
        unknowns = unknown_atoms.get(pred, [])
        # Unknown atoms become true
        predicates_true[pred] = {"true": known_true + unknowns, "false": []}
    world_all_true["predicates"] = predicates_true
    world_all_true.pop("unknownAtoms", None)

    # Compute costs using fast evaluation
    cost_false = compute_alpha_cost_fast(world_all_false, alpha_ast)
    cost_true = compute_alpha_cost_fast(world_all_true, alpha_ast)

    return {
        "goldCost_allFalse": cost_false,
        "goldCost_allTrue": cost_true,
        "goldCostRobust": max(cost_false, cost_true),
        "robustSpread": abs(cost_true - cost_false),
    }


def compute_violation_extreme(
    world: Dict[str, Any],
    axioms_ast: List[FOFormula],
    alpha_ast: FOFormula,
    theory_id: str,
) -> Dict[str, Any]:
    """
    Compute worst-case violation counts under extreme completions (allFalse, allTrue).

    This is a structural lower bound on robust Ab size for default theories.
    Uses fast O(n) violation counting (no Z3).

    Args:
        world: Partial observation world dict
        axioms_ast: List of theory axiom ASTs
        alpha_ast: Alpha formula AST
        theory_id: Theory ID for violation counting

    Returns:
        Dict with viol_false, viol_true, viol_extreme, and the completed worlds
    """
    unknown_atoms = world.get("unknownAtoms", {})
    domain = world.get("domain", [])
    predicates = world.get("predicates", {})

    # Build all-false completion world
    world_all_false = world.copy()
    world_all_false["domain"] = list(domain)
    predicates_false = {}
    for pred in ["P", "Q", "R", "S"]:
        known_true = list(predicates.get(pred, {}).get("true", []))
        predicates_false[pred] = {"true": known_true, "false": []}
    world_all_false["predicates"] = predicates_false
    world_all_false.pop("unknownAtoms", None)

    # Build all-true completion world
    world_all_true = world.copy()
    world_all_true["domain"] = list(domain)
    predicates_true = {}
    for pred in ["P", "Q", "R", "S"]:
        known_true = list(predicates.get(pred, {}).get("true", []))
        unknowns = unknown_atoms.get(pred, [])
        predicates_true[pred] = {"true": known_true + unknowns, "false": []}
    world_all_true["predicates"] = predicates_true
    world_all_true.pop("unknownAtoms", None)

    # Compute violation counts using fast counting
    viol_false = compute_opt_cost_by_violation_count(world_all_false, axioms_ast, theory_id)
    viol_true = compute_opt_cost_by_violation_count(world_all_true, axioms_ast, theory_id)

    # Handle -1 return (theory doesn't match canonical shape)
    if viol_false < 0:
        viol_false = 0
    if viol_true < 0:
        viol_true = 0

    return {
        "viol_false": viol_false,
        "viol_true": viol_true,
        "viol_extreme": max(viol_false, viol_true),
        "world_all_false": world_all_false,
        "world_all_true": world_all_true,
    }


def exists_completion_with_cost_at_least_k(
    world: Dict[str, Any],
    axioms_ast: List[FOFormula],
    alpha_ast: FOFormula,
    k: int,
    world_id: Optional[str] = None,
    timeout_ms: int = 2000,
) -> bool:
    """
    Check if there exists a completion with alpha cost >= k (SAT threshold check).

    Used to enforce goldCostRobust <= 4 without Optimize:
    - Call with k=5
    - If SAT: worst-case cost >= 5, reject
    - If UNSAT: worst-case cost <= 4, accept

    Encoding:
    - Unknown atoms are free Bool vars
    - Assert axioms must hold (only valid completions)
    - Assert Sum_x If(alpha(x), 1, 0) >= k
    - SAT? Then exists completion with cost >= k

    Args:
        world: Partial observation world dict
        axioms_ast: List of theory axiom ASTs
        alpha_ast: Alpha formula AST
        k: Cost threshold
        world_id: Optional world identifier
        timeout_ms: Z3 timeout

    Returns:
        True if SAT (exists completion with cost >= k), False otherwise
    """
    if not Z3_AVAILABLE:
        return True  # Conservative: assume bad

    if world_id is None:
        world_id = world.get("worldId", "W")

    domain = world.get("domain", [])
    unknown_atoms = world.get("unknownAtoms", {})

    # Create unknown vars
    unknown_vars = create_unknown_vars(unknown_atoms, world_id)

    # Create grounding context with alpha substitution
    ctx = create_abd_grounding_context(
        world, world_id, alpha_ast=alpha_ast, unknown_vars=unknown_vars
    )

    # Ground axioms
    grounded_axioms = ground_axioms(axioms_ast, ctx)

    # Ground alpha for each element
    alpha_exprs = []
    for a in domain:
        env = {"x": a}
        alpha_expr = ground_formula_to_z3(alpha_ast, env, ctx)
        alpha_exprs.append(alpha_expr)

    # Create solver
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)

    # Assert axioms must hold (valid completions only)
    for axiom_expr in grounded_axioms:
        solver.add(axiom_expr)

    # Assert cost >= k using PbGe (pseudo-boolean)
    solver.add(z3.PbGe([(expr, 1) for expr in alpha_exprs], k))

    result = solver.check()
    return result == z3.sat


def build_unknown_set_incremental(
    full_world: Dict[str, Any],
    theory_id: str,
    axioms_ast: List[FOFormula],
    alpha_ast: FOFormula,
    alpha_preds: Set[str],
    target_unknown_min: int = 8,
    target_unknown_max: int = 20,
    min_unknown_in_alpha_preds: int = 4,
    rng: Any = None,
    timeout_ms: int = 1000,
    # v2 fast preset options
    candidate_pool_limit: Optional[int] = None,
    prefer_alpha_preds: bool = True,
    extreme_cost_cap: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Build unknown set incrementally, preserving skeptical validity.

    Algorithm:
    1. Start with empty unknown set U = {}
    2. Build candidate pool from R/S atoms (biased toward alpha predicates)
    3. For each candidate:
       - Tentatively add to U
       - Run counterexample SAT check
       - If SAT (counterexample exists): undo, mark unsafe
       - If UNSAT: keep
    4. Return masked world if |U| >= target_min, else None

    Args:
        full_world: Full observation world dict
        theory_id: Theory ID
        axioms_ast: List of theory axiom ASTs
        alpha_ast: Alpha formula AST
        alpha_preds: Set of predicates used by alpha
        target_unknown_min: Minimum unknowns required
        target_unknown_max: Maximum unknowns allowed
        min_unknown_in_alpha_preds: Min unknowns in alpha's predicates
        rng: Random number generator
        timeout_ms: Z3 timeout for counterexample checks
        candidate_pool_limit: Optional max candidates to consider (for fast preset)
        prefer_alpha_preds: If True, alpha preds first; if False, others first
        extreme_cost_cap: Optional max extreme cost; reject candidate if exceeded

    Returns:
        Masked world dict, or None if couldn't build valid unknown set
    """
    import random
    if rng is None:
        rng = random.Random()

    if not Z3_AVAILABLE:
        return None

    domain = full_world.get("domain", [])
    predicates = full_world.get("predicates", {})
    world_id = full_world.get("worldId", "W_inc")

    # Build candidate pool: all R/S ground atoms
    candidates = []
    for pred in ["R", "S"]:
        true_list = predicates.get(pred, {}).get("true", [])
        true_set = set(str(p) for p in true_list)
        # Include both true and false atoms as candidates
        for pair in true_list:
            candidates.append((pred, str(pair), True))  # (pred, atom, was_true)
        # Also include atoms not in true list (i.e., false atoms)
        for a in domain:
            for b in domain:
                pair_str = f"({a}, {b})"
                if pair_str not in true_set:
                    candidates.append((pred, pair_str, False))

    # Separate by alpha predicates
    alpha_pred_candidates = [c for c in candidates if c[0] in alpha_preds]
    other_candidates = [c for c in candidates if c[0] not in alpha_preds]

    # v2: Theory-aware ordering for TH2/TH3 yield improvement
    # TH2: relational antecedent ∃y R(x,y) ∧ P(y) - masking false R can explode allTrue violations
    #      Prefer masking true R atoms before false R atoms
    # TH3: existential consequent ∃y R(x,y) - masking true R removes witnesses in allFalse
    #      Prefer masking false R atoms before true R atoms
    if theory_id == "TH2":
        # For TH2: sort so true atoms come before false atoms (prefer true)
        alpha_pred_candidates.sort(key=lambda c: (0 if c[2] else 1, c[0], c[1]))
        other_candidates.sort(key=lambda c: (0 if c[2] else 1, c[0], c[1]))
    elif theory_id == "TH3":
        # For TH3: sort so false atoms come before true atoms (prefer false)
        alpha_pred_candidates.sort(key=lambda c: (1 if c[2] else 0, c[0], c[1]))
        other_candidates.sort(key=lambda c: (1 if c[2] else 0, c[0], c[1]))
    else:
        # Default: shuffle within groups
        rng.shuffle(alpha_pred_candidates)
        rng.shuffle(other_candidates)

    # Order: alpha preds first (original) or others first (fast preset)
    if prefer_alpha_preds:
        candidates = alpha_pred_candidates + other_candidates
    else:
        candidates = other_candidates + alpha_pred_candidates

    # v2: Limit candidate pool size for fast preset
    if candidate_pool_limit is not None and len(candidates) > candidate_pool_limit:
        rng.shuffle(candidates)
        candidates = candidates[:candidate_pool_limit]

    # Track unknown set
    unknown_set: Dict[str, List[str]] = {"R": [], "S": []}
    unsafe_atoms: Set[Tuple[str, str]] = set()

    def build_masked_world() -> Dict[str, Any]:
        """Build a masked world from current unknown_set."""
        masked = full_world.copy()
        masked["domain"] = list(domain)
        new_predicates = {}
        for pred in ["P", "Q", "R", "S"]:
            true_list = list(predicates.get(pred, {}).get("true", []))
            unknown_list = unknown_set.get(pred, [])
            # Remove unknowns from true list
            unknown_set_strs = set(str(u) for u in unknown_list)
            new_true = [a for a in true_list if str(a) not in unknown_set_strs]
            new_predicates[pred] = {"true": new_true}
        masked["predicates"] = new_predicates
        masked["unknownAtoms"] = {k: list(v) for k, v in unknown_set.items()}
        masked["totalUnknownAtoms"] = sum(len(v) for v in unknown_set.values())
        return masked

    def check_skeptical_valid(masked_world: Dict[str, Any]) -> bool:
        """Quick skeptical validity check."""
        result = check_abd_skeptical_validity_fast(
            masked_world, axioms_ast, alpha_ast, world_id, timeout_ms
        )
        return result.valid

    # Incrementally add unknowns
    for pred, atom, was_true in candidates:
        if sum(len(v) for v in unknown_set.values()) >= target_unknown_max:
            break

        # Skip if already marked unsafe
        if (pred, atom) in unsafe_atoms:
            continue

        # Tentatively add
        unknown_set[pred].append(atom)

        # Build masked world and check
        masked = build_masked_world()

        # v2: Check extreme cost cap BEFORE expensive SAT call (if enabled)
        if extreme_cost_cap is not None:
            extreme_costs = compute_extreme_completion_costs_fast(masked, alpha_ast)
            if extreme_costs["goldCostRobust"] > extreme_cost_cap:
                # Undo - this unknown causes extreme cost to exceed cap
                unknown_set[pred].pop()
                unsafe_atoms.add((pred, atom))
                continue

        if check_skeptical_valid(masked):
            # Keep this unknown
            pass
        else:
            # Undo - this unknown breaks skeptical validity
            unknown_set[pred].pop()
            unsafe_atoms.add((pred, atom))

    # Check if we reached minimum
    total_unknowns = sum(len(v) for v in unknown_set.values())
    if total_unknowns < target_unknown_min:
        return None

    # Check unknown-alpha interaction
    unknown_in_alpha = sum(
        len(unknown_set.get(pred, []))
        for pred in alpha_preds
        if pred not in ("Ab",)
    )
    if unknown_in_alpha < min_unknown_in_alpha_preds:
        return None

    # Build final masked world
    masked = build_masked_world()
    masked["unknownInAlphaPreds"] = unknown_in_alpha

    return masked


def sample_alpha_template_skeptical(
    theory: Any,
    rng: Any,
    tier_weights: Optional[Dict[int, float]] = None,
) -> Optional[Any]:
    """
    Sample alpha template with tier weighting for skeptical generation.

    Default weights bias away from Tier3 and require R/S interaction:
    - Tier 0: 0% (unless uses R or S)
    - Tier 1: 35%
    - Tier 2: 55%
    - Tier 3: 10%

    Args:
        theory: TheorySpec object
        rng: Random number generator
        tier_weights: Optional custom tier weights (default: {0: 0, 1: 0.35, 2: 0.55, 3: 0.10})

    Returns:
        AlphaTemplate or None if no valid templates
    """
    from concept_synth.abd_b1_alpha_templates import get_templates_for_tier

    if tier_weights is None:
        tier_weights = {0: 0.0, 1: 0.35, 2: 0.55, 3: 0.10}

    # Get all compatible templates by tier
    templates_by_tier: Dict[int, List[Any]] = {}
    for tier in [0, 1, 2, 3]:
        tier_templates = get_templates_for_tier(tier)
        compatible = [t for t in tier_templates if t.compatible_with_theory(theory)]
        # Filter: must use R or S (the predicates that can be unknown)
        compatible = [t for t in compatible if "R" in t.uses_preds or "S" in t.uses_preds]
        if compatible:
            templates_by_tier[tier] = compatible

    # Build weighted pool
    weighted_templates = []
    for tier, templates in templates_by_tier.items():
        weight = tier_weights.get(tier, 0)
        if weight > 0 and templates:
            # Weight per template in this tier
            per_template_weight = weight / len(templates)
            for t in templates:
                weighted_templates.append((t, per_template_weight))

    if not weighted_templates:
        return None

    # Sample by weight
    total_weight = sum(w for _, w in weighted_templates)
    r = rng.random() * total_weight
    cumulative = 0
    for template, weight in weighted_templates:
        cumulative += weight
        if r <= cumulative:
            return template

    # Fallback
    return weighted_templates[-1][0] if weighted_templates else None
