"""
fo_grounding_z3.py - Shared Finite Grounding Engine for FO Formulas using Z3

This module provides reusable building blocks for:
1) Creating Z3 boolean variables for unknown ground atoms
2) Treating known facts as True/False under CWA-with-unknown-exceptions
3) Grounding FO formulas by finite unrolling of quantifiers
4) Supporting special predicate bindings (e.g., Ab(a) := α(a) for ABD)

Used by:
- Induction Scenario E (exists-completion semantics)
- ABD-Partial (abduction with partial observations) [future]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple

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
from concept_synth.predicate_format import (
    get_binary_extension,
    get_binary_false_extension,
    get_unary_extension,
    get_unary_false_extension,
    parse_binary_pair,
)
from concept_synth.signature_utils import get_world_predicate_arities, split_predicates_by_arity

# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class GroundingContext:
    """
    Context for grounding FO formulas to Z3 expressions.

    Holds all information needed to convert a formula AST to Z3:
    - Domain elements
    - Known true/false atoms
    - Unknown atoms with Z3 boolean variables
    - Optional predicate bindings for special handling
    - Memoization cache
    """

    domain: List[str]
    known_true: Set[Tuple[str, Tuple[str, ...]]]
    known_false: Set[Tuple[str, Tuple[str, ...]]]
    unknown_atoms: Dict[str, Set[Tuple[str, ...]]]  # pred_name -> set of arg tuples
    unknown_vars: Dict[Tuple[str, Tuple[str, ...]], Any]  # (pred, args) -> Z3 BoolRef
    pred_bindings: Dict[str, Callable[[Tuple[str, ...]], Any]] = field(default_factory=dict)
    cache: Dict[Tuple[int, FrozenSet], Any] = field(default_factory=dict)

    def clear_cache(self):
        """Clear the memoization cache."""
        self.cache.clear()


# =============================================================================
# Unknown Variable Creation
# =============================================================================


def build_unknown_vars_from_atoms(
    unknown_atoms: Dict[str, List[Any]], world_id: str, predicate_arities: Dict[str, int]
) -> Dict[Tuple[str, Tuple[str, ...]], Any]:
    """
    Build Z3 boolean variables for unknown atoms.

    Args:
        unknown_atoms: Dictionary mapping predicate names to lists of unknown ground atoms.
                       Format: {'P': ['a0', 'a1'], 'R': ['(a0, a1)', '(a2, a3)']}
        world_id: Unique identifier for this world (for stable variable naming)

    Returns:
        Dictionary mapping (pred_name, args_tuple) to Z3 BoolRef

    Variable naming convention: U__{world_id}__{pred}__{arg0}__{arg1}...
    """
    if not Z3_AVAILABLE:
        raise RuntimeError("Z3 is not installed. Install with: pip install z3-solver")

    unknown_vars = {}

    unary_preds, binary_preds = split_predicates_by_arity(predicate_arities)

    # Process unary predicates
    for pred_name in unary_preds:
        if pred_name in unknown_atoms:
            for const in unknown_atoms[pred_name]:
                var_name = f"U__{world_id}__{pred_name}__{const}"
                key = (pred_name, (str(const),))
                unknown_vars[key] = z3.Bool(var_name)

    # Process binary predicates
    for pred_name in binary_preds:
        if pred_name in unknown_atoms:
            for pair_str in unknown_atoms[pred_name]:
                try:
                    left, right = parse_binary_pair(pair_str)
                except ValueError:
                    continue
                var_name = f"U__{world_id}__{pred_name}__{left}__{right}"
                key = (pred_name, (left, right))
                unknown_vars[key] = z3.Bool(var_name)

    return unknown_vars


def build_unknown_vars(
    world: Dict[str, Any], world_id: str
) -> Dict[Tuple[str, Tuple[str, ...]], Any]:
    """
    Build Z3 boolean variables for unknown atoms in a world.

    Convenience wrapper around build_unknown_vars_from_atoms that extracts
    the unknownAtoms field from a world dictionary.

    Args:
        world: World dictionary with 'unknownAtoms' field
        world_id: Unique identifier for this world

    Returns:
        Dictionary mapping (pred_name, args_tuple) to Z3 BoolRef
    """
    unknown_atoms = world.get("unknownAtoms", {})
    predicate_arities = get_world_predicate_arities(world)
    return build_unknown_vars_from_atoms(unknown_atoms, world_id, predicate_arities)


# =============================================================================
# Known Atom Extraction
# =============================================================================


def build_known_atoms(
    world: Dict[str, Any]
) -> Tuple[Set[Tuple[str, Tuple[str, ...]]], Set[Tuple[str, Tuple[str, ...]]]]:
    """
    Build sets of known true and known false atoms from world predicates.

    Args:
        world: World dictionary with 'predicates' field

    Returns:
        Tuple of (known_true_set, known_false_set)
        Each set contains tuples of (pred_name, args_tuple)
    """
    from concept_synth.predicate_format import (
        get_binary_extension,
        get_binary_false_extension,
        get_unary_extension,
        get_unary_false_extension,
    )

    known_true = set()
    known_false = set()

    predicates = world.get("predicates", {})
    predicate_arities = get_world_predicate_arities(world)
    unary_preds, binary_preds = split_predicates_by_arity(predicate_arities)

    # Unary predicates
    for pred_name in unary_preds:
        if pred_name in predicates:
            true_list = get_unary_extension(predicates.get(pred_name))
            false_list = get_unary_false_extension(predicates.get(pred_name))

            for const in true_list:
                known_true.add((pred_name, (str(const),)))

            for const in false_list:
                known_false.add((pred_name, (str(const),)))

    # Binary predicates
    for pred_name in binary_preds:
        if pred_name in predicates:
            true_list = get_binary_extension(predicates.get(pred_name))
            false_list = get_binary_false_extension(predicates.get(pred_name))

            for pair_str in true_list:
                try:
                    known_true.add((pred_name, parse_binary_pair(pair_str)))
                except ValueError:
                    continue

            for pair_str in false_list:
                try:
                    known_false.add((pred_name, parse_binary_pair(pair_str)))
                except ValueError:
                    continue

    return known_true, known_false


def build_unknown_atoms_set(world: Dict[str, Any]) -> Dict[str, Set[Tuple[str, ...]]]:
    """
    Convert world's unknownAtoms to a structured set format.

    Args:
        world: World dictionary with 'unknownAtoms' field

    Returns:
        Dictionary mapping pred_name to set of arg tuples
    """
    unknown_atoms = world.get("unknownAtoms", {})
    result = {}
    predicate_arities = get_world_predicate_arities(world)
    unary_preds, binary_preds = split_predicates_by_arity(predicate_arities)

    # Unary predicates
    for pred_name in unary_preds:
        if pred_name in unknown_atoms:
            result[pred_name] = {(str(c),) for c in unknown_atoms[pred_name]}

    # Binary predicates
    for pred_name in binary_preds:
        if pred_name in unknown_atoms:
            pairs = set()
            for pair_str in unknown_atoms[pred_name]:
                try:
                    pairs.add(parse_binary_pair(pair_str))
                except ValueError:
                    continue
            result[pred_name] = pairs

    return result


# =============================================================================
# Grounding Context Factory
# =============================================================================


def create_grounding_context(
    world: Dict[str, Any],
    world_id: str,
    pred_bindings: Optional[Dict[str, Callable[[Tuple[str, ...]], Any]]] = None,
) -> GroundingContext:
    """
    Create a complete grounding context from a world dictionary.

    This is the main factory function for creating a GroundingContext.

    Args:
        world: World dictionary with domain, predicates, unknownAtoms
        world_id: Unique identifier for this world
        pred_bindings: Optional special predicate handlers

    Returns:
        GroundingContext ready for use with ground_formula_to_z3
    """
    domain = world.get("domain", [])
    known_true, known_false = build_known_atoms(world)
    unknown_atoms_set = build_unknown_atoms_set(world)
    unknown_vars = build_unknown_vars(world, world_id)

    return GroundingContext(
        domain=domain,
        known_true=known_true,
        known_false=known_false,
        unknown_atoms=unknown_atoms_set,
        unknown_vars=unknown_vars,
        pred_bindings=pred_bindings or {},
        cache={},
    )


# =============================================================================
# Atom Semantics (CWA-with-unknown-exceptions)
# =============================================================================


def atom_to_z3(pred_name: str, args: Tuple[str, ...], ctx: GroundingContext) -> Any:
    """
    Convert an atom to Z3 expression using CWA-with-unknown-exceptions.

    Truth rules (in order of priority):
    1. If pred_name in ctx.pred_bindings: return ctx.pred_bindings[pred_name](args)
    2. If (pred, args) in known_true: return True
    3. If (pred, args) in known_false: return False
    4. If (pred, args) in unknown_vars: return ctx.unknown_vars[(pred, args)]
    5. Else (CWA): return False

    Args:
        pred_name: Predicate name (e.g., 'P', 'R')
        args: Tuple of ground arguments (e.g., ('a0',) or ('a0', 'a1'))
        ctx: Grounding context

    Returns:
        Z3 BoolRef expression
    """
    if not Z3_AVAILABLE:
        raise RuntimeError("Z3 is not installed")

    # Check for special predicate binding first
    if pred_name in ctx.pred_bindings:
        return ctx.pred_bindings[pred_name](args)

    key = (pred_name, args)

    # Check known atoms
    if key in ctx.known_true:
        return z3.BoolVal(True)
    elif key in ctx.known_false:
        return z3.BoolVal(False)
    elif key in ctx.unknown_vars:
        return ctx.unknown_vars[key]
    else:
        # Closed World Assumption: unknown and not in unknown set means false
        return z3.BoolVal(False)


# =============================================================================
# Formula Grounding (FOFormula AST -> Z3)
# =============================================================================


def ground_formula_to_z3(node: FOFormula, env: Dict[str, str], ctx: GroundingContext) -> Any:
    """
    Recursively convert FOL formula to Z3 expression.

    Supports: atoms (P/Q unary, R/S binary), equality (=),
              not, and, or, implies, biconditional, forall, exists

    Uses finite unrolling for quantifiers over ctx.domain.
    Memoizes on (node_id, frozen_env) for efficiency.

    Args:
        node: FOL formula AST node
        env: Current variable bindings (var_name -> domain_constant)
        ctx: Grounding context

    Returns:
        Z3 BoolRef expression
    """
    if not Z3_AVAILABLE:
        raise RuntimeError("Z3 is not installed")

    # Memoization
    cache_key = (id(node), frozenset(env.items()))
    if cache_key in ctx.cache:
        return ctx.cache[cache_key]

    result = _ground_formula_impl(node, env, ctx)
    ctx.cache[cache_key] = result
    return result


def _ground_formula_impl(node: FOFormula, env: Dict[str, str], ctx: GroundingContext) -> Any:
    """Implementation of ground_formula_to_z3."""

    if isinstance(node, Pred):
        # Predicate atom: Pred(name, args)
        args = []
        for arg in node.args:
            if isinstance(arg, Var):
                if arg.name not in env:
                    raise ValueError(f"Unbound variable: {arg.name}")
                args.append(env[arg.name])
            elif isinstance(arg, Constant):
                args.append(arg.name)
            else:
                raise ValueError(f"Unexpected argument type: {type(arg)}")

        return atom_to_z3(node.name, tuple(args), ctx)

    elif isinstance(node, Eq):
        # Equality: (= x y)
        def get_const(term):
            if isinstance(term, Var):
                if term.name not in env:
                    raise ValueError(f"Unbound variable: {term.name}")
                return env[term.name]
            elif isinstance(term, Constant):
                return term.name
            else:
                raise ValueError(f"Unexpected term type: {type(term)}")

        left_const = get_const(node.left)
        right_const = get_const(node.right)
        return z3.BoolVal(left_const == right_const)

    elif isinstance(node, FONot):
        sub = ground_formula_to_z3(node.child, env, ctx)
        return z3.Not(sub)

    elif isinstance(node, FOAnd):
        left = ground_formula_to_z3(node.left, env, ctx)
        right = ground_formula_to_z3(node.right, env, ctx)
        return z3.And(left, right)

    elif isinstance(node, FOOr):
        left = ground_formula_to_z3(node.left, env, ctx)
        right = ground_formula_to_z3(node.right, env, ctx)
        return z3.Or(left, right)

    elif isinstance(node, FOImplies):
        left = ground_formula_to_z3(node.left, env, ctx)
        right = ground_formula_to_z3(node.right, env, ctx)
        return z3.Implies(left, right)

    elif isinstance(node, FOBiconditional):
        left = ground_formula_to_z3(node.left, env, ctx)
        right = ground_formula_to_z3(node.right, env, ctx)
        return z3.And(z3.Implies(left, right), z3.Implies(right, left))

    elif isinstance(node, Forall):
        # Finite unrolling: forall x. φ(x) = φ(a0) ∧ φ(a1) ∧ ... ∧ φ(an)
        var_name = node.var.name
        conjuncts = []
        for const in ctx.domain:
            new_env = env.copy()
            new_env[var_name] = const
            sub_expr = ground_formula_to_z3(node.body, new_env, ctx)
            conjuncts.append(sub_expr)

        if len(conjuncts) == 0:
            return z3.BoolVal(True)
        elif len(conjuncts) == 1:
            return conjuncts[0]
        else:
            return z3.And(*conjuncts)

    elif isinstance(node, Exists):
        # Finite unrolling: exists x. φ(x) = φ(a0) ∨ φ(a1) ∨ ... ∨ φ(an)
        var_name = node.var.name
        disjuncts = []
        for const in ctx.domain:
            new_env = env.copy()
            new_env[var_name] = const
            sub_expr = ground_formula_to_z3(node.body, new_env, ctx)
            disjuncts.append(sub_expr)

        if len(disjuncts) == 0:
            return z3.BoolVal(False)
        elif len(disjuncts) == 1:
            return disjuncts[0]
        else:
            return z3.Or(*disjuncts)

    else:
        raise ValueError(f"Unsupported formula type: {type(node)}")


# =============================================================================
# Constraint Building Helpers
# =============================================================================


def build_label_match_constraints(
    formula_ast: FOFormula, t_true: Set[str], t_false: Set[str], ctx: GroundingContext
) -> Any:
    """
    Build Z3 constraints requiring formula to match target labels.

    For each element a in domain:
    - If a in t_true: require φ(a) = True
    - If a in t_false: require φ(a) = False

    Args:
        formula_ast: Formula AST to ground
        t_true: Set of elements where formula should be true
        t_false: Set of elements where formula should be false
        ctx: Grounding context

    Returns:
        Z3 And expression of all constraints
    """
    if not Z3_AVAILABLE:
        raise RuntimeError("Z3 is not installed")

    constraints = []

    for elem in ctx.domain:
        env = {"x": elem}
        formula_expr = ground_formula_to_z3(formula_ast, env, ctx)

        if elem in t_true:
            constraints.append(formula_expr)
        elif elem in t_false:
            constraints.append(z3.Not(formula_expr))
        # Elements not in either set are ignored (unconstrained)

    if len(constraints) == 0:
        return z3.BoolVal(True)
    elif len(constraints) == 1:
        return constraints[0]
    else:
        return z3.And(*constraints)


def build_match_constraints_from_world(
    world: Dict[str, Any], formula_ast: FOFormula, ctx: GroundingContext
) -> Any:
    """
    Build Z3 constraints from world's targetExtension.

    Convenience wrapper that extracts T_true/T_false from world.

    Args:
        world: World dictionary with 'targetExtension' and 'domain'
        formula_ast: Formula AST to ground
        ctx: Grounding context

    Returns:
        Z3 And expression of all constraints
    """
    domain = world.get("domain", [])
    target = world.get("targetExtension", {})
    t_true_set = set(target.get("T_true", []))
    t_false_set = set(domain) - t_true_set

    return build_label_match_constraints(formula_ast, t_true_set, t_false_set, ctx)


def ground_axioms_to_z3(axioms_ast_list: List[FOFormula], ctx: GroundingContext) -> List[Any]:
    """
    Ground a list of axiom formulas to Z3 expressions.

    Axioms are typically closed formulas (no free variables).

    Args:
        axioms_ast_list: List of axiom formula ASTs
        ctx: Grounding context

    Returns:
        List of Z3 BoolRef expressions
    """
    if not Z3_AVAILABLE:
        raise RuntimeError("Z3 is not installed")

    grounded_axioms = []
    for axiom in axioms_ast_list:
        # Axioms have no free variables, so env is empty
        axiom_expr = ground_formula_to_z3(axiom, {}, ctx)
        grounded_axioms.append(axiom_expr)

    return grounded_axioms


# =============================================================================
# Single Element Constraint Building
# =============================================================================


def build_element_constraint(
    formula_ast: FOFormula, element: str, target_label: bool, ctx: GroundingContext
) -> Any:
    """
    Build Z3 constraint for a single element.

    Args:
        formula_ast: Formula AST to ground
        element: Domain element
        target_label: True if formula should be true at element, False otherwise
        ctx: Grounding context

    Returns:
        Z3 BoolRef constraint
    """
    if not Z3_AVAILABLE:
        raise RuntimeError("Z3 is not installed")

    env = {"x": element}
    formula_expr = ground_formula_to_z3(formula_ast, env, ctx)

    if target_label:
        return formula_expr
    else:
        return z3.Not(formula_expr)


# =============================================================================
# Direct Evaluation (without Z3)
# =============================================================================


def eval_formula_direct(
    node: FOFormula,
    env: Dict[str, str],
    domain: List[str],
    known_true: Set[Tuple[str, Tuple[str, ...]]],
    known_false: Optional[Set[Tuple[str, Tuple[str, ...]]]] = None,
) -> bool:
    """
    Direct evaluation of formula without Z3.

    Uses closed-world assumption: atoms not in known_true are False.
    This is useful for evaluating under fixed completions.

    Args:
        node: Formula AST node
        env: Variable bindings
        domain: Domain elements
        known_true: Set of true atoms
        known_false: Optional set of false atoms (not used, for API compatibility)

    Returns:
        Boolean truth value
    """
    if isinstance(node, Pred):
        args = []
        for arg in node.args:
            if isinstance(arg, Var):
                if arg.name not in env:
                    raise ValueError(f"Unbound variable: {arg.name}")
                args.append(env[arg.name])
            elif isinstance(arg, Constant):
                args.append(arg.name)
            else:
                raise ValueError(f"Unexpected argument type: {type(arg)}")

        key = (node.name, tuple(args))
        return key in known_true

    elif isinstance(node, Eq):

        def get_const(term):
            if isinstance(term, Var):
                if term.name not in env:
                    raise ValueError(f"Unbound variable: {term.name}")
                return env[term.name]
            elif isinstance(term, Constant):
                return term.name
            else:
                raise ValueError(f"Unexpected term type: {type(term)}")

        return get_const(node.left) == get_const(node.right)

    elif isinstance(node, FONot):
        return not eval_formula_direct(node.child, env, domain, known_true, known_false)

    elif isinstance(node, FOAnd):
        return eval_formula_direct(
            node.left, env, domain, known_true, known_false
        ) and eval_formula_direct(node.right, env, domain, known_true, known_false)

    elif isinstance(node, FOOr):
        return eval_formula_direct(
            node.left, env, domain, known_true, known_false
        ) or eval_formula_direct(node.right, env, domain, known_true, known_false)

    elif isinstance(node, FOImplies):
        left = eval_formula_direct(node.left, env, domain, known_true, known_false)
        right = eval_formula_direct(node.right, env, domain, known_true, known_false)
        return (not left) or right

    elif isinstance(node, FOBiconditional):
        left = eval_formula_direct(node.left, env, domain, known_true, known_false)
        right = eval_formula_direct(node.right, env, domain, known_true, known_false)
        return left == right

    elif isinstance(node, Forall):
        var_name = node.var.name
        for const in domain:
            new_env = env.copy()
            new_env[var_name] = const
            if not eval_formula_direct(node.body, new_env, domain, known_true, known_false):
                return False
        return True

    elif isinstance(node, Exists):
        var_name = node.var.name
        for const in domain:
            new_env = env.copy()
            new_env[var_name] = const
            if eval_formula_direct(node.body, new_env, domain, known_true, known_false):
                return True
        return False

    else:
        raise ValueError(f"Unsupported formula type: {type(node)}")


def eval_formula_under_completion(
    world: Dict[str, Any], formula_ast: FOFormula, all_unknowns_true: bool
) -> Dict[str, bool]:
    """
    Evaluate formula at each domain element under a fixed completion.

    This does NOT use Z3 - it's a direct evaluation where all unknown atoms
    are assigned the same truth value.

    Args:
        world: World dictionary
        formula_ast: Parsed formula AST
        all_unknowns_true: If True, all unknown atoms = True; else all = False

    Returns:
        Dictionary mapping element -> truth value of formula at that element
    """
    domain = world.get("domain", [])
    unknown_atoms = world.get("unknownAtoms", {})

    # Build known atoms
    known_true, known_false = build_known_atoms(world)

    # Add unknown atoms to known_true or known_false based on completion
    for pred_name in ["P", "Q"]:
        if pred_name in unknown_atoms:
            for const in unknown_atoms[pred_name]:
                key = (pred_name, (str(const),))
                if all_unknowns_true:
                    known_true.add(key)
                else:
                    known_false.add(key)

    for pred_name in ["R", "S"]:
        if pred_name in unknown_atoms:
            for pair_str in unknown_atoms[pred_name]:
                pair_str = str(pair_str).strip("()")
                parts = [p.strip() for p in pair_str.split(",")]
                if len(parts) == 2:
                    key = (pred_name, (parts[0], parts[1]))
                    if all_unknowns_true:
                        known_true.add(key)
                    else:
                        known_false.add(key)

    # Evaluate formula at each element
    results = {}
    for elem in domain:
        env = {"x": elem}
        try:
            value = eval_formula_direct(formula_ast, env, domain, known_true, known_false)
            results[elem] = value
        except Exception:
            results[elem] = False  # Default to False on error

    return results


# =============================================================================
# Utility Functions
# =============================================================================


def check_z3_available() -> bool:
    """Check if Z3 is available."""
    return Z3_AVAILABLE


def create_solver(timeout_ms: int = 5000) -> Any:
    """
    Create a Z3 solver with timeout.

    Args:
        timeout_ms: Timeout in milliseconds

    Returns:
        Z3 Solver instance
    """
    if not Z3_AVAILABLE:
        raise RuntimeError("Z3 is not installed")

    solver = z3.Solver()
    solver.set("timeout", timeout_ms)
    return solver
