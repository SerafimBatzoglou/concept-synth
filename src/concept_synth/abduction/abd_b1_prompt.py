"""
abd_b1_prompt.py - Prompt Builders for ABD-B1 Abduction Tasks

Generates prompts for LLMs to solve ABD-Full, ABD-Partial, and ABD-Skeptical scenarios.

Uses template files from the prompts/ directory:
- abd_full_scenario_task.txt / abd_full_scenario_suffix.txt
- abd_partial_scenario_task.txt / abd_partial_scenario_suffix.txt
- abd_skeptical_scenario_task.txt / abd_skeptical_scenario_suffix.txt

The prompts explain:
- The task: output alpha(x) defining abnormality Ab(x)
- Semantics: Ab(x) is treated as equivalent to alpha(x)
- Validity conditions (scenario-specific):
  - ABD_FULL: closed-world, full observation
  - ABD_PARTIAL: existential completion (SOME completion must work)
  - ABD_SKEPTICAL: universal completion (ALL completions must work)
- Parsimony goal: minimize abnormal set (best-case for PARTIAL, worst-case for SKEPTICAL)
- Theory axioms and world facts

The prompts include dynamic AllowedAlphaPredicates and ForbiddenAlphaPredicates
based on the theory specification.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Set

# =============================================================================
# Template Loading
# =============================================================================


def _get_prompts_dir() -> str:
    """Get the path to the prompts directory."""
    return os.path.join(os.path.dirname(__file__), "prompts")


def _load_template(filename: str) -> str:
    """Load a template file from the prompts directory."""
    path = os.path.join(_get_prompts_dir(), filename)
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read()
    return ""


def load_abd_full_templates() -> tuple[str, str]:
    """Load ABD-Full task and suffix templates."""
    task = _load_template("abd_full_scenario_task.txt")
    suffix = _load_template("abd_full_scenario_suffix.txt")
    return task, suffix


def load_abd_partial_templates() -> tuple[str, str]:
    """Load ABD-Partial task and suffix templates."""
    task = _load_template("abd_partial_scenario_task.txt")
    suffix = _load_template("abd_partial_scenario_suffix.txt")
    return task, suffix


def load_abd_skeptical_templates() -> tuple[str, str]:
    """Load ABD-Skeptical task and suffix templates."""
    task = _load_template("abd_skeptical_scenario_task.txt")
    suffix = _load_template("abd_skeptical_scenario_suffix.txt")
    return task, suffix


# =============================================================================
# Predicate Scope Formatting
# =============================================================================


def format_predicate_scope(allowed_preds: Set[str], forbidden_preds: Set[str]) -> str:
    """
    Format the AllowedAlphaPredicates and ForbiddenAlphaPredicates block.

    Args:
        allowed_preds: Set of predicate names allowed in alpha
        forbidden_preds: Set of predicate names forbidden in alpha

    Returns:
        Formatted string with both lines
    """
    # Sort for deterministic output
    allowed_list = sorted(allowed_preds)
    forbidden_list = sorted(forbidden_preds)

    lines = [
        f"**AllowedAlphaPredicates**: {json.dumps(allowed_list)}",
        f"**ForbiddenAlphaPredicates**: {json.dumps(forbidden_list)}",
    ]
    return "\n".join(lines)


def get_predicate_scope_from_problem(problem: Dict[str, Any]) -> tuple[Set[str], Set[str]]:
    """
    Extract allowed and forbidden predicates from problem metadata or theory library.

    Priority:
    1. TheorySpec from theory library (using theoryId)
    2. Explicit problem['problem']['allowedAlphaPreds'] (new schema)
    3. Explicit problem['theory']['allowedAlphaPredicates'] / forbiddenAlphaPredicates (old schema)
    4. problem['problemDescription']['allowedAlphaPredicates'] / etc.
    5. Falls back to default P, Q, R, S with Ab forbidden

    Args:
        problem: Problem dictionary

    Returns:
        Tuple of (allowed_preds, forbidden_preds)
    """
    from .abd_b1_theory_library import get_theory

    prob_data = problem.get("problem", problem)
    theory = prob_data.get("theory", {})
    desc = problem.get("problemDescription", {})

    # Priority 1: Try to get from TheorySpec using theoryId
    theory_id = theory.get("theoryId", "") or prob_data.get("theoryId", "")
    if theory_id:
        try:
            theory_spec = get_theory(theory_id)
            allowed = theory_spec.get_effective_allowed_preds()
            forbidden = theory_spec.get_forbidden_preds()
            return allowed, forbidden
        except KeyError:
            pass  # Unknown theory, fall through

    # Priority 2: Try to get from the normalized problem schema
    allowed = prob_data.get("allowedAlphaPreds")

    # Priority 3: Try to get from theory dict
    if allowed is None:
        allowed = theory.get("allowedAlphaPredicates")
    forbidden = theory.get("forbiddenAlphaPredicates")

    # Priority 4: Try problemDescription if not in theory
    if allowed is None:
        allowed = desc.get("allowedAlphaPredicates")
    if forbidden is None:
        forbidden = desc.get("forbiddenAlphaPredicates")

    # Convert to sets
    if allowed is not None:
        allowed = set(allowed) if isinstance(allowed, list) else allowed
    else:
        # Default: all standard predicates
        allowed = {"P", "Q", "R", "S"}

    if forbidden is not None:
        forbidden = set(forbidden) if isinstance(forbidden, list) else forbidden
    else:
        # Default: only Ab is forbidden
        forbidden = {"Ab"}

    return allowed, forbidden


# =============================================================================
# World Formatting
# =============================================================================


def format_world_full(world: Dict[str, Any], world_idx: int) -> str:
    """Format a fully-observed world for the prompt."""
    lines = []
    world_id = world.get("worldId", f"W{world_idx}")
    domain = world.get("domain", [])
    predicates = world.get("predicates", {})

    lines.append(f"### World {world_id}")
    lines.append(f"Domain: {{{', '.join(domain)}}}")
    lines.append("")
    lines.append("**Predicates** (Closed World Assumption: unlisted atoms are false):")

    # Unary predicates
    for pred in ["P", "Q"]:
        if pred in predicates:
            true_list = predicates[pred].get("true", [])
            if true_list:
                lines.append(f"- {pred}: {{{', '.join(str(x) for x in true_list)}}}")
            else:
                lines.append(f"- {pred}: (none)")

    # Binary predicates
    for pred in ["R", "S"]:
        if pred in predicates:
            true_list = predicates[pred].get("true", [])
            if true_list:
                pairs = [str(p) for p in true_list]
                lines.append(f"- {pred}: {{{', '.join(pairs)}}}")
            else:
                lines.append(f"- {pred}: (none)")

    return "\n".join(lines)


def format_world_partial(world: Dict[str, Any], world_idx: int) -> str:
    """
    Format a partially-observed world for the prompt.

    Semantics:
    - Known TRUE facts are listed explicitly (fixed, cannot be flipped)
    - Unknown atoms are listed explicitly (can be completed either way)
    - Everything else is KNOWN FALSE (implicit, not listed)
    """
    lines = []
    world_id = world.get("worldId", f"W{world_idx}")
    domain = world.get("domain", [])
    predicates = world.get("predicates", {})
    unknown_atoms = world.get("unknownAtoms", {})

    lines.append(f"### World {world_id}")
    lines.append(f"Domain: {{{', '.join(domain)}}}")
    lines.append("")

    # Known TRUE facts only (FALSE is implicit for non-unknown atoms)
    lines.append("**Known Facts** (unlisted atoms that are not Unknown are known FALSE):")
    has_known = False
    for pred in ["P", "Q"]:
        if pred in predicates:
            true_list = predicates[pred].get("true", [])
            if true_list:
                has_known = True
                lines.append(f"- {pred}: {{{', '.join(str(x) for x in true_list)}}}")

    for pred in ["R", "S"]:
        if pred in predicates:
            true_list = predicates[pred].get("true", [])
            if true_list:
                has_known = True
                pairs = [str(p) for p in true_list]
                lines.append(f"- {pred}: {{{', '.join(pairs)}}}")

    if not has_known:
        lines.append("- (none)")

    # Unknown atoms
    lines.append("")
    lines.append("**Unknown Atoms** (truth value not observed, can be completed either way):")
    has_unknowns = False
    for pred in ["P", "Q"]:
        if pred in unknown_atoms and unknown_atoms[pred]:
            has_unknowns = True
            lines.append(f"- {pred}: {{{', '.join(str(x) for x in unknown_atoms[pred])}}}")

    for pred in ["R", "S"]:
        if pred in unknown_atoms and unknown_atoms[pred]:
            has_unknowns = True
            pairs = [str(p) for p in unknown_atoms[pred]]
            lines.append(f"- {pred}: {{{', '.join(pairs)}}}")

    if not has_unknowns:
        lines.append("- (none)")

    return "\n".join(lines)


def format_axioms(axioms: List[str]) -> str:
    """Format axioms for the prompt."""
    lines = []
    for i, axiom in enumerate(axioms, 1):
        lines.append(f"{i}. `{axiom}`")
    return "\n".join(lines)


# =============================================================================
# ABD-Full Prompt Builder
# =============================================================================


def build_abd_full_prompt(problem: Dict[str, Any]) -> str:
    """
    Build a prompt for ABD-Full scenario using template files.

    Args:
        problem: Problem dictionary with theory, worlds, etc.
                 Can be either {'problem': {...}} or flat format.

    Returns:
        Formatted prompt string
    """
    # Handle nested 'problem' key
    prob_data = problem.get("problem", problem)

    # Load templates
    task_template, suffix_template = load_abd_full_templates()

    # Extract theory info (handle both old and new schema)
    theory = prob_data.get("theory", {})
    axioms = theory.get("axioms", []) or prob_data.get("axioms", [])
    theory_id = theory.get("theoryId", "") or prob_data.get("theoryId", "Unknown")
    theory_desc = theory.get("description", "") or prob_data.get("theoryName", "")

    # Extract worlds (training only) - handle both 'worlds' and 'trainWorlds' schemas
    all_worlds = prob_data.get("worlds", [])
    if all_worlds:
        train_worlds = [w for w in all_worlds if not w.get("isHeldout", False)]
    else:
        # New schema: trainWorlds is already filtered
        train_worlds = prob_data.get("trainWorlds", [])

    # Get predicate scope
    allowed_preds, forbidden_preds = get_predicate_scope_from_problem(problem)
    predicate_scope = format_predicate_scope(allowed_preds, forbidden_preds)

    # Build theory section
    theory_section = f"**Theory ID**: {theory_id}\n"
    if theory_desc:
        theory_section += f"**Description**: {theory_desc}\n"
    theory_section += "\n**Axioms**:\n"
    theory_section += format_axioms(axioms)

    # Build worlds section
    worlds_section = ""
    for i, world in enumerate(train_worlds):
        worlds_section += format_world_full(world, i) + "\n\n"

    # Combine: task + predicate scope + theory + worlds + suffix
    prompt = task_template
    prompt += predicate_scope + "\n\n"
    prompt += theory_section + "\n\n"
    prompt += "## Training Worlds\n\n"
    prompt += worlds_section.strip() + "\n"
    prompt += suffix_template

    return prompt


# =============================================================================
# ABD-Partial Prompt Builder
# =============================================================================


def build_abd_partial_prompt(problem: Dict[str, Any]) -> str:
    """
    Build a prompt for ABD-Partial scenario using template files.

    Args:
        problem: Problem dictionary with theory, worlds, unknownAtoms, etc.
                 Can be either {'problem': {...}} or flat format.

    Returns:
        Formatted prompt string
    """
    # Handle nested 'problem' key
    prob_data = problem.get("problem", problem)

    # Load templates
    task_template, suffix_template = load_abd_partial_templates()

    # Extract theory info (handle both old and new schema)
    theory = prob_data.get("theory", {})
    axioms = theory.get("axioms", []) or prob_data.get("axioms", [])
    theory_id = theory.get("theoryId", "") or prob_data.get("theoryId", "Unknown")
    theory_desc = theory.get("description", "") or prob_data.get("theoryName", "")

    # Extract worlds (training only) - handle both 'worlds' and 'trainWorlds' schemas
    all_worlds = prob_data.get("worlds", [])
    if all_worlds:
        train_worlds = [w for w in all_worlds if not w.get("isHeldout", False)]
    else:
        # New schema: trainWorlds is already filtered
        train_worlds = prob_data.get("trainWorlds", [])

    # Get predicate scope
    allowed_preds, forbidden_preds = get_predicate_scope_from_problem(problem)
    predicate_scope = format_predicate_scope(allowed_preds, forbidden_preds)

    # Build theory section
    theory_section = f"**Theory ID**: {theory_id}\n"
    if theory_desc:
        theory_section += f"**Description**: {theory_desc}\n"
    theory_section += "\n**Axioms**:\n"
    theory_section += format_axioms(axioms)

    # Build worlds section
    worlds_section = ""
    for i, world in enumerate(train_worlds):
        worlds_section += format_world_partial(world, i) + "\n\n"

    # Combine: task + predicate scope + theory + worlds + suffix
    prompt = task_template
    prompt += predicate_scope + "\n\n"
    prompt += theory_section + "\n\n"
    prompt += "## Training Worlds\n\n"
    prompt += worlds_section.strip() + "\n"
    prompt += suffix_template

    return prompt


# =============================================================================
# ABD-Skeptical Prompt Builder
# =============================================================================


def build_abd_skeptical_prompt(problem: Dict[str, Any]) -> str:
    """
    Build a prompt for ABD-Skeptical scenario using template files.

    ABD-Skeptical uses universal/forall completion semantics:
    - The formula must work for ALL completions of unknown atoms
    - Cost is measured as WORST-CASE (maximum) abnormality count

    Args:
        problem: Problem dictionary with theory, worlds, unknownAtoms, etc.
                 Can be either {'problem': {...}} or flat format.

    Returns:
        Formatted prompt string
    """
    # Handle nested 'problem' key
    prob_data = problem.get("problem", problem)

    # Load templates
    task_template, suffix_template = load_abd_skeptical_templates()

    # Extract theory info (handle both old and new schema)
    theory = prob_data.get("theory", {})
    axioms = theory.get("axioms", []) or prob_data.get("axioms", [])
    theory_id = theory.get("theoryId", "") or prob_data.get("theoryId", "Unknown")
    theory_desc = theory.get("description", "") or prob_data.get("theoryName", "")

    # Extract worlds (training only) - handle both 'worlds' and 'trainWorlds' schemas
    all_worlds = prob_data.get("worlds", [])
    if all_worlds:
        train_worlds = [w for w in all_worlds if not w.get("isHeldout", False)]
    else:
        # New schema: trainWorlds is already filtered
        train_worlds = prob_data.get("trainWorlds", [])

    # Get predicate scope
    allowed_preds, forbidden_preds = get_predicate_scope_from_problem(problem)
    predicate_scope = format_predicate_scope(allowed_preds, forbidden_preds)

    # Build theory section
    theory_section = f"**Theory ID**: {theory_id}\n"
    if theory_desc:
        theory_section += f"**Description**: {theory_desc}\n"
    theory_section += "\n**Axioms**:\n"
    theory_section += format_axioms(axioms)

    # Build worlds section - use partial format (with unknown atoms)
    worlds_section = ""
    for i, world in enumerate(train_worlds):
        worlds_section += format_world_partial(world, i) + "\n\n"

    # Combine: task + predicate scope + theory + worlds + suffix
    prompt = task_template
    prompt += predicate_scope + "\n\n"
    prompt += theory_section + "\n\n"
    prompt += "## Training Worlds\n\n"
    prompt += worlds_section.strip() + "\n"
    prompt += suffix_template

    return prompt


# =============================================================================
# Unified Prompt Builder
# =============================================================================


def build_abd_b1_prompt(problem: Dict[str, Any]) -> str:
    """
    Build a prompt for ABD-B1 scenario (auto-detect Full vs Partial vs Skeptical).

    Args:
        problem: Problem dictionary

    Returns:
        Formatted prompt string
    """
    prob_data = problem.get("problem", problem)
    scenario = prob_data.get("scenario", "ABD_FULL")

    if scenario == "ABD_PARTIAL":
        return build_abd_partial_prompt(problem)
    elif scenario == "ABD_SKEPTICAL":
        return build_abd_skeptical_prompt(problem)
    else:
        return build_abd_full_prompt(problem)


def get_abd_b1_system_prompt() -> str:
    """Get the system prompt for ABD-B1 tasks."""
    prompt = _load_template("system_prompt.txt")
    if prompt:
        return prompt.strip()
    return """You are an expert in first-order logic and abductive reasoning.
Your task is to find concise formulas that explain abnormal behavior in logical systems.
Always output valid JSON with the required fields."""


# =============================================================================
# Prompt Validation
# =============================================================================


def validate_prompt_no_leaks(prompt: str) -> List[str]:
    """
    Check that the prompt doesn't leak reserved words.

    Uses word boundary matching to avoid false positives
    (e.g., "optimize" should not trigger "opt").

    Returns:
        List of leaked words found (empty if clean)
    """
    import re

    reserved_words = ["gold", "optimal", "opt", "decoy", "distractor", "answer", "solution"]
    found = []
    prompt_lower = prompt.lower()
    for word in reserved_words:
        # Use word boundary regex to avoid false positives
        pattern = r"\b" + re.escape(word) + r"\b"
        if re.search(pattern, prompt_lower):
            found.append(word)
    return found
