"""Prompt rendering for the public INDUCTION benchmark."""

from __future__ import annotations

from importlib import resources
from typing import Any, Dict, List, Optional, Tuple

from concept_synth.predicate_format import get_binary_extension, get_unary_extension
from concept_synth.signature_utils import (
    get_problem_prompt_variant,
    get_problem_signature,
    get_world_predicate_arities,
    split_predicates_by_arity,
)


TASK_TO_SCENARIO = {
    "FullObs": "AD",
    "CI": "C",
    "EC": "E",
}

SCENARIO_TO_TASK = {
    "AD": "FullObs",
    "C": "CI",
    "E": "EC",
}


def normalize_task(record_or_problem: Dict[str, Any]) -> str:
    task = record_or_problem.get("task")
    if isinstance(task, str) and task in TASK_TO_SCENARIO:
        return task
    problem = record_or_problem.get("problem", record_or_problem)
    scenario = str(problem.get("scenario") or record_or_problem.get("scenario") or "AD")
    return SCENARIO_TO_TASK.get(scenario, scenario)


def normalize_scenario(record_or_problem: Dict[str, Any]) -> str:
    problem = record_or_problem.get("problem", record_or_problem)
    scenario = str(problem.get("scenario") or record_or_problem.get("scenario") or "")
    if scenario:
        return scenario
    return TASK_TO_SCENARIO.get(normalize_task(record_or_problem), "AD")


def _template_stem(task: str) -> str:
    return {"FullObs": "fullobs", "CI": "ci", "EC": "ec"}.get(task, "fullobs")


def load_prompt_template(task: str) -> Tuple[str, str, str]:
    """Load the released system, task, and suffix templates for a task."""
    stem = _template_stem(task)
    prompt_pkg = resources.files("concept_synth.induction") / "prompts"

    def read_optional(name: str) -> str:
        path = prompt_pkg / name
        if path.is_file():
            return path.read_text(encoding="utf-8")
        return ""

    return (
        read_optional(f"{stem}_system.txt"),
        read_optional(f"{stem}_task.txt"),
        read_optional(f"{stem}_suffix.txt"),
    )


def format_world_for_prompt(
    world: Dict[str, Any],
    scenario: str,
    predicate_arities: Optional[Dict[str, int]] = None,
) -> str:
    """Format a finite world using the same public prompt language as the release."""
    lines: List[str] = []

    world_id = world.get("worldId", "unknown")
    domain = world.get("domain", [])
    predicates = world.get("predicates", {})
    target = world.get("targetExtension", {})
    unknown_atoms = world.get("unknownAtoms", {})

    lines.append(f"### World: {world_id}")
    lines.append(f"Domain: {{{', '.join(domain)}}}")
    lines.append("")

    world_arities = get_world_predicate_arities(world, predicate_arities)
    unary_preds, binary_preds = split_predicates_by_arity(world_arities)

    if scenario == "E" and unknown_atoms and isinstance(unknown_atoms, dict):
        unknown_pred_set = set(unknown_atoms)

        lines.append("**Fully Observed Predicates:**")
        for pred_name in unary_preds:
            if pred_name in predicates and pred_name not in unknown_pred_set:
                values = get_unary_extension(predicates.get(pred_name))
                lines.append(f"- {pred_name}: {', '.join(str(x) for x in values) if values else '(none)'}")
        for pred_name in binary_preds:
            if pred_name in predicates and pred_name not in unknown_pred_set:
                values = get_binary_extension(predicates.get(pred_name))
                lines.append(f"- {pred_name}: {', '.join(str(x) for x in values) if values else '(none)'}")
        lines.append("")

        lines.append("**Partially Observed Predicates:**")
        for pred_name in unary_preds:
            if pred_name in predicates and pred_name in unknown_pred_set:
                values = get_unary_extension(predicates.get(pred_name))
                lines.append(f"- {pred_name} (known TRUE): {', '.join(str(x) for x in values) if values else '(none)'}")
        for pred_name in binary_preds:
            if pred_name in predicates and pred_name in unknown_pred_set:
                values = get_binary_extension(predicates.get(pred_name))
                lines.append(f"- {pred_name} (known TRUE): {', '.join(str(x) for x in values) if values else '(none)'}")
        lines.append("")

        lines.append("**Unknown Atoms** (truth value not observed):")
        for pred_name in unary_preds + binary_preds:
            atoms = unknown_atoms.get(pred_name, [])
            if atoms:
                lines.append(f"- {pred_name}: {', '.join(str(x) for x in atoms)}")
    else:
        lines.append("**Predicates:**")
        for pred_name in unary_preds:
            if pred_name in predicates:
                values = get_unary_extension(predicates.get(pred_name))
                lines.append(f"- {pred_name}: {', '.join(str(x) for x in values) if values else '(none)'}")
        for pred_name in binary_preds:
            if pred_name in predicates:
                values = get_binary_extension(predicates.get(pred_name))
                lines.append(f"- {pred_name}: {', '.join(str(x) for x in values) if values else '(none)'}")

    lines.append("")
    lines.append("**Target T(x):**")
    t_true = target.get("T_true", [])
    t_false = target.get("T_false", [])
    lines.append(f"- T is TRUE for: {{{', '.join(t_true)}}}")
    if t_false:
        lines.append(f"- T is FALSE for: {{{', '.join(t_false)}}}")

    return "\n".join(lines)


def build_prompt(record: Dict[str, Any]) -> Tuple[str, str]:
    """Build the full user prompt and optional system prompt for a released record."""
    problem = record.get("problem", record)
    task = normalize_task(record)
    scenario = normalize_scenario(record)
    problem_signature = get_problem_signature(problem)
    _ = get_problem_prompt_variant(problem)  # Kept for forward compatibility with renamed releases.
    system_prompt, task_prompt, suffix_prompt = load_prompt_template(task)

    worlds = problem.get("worlds", [])
    task_info = problem.get("task", {})
    world_sections: List[str] = []

    if scenario == "AD":
        train_ids = set(task_info.get("trainWorldIds", []))
        test_ids = set(task_info.get("testWorldIds", []))
        world_sections.append("## Training Worlds (learn from these):\n")
        for world in worlds:
            if not train_ids or world.get("worldId") in train_ids:
                world_sections.append(format_world_for_prompt(world, scenario, problem_signature))
                world_sections.append("")
        test_worlds = [world for world in worlds if world.get("worldId") in test_ids]
        if test_worlds:
            world_sections.append("\n## Test Worlds (generalize to these):\n")
            for world in test_worlds:
                world_sections.append(format_world_for_prompt(world, scenario, problem_signature))
                world_sections.append("")

    elif scenario == "C":
        yes_ids = set(task_info.get("yesWorldIds", []))
        no_ids = set(task_info.get("noWorldIds", []))
        world_sections.append("## YES Worlds (formula is perfect here):\n")
        for world in worlds:
            if world.get("worldId") in yes_ids or world.get("splitLabel") is True:
                world_sections.append(format_world_for_prompt(world, scenario, problem_signature))
                world_sections.append("")
        world_sections.append("\n## NO Worlds (formula fails here):\n")
        for world in worlds:
            if world.get("worldId") in no_ids or world.get("splitLabel") is False:
                world_sections.append(format_world_for_prompt(world, scenario, problem_signature))
                world_sections.append("")

    elif scenario == "E":
        train_ids = set(task_info.get("trainWorldIds", []))
        world_sections.append("## Worlds with Partial Observations:\n")
        for world in worlds:
            if not train_ids or world.get("worldId") in train_ids:
                world_sections.append(format_world_for_prompt(world, scenario, problem_signature))
                world_sections.append("")
    else:
        raise ValueError(f"Unsupported INDUCTION scenario: {scenario}")

    worlds_text = "\n".join(world_sections)
    full_prompt = f"{task_prompt}\n\n{worlds_text}\n\n{suffix_prompt}".strip() + "\n"
    return full_prompt, system_prompt
