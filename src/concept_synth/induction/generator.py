"""Clean deterministic generators for public INDUCTION benchmark instances.

These generators are intentionally small and release-facing. They create new
instances in the public schema, but they do not attempt to reproduce the full
internal calibration pipeline used for the ICML benchmark release.
"""

from __future__ import annotations

import gzip
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import yaml

from concept_synth.fol.formulas import Exists, FOAnd, FONot, FOOr, Forall, Pred, Var
from concept_synth.fol.model import FiniteModel
from concept_synth.metrics import ast_size, quantifier_depth
from concept_synth.sexpr_printer import to_sexpr
from concept_synth.target import compute_target_extension

from .evaluator import evaluate_prediction


SIGNATURE = {
    "predicates": [
        {"name": "P", "arity": 1},
        {"name": "Q", "arity": 1},
        {"name": "R", "arity": 2},
        {"name": "S", "arity": 2},
        {"name": "T", "arity": 1},
    ]
}

TASK_TO_SCENARIO = {"FullObs": "AD", "CI": "C", "EC": "E"}


@dataclass(frozen=True)
class FormulaTemplate:
    family_id: str
    subfamily_key: str
    description: str
    formula: Any

    @property
    def sexpr(self) -> str:
        return to_sexpr(self.formula)

    @property
    def ast(self) -> int:
        return ast_size(self.formula)

    @property
    def qd(self) -> int:
        return quantifier_depth(self.formula)


def _and(*args: Any) -> Any:
    if len(args) < 2:
        raise ValueError("_and requires at least two arguments")
    result = args[-1]
    for formula in reversed(args[:-1]):
        result = FOAnd(formula, result)
    return result


def _or(*args: Any) -> Any:
    if len(args) < 2:
        raise ValueError("_or requires at least two arguments")
    result = args[-1]
    for formula in reversed(args[:-1]):
        result = FOOr(formula, result)
    return result


def formula_library() -> List[FormulaTemplate]:
    """Return a compact public formula library spanning common INDUCTION patterns."""
    x = Var("x")
    y = Var("y")
    z = Var("z")

    px = Pred("P", [x])
    qx = Pred("Q", [x])
    py = Pred("P", [y])
    qy = Pred("Q", [y])
    pz = Pred("P", [z])
    qz = Pred("Q", [z])
    rxy = Pred("R", [x, y])
    sxy = Pred("S", [x, y])
    ryz = Pred("R", [y, z])
    syz = Pred("S", [y, z])

    return [
        FormulaTemplate("UNARY", "P", "x satisfies P.", px),
        FormulaTemplate("UNARY", "Q_AND_NOT_P", "x satisfies Q and not P.", _and(qx, FONot(px))),
        FormulaTemplate("EXISTS", "R_SUCCESSOR", "x has an R-successor.", Exists(y, "D", rxy)),
        FormulaTemplate(
            "EXISTS",
            "R_SUCCESSOR_P",
            "x has an R-successor satisfying P.",
            Exists(y, "D", _and(rxy, py)),
        ),
        FormulaTemplate(
            "CONJ_EXISTS",
            "Q_AND_R_TO_PQ",
            "x satisfies Q and has an R-successor satisfying P and Q.",
            _and(qx, Exists(y, "D", _and(rxy, py, qy))),
        ),
        FormulaTemplate(
            "FORALL",
            "ALL_R_SUCCESSORS_P",
            "Every R-successor of x satisfies P.",
            Forall(y, "D", _or(FONot(rxy), py)),
        ),
        FormulaTemplate(
            "FORALL",
            "ALL_S_SUCCESSORS_NOT_Q",
            "Every S-successor of x fails Q.",
            Forall(y, "D", _or(FONot(sxy), FONot(qy))),
        ),
        FormulaTemplate(
            "NESTED",
            "R_THEN_S_TO_Q",
            "x has an R-successor that has an S-successor satisfying Q.",
            Exists(y, "D", _and(rxy, Exists(z, "D", _and(syz, qz)))),
        ),
        FormulaTemplate(
            "NESTED_FORALL",
            "ALL_R_HAVE_S_TO_Q",
            "Every R-successor of x has an S-successor satisfying Q.",
            Forall(y, "D", _or(FONot(rxy), Exists(z, "D", _and(syz, qz)))),
        ),
        FormulaTemplate(
            "NESTED_FORALL",
            "ALL_R_TO_EXISTS_R_P",
            "Every R-successor of x has an R-successor satisfying P.",
            Forall(y, "D", _or(FONot(rxy), Exists(z, "D", _and(ryz, pz)))),
        ),
    ]


def _pair(left: str, right: str) -> str:
    return f"({left}, {right})"


def _target_ok(target: Dict[str, List[str]], domain: Sequence[str]) -> bool:
    n_true = len(target.get("T_true", []))
    if not domain:
        return False
    ratio = n_true / len(domain)
    return 0.15 <= ratio <= 0.85


def _predicate_dict(true_atoms: Sequence[str], all_atoms: Sequence[str]) -> Dict[str, List[str]]:
    true_set = set(true_atoms)
    return {
        "true": [atom for atom in all_atoms if atom in true_set],
        "false": [atom for atom in all_atoms if atom not in true_set],
    }


def _random_world(
    rng: random.Random,
    *,
    domain_size: int,
    world_id: str,
    observation_mode: str = "full",
) -> Tuple[Dict[str, Any], FiniteModel]:
    domain = [f"a{i}" for i in range(domain_size)]
    unary_atoms = list(domain)
    binary_atoms = [_pair(left, right) for left in domain for right in domain]

    densities = {
        "P": min(0.8, max(0.15, rng.uniform(0.25, 0.55))),
        "Q": min(0.8, max(0.15, rng.uniform(0.25, 0.55))),
        "R": min(0.6, max(0.08, rng.uniform(0.18, 0.40))),
        "S": min(0.6, max(0.08, rng.uniform(0.18, 0.40))),
    }
    true_p = [atom for atom in unary_atoms if rng.random() < densities["P"]]
    true_q = [atom for atom in unary_atoms if rng.random() < densities["Q"]]
    true_r = [atom for atom in binary_atoms if rng.random() < densities["R"]]
    true_s = [atom for atom in binary_atoms if rng.random() < densities["S"]]

    predicates = {
        "P": _predicate_dict(true_p, unary_atoms),
        "Q": _predicate_dict(true_q, unary_atoms),
        "R": _predicate_dict(true_r, binary_atoms),
        "S": _predicate_dict(true_s, binary_atoms),
    }

    model = FiniteModel(domain_size, const_names=domain)
    for const in true_p:
        model.set_unary("P", model.const_to_index(const), True)
    for const in true_q:
        model.set_unary("Q", model.const_to_index(const), True)
    for pair in true_r:
        left, right = pair.strip("()").split(", ")
        model.set_binary("R", model.const_to_index(left), model.const_to_index(right), True)
    for pair in true_s:
        left, right = pair.strip("()").split(", ")
        model.set_binary("S", model.const_to_index(left), model.const_to_index(right), True)

    world = {
        "observationMode": observation_mode,
        "worldId": world_id,
        "domain": domain,
        "domainSize": domain_size,
        "predicates": predicates,
    }
    return world, model


def _world_with_target(
    rng: random.Random,
    template: FormulaTemplate,
    *,
    world_id: str,
    domain_min: int,
    domain_max: int,
    observation_mode: str = "full",
    max_tries: int = 200,
) -> Tuple[Dict[str, Any], FiniteModel]:
    for _ in range(max_tries):
        domain_size = rng.randint(domain_min, domain_max)
        world, model = _random_world(
            rng,
            domain_size=domain_size,
            world_id=world_id,
            observation_mode=observation_mode,
        )
        target = compute_target_extension(model, template.formula).to_dict()
        world["targetExtension"] = target
        if _target_ok(target, world["domain"]):
            return world, model

    raise RuntimeError(
        f"Could not generate a nontrivial world for {template.family_id}/{template.subfamily_key} "
        f"after {max_tries} attempts"
    )


def _flip_target_for_no_world(
    rng: random.Random,
    target: Dict[str, List[str]],
    domain: Sequence[str],
) -> Dict[str, List[str]]:
    true_set = set(target.get("T_true", []))
    for _ in range(20):
        candidate = rng.choice(list(domain))
        flipped = set(true_set)
        if candidate in flipped:
            flipped.remove(candidate)
        else:
            flipped.add(candidate)
        new_target = {
            "T_true": [elem for elem in domain if elem in flipped],
            "T_false": [elem for elem in domain if elem not in flipped],
        }
        if _target_ok(new_target, domain):
            return new_target

    candidate = rng.choice(list(domain))
    if candidate in true_set:
        true_set.remove(candidate)
    else:
        true_set.add(candidate)
    return {
        "T_true": [elem for elem in domain if elem in true_set],
        "T_false": [elem for elem in domain if elem not in true_set],
    }


def _mask_world_for_ec(
    rng: random.Random,
    full_world: Dict[str, Any],
    *,
    unknown_rate: float,
    unknown_predicates: Sequence[str],
) -> Dict[str, Any]:
    world = {
        key: value
        for key, value in full_world.items()
        if key not in {"predicates", "observationMode"}
    }
    world["observationMode"] = "partial"
    masked_predicates: Dict[str, Dict[str, List[str]]] = {}
    unknown_atoms: Dict[str, List[str]] = {}

    for pred_name, pred_data in full_world["predicates"].items():
        all_atoms = list(pred_data.get("true", [])) + list(pred_data.get("false", []))
        all_atom_set = set(all_atoms)
        unknown = {
            atom
            for atom in all_atoms
            if pred_name in unknown_predicates and rng.random() < unknown_rate
        }
        if unknown:
            unknown_atoms[pred_name] = [atom for atom in all_atoms if atom in unknown]
        masked_predicates[pred_name] = {
            "true": [atom for atom in pred_data.get("true", []) if atom not in unknown],
            "false": [atom for atom in pred_data.get("false", []) if atom not in unknown and atom in all_atom_set],
        }

    if not unknown_atoms and unknown_predicates:
        pred_name = rng.choice(list(unknown_predicates))
        pred_data = masked_predicates[pred_name]
        candidates = pred_data.get("true", []) or pred_data.get("false", [])
        if candidates:
            atom = rng.choice(candidates)
            unknown_atoms[pred_name] = [atom]
            pred_data["true"] = [item for item in pred_data.get("true", []) if item != atom]
            pred_data["false"] = [item for item in pred_data.get("false", []) if item != atom]

    world["predicates"] = masked_predicates
    world["unknownAtoms"] = unknown_atoms
    return world


def _base_problem_description(
    *,
    scenario: str,
    seed: int,
    instance_seed: int,
    template: FormulaTemplate,
    domain_sizes: List[int],
) -> Dict[str, Any]:
    return {
        "scenario": scenario,
        "seed": seed,
        "instance_seed": instance_seed,
        "observationMode": "partial" if scenario == "E" else "full",
        "domainSizes": domain_sizes,
        "hasAxioms": False,
        "numAxioms": 0,
        "hiddenTarget": {
            "formula": template.sexpr,
            "description": template.description,
            "astSize": template.ast,
            "quantifierDepth": template.qd,
        },
        "filtersApplied": ["nontrivial_target_extension"],
        "benchmark_name": "induction_public_generator",
        "benchmark_version": "public_generator_v1",
        "gold_family_id": template.family_id,
        "gold_subfamily_key": template.subfamily_key,
        "gold_is_lift_hard": template.qd >= 2,
        "gold_ast": template.ast,
        "gold_qd": template.qd,
    }


def _record(
    *,
    instance_id: str,
    task: str,
    scenario: str,
    worlds: List[Dict[str, Any]],
    task_spec: Dict[str, Any],
    problem_description: Dict[str, Any],
) -> Dict[str, Any]:
    problem = {
        "instanceId": instance_id,
        "schemaVersion": "fol-concept-synth-v1",
        "scenario": scenario,
        "signature": SIGNATURE,
        "backgroundAxioms": [],
        "worlds": worlds,
        "task": task_spec,
    }
    return {
        "schemaVersion": "induction_benchmark_record_v1",
        "instanceId": instance_id,
        "task": task,
        "scenario": scenario,
        "problemType": "foInduction",
        "problem": problem,
        "problemDescription": problem_description,
    }


def generate_records(
    task: str,
    *,
    n: int,
    seed: int = 0,
    worlds: int = 4,
    yes_worlds: int = 6,
    no_worlds: int = 2,
    domain_min: int = 5,
    domain_max: int = 8,
    unknown_rate: float = 0.35,
) -> List[Dict[str, Any]]:
    """Generate public-schema INDUCTION records for one task."""
    if task not in TASK_TO_SCENARIO:
        raise ValueError(f"Unknown task {task!r}; expected one of {sorted(TASK_TO_SCENARIO)}")
    if n < 0:
        raise ValueError("n must be nonnegative")
    if domain_min < 2 or domain_max < domain_min:
        raise ValueError("invalid domain size range")
    if worlds < 1 or yes_worlds < 1 or no_worlds < 1:
        raise ValueError("world counts must be positive")

    rng = random.Random(seed)
    templates = formula_library()
    scenario = TASK_TO_SCENARIO[task]
    records: List[Dict[str, Any]] = []

    for index in range(n):
        instance_seed = rng.randrange(2**31)
        local_rng = random.Random(instance_seed)
        template = templates[index % len(templates)] if n <= len(templates) else local_rng.choice(templates)
        prefix = {"FullObs": "generated_fullobs", "CI": "generated_ci", "EC": "generated_ec"}[task]
        instance_id = f"{prefix}_{seed}_{index:04d}"

        if task == "FullObs":
            generated_worlds = [
                _world_with_target(
                    local_rng,
                    template,
                    world_id=f"train_{world_idx}",
                    domain_min=domain_min,
                    domain_max=domain_max,
                )[0]
                for world_idx in range(worlds)
            ]
            desc = _base_problem_description(
                scenario=scenario,
                seed=seed,
                instance_seed=instance_seed,
                template=template,
                domain_sizes=[len(world["domain"]) for world in generated_worlds],
            )
            desc.update(
                {
                    "numTrainWorlds": len(generated_worlds),
                    "numTestWorlds": 0,
                    "ad_band": "generated",
                }
            )
            records.append(
                _record(
                    instance_id=instance_id,
                    task=task,
                    scenario=scenario,
                    worlds=generated_worlds,
                    task_spec={
                        "trainWorldIds": [world["worldId"] for world in generated_worlds],
                        "testWorldIds": [],
                    },
                    problem_description=desc,
                )
            )

        elif task == "CI":
            generated_worlds = []
            for world_idx in range(yes_worlds):
                world, _model = _world_with_target(
                    local_rng,
                    template,
                    world_id=f"yes_{world_idx}",
                    domain_min=domain_min,
                    domain_max=domain_max,
                )
                world["splitLabel"] = True
                generated_worlds.append(world)
            for world_idx in range(no_worlds):
                world, _model = _world_with_target(
                    local_rng,
                    template,
                    world_id=f"no_{world_idx}",
                    domain_min=domain_min,
                    domain_max=domain_max,
                )
                world["targetExtension"] = _flip_target_for_no_world(
                    local_rng,
                    world["targetExtension"],
                    world["domain"],
                )
                world["splitLabel"] = False
                generated_worlds.append(world)
            desc = _base_problem_description(
                scenario=scenario,
                seed=seed,
                instance_seed=instance_seed,
                template=template,
                domain_sizes=[len(world["domain"]) for world in generated_worlds],
            )
            desc.update(
                {
                    "numYesWorlds": yes_worlds,
                    "numNoWorlds": no_worlds,
                    "c_band": "generated",
                }
            )
            records.append(
                _record(
                    instance_id=instance_id,
                    task=task,
                    scenario=scenario,
                    worlds=generated_worlds,
                    task_spec={
                        "yesWorldIds": [world["worldId"] for world in generated_worlds if world["splitLabel"] is True],
                        "noWorldIds": [world["worldId"] for world in generated_worlds if world["splitLabel"] is False],
                    },
                    problem_description=desc,
                )
            )

        else:
            generated_worlds = []
            for world_idx in range(worlds):
                full_world, _model = _world_with_target(
                    local_rng,
                    template,
                    world_id=f"train_{world_idx}",
                    domain_min=domain_min,
                    domain_max=domain_max,
                    observation_mode="full",
                )
                generated_worlds.append(
                    _mask_world_for_ec(
                        local_rng,
                        full_world,
                        unknown_rate=unknown_rate,
                        unknown_predicates=("P", "Q", "R", "S"),
                    )
                )
            desc = _base_problem_description(
                scenario=scenario,
                seed=seed,
                instance_seed=instance_seed,
                template=template,
                domain_sizes=[len(world["domain"]) for world in generated_worlds],
            )
            desc.update(
                {
                    "numTrainWorlds": len(generated_worlds),
                    "numTestWorlds": 0,
                    "e_semantics_used": "exact_exists",
                    "e_band": "generated",
                    "e_unknown_rate_target": unknown_rate,
                    "e_unknown_predicates": ["P", "Q", "R", "S"],
                }
            )
            records.append(
                _record(
                    instance_id=instance_id,
                    task=task,
                    scenario=scenario,
                    worlds=generated_worlds,
                    task_spec={
                        "trainWorldIds": [world["worldId"] for world in generated_worlds],
                        "testWorldIds": [],
                    },
                    problem_description=desc,
                )
            )

    return records


def generate_task_bundle(
    *,
    task: str,
    n: int,
    seed: int = 0,
    worlds: int = 4,
    yes_worlds: int = 6,
    no_worlds: int = 2,
    domain_min: int = 5,
    domain_max: int = 8,
    unknown_rate: float = 0.35,
) -> List[Dict[str, Any]]:
    """Generate records for one task or for all tasks."""
    if task == "all":
        records: List[Dict[str, Any]] = []
        for offset, task_name in enumerate(("FullObs", "CI", "EC")):
            records.extend(
                generate_records(
                    task_name,
                    n=n,
                    seed=seed + offset,
                    worlds=worlds,
                    yes_worlds=yes_worlds,
                    no_worlds=no_worlds,
                    domain_min=domain_min,
                    domain_max=domain_max,
                    unknown_rate=unknown_rate,
                )
            )
        return records
    return generate_records(
        task,
        n=n,
        seed=seed,
        worlds=worlds,
        yes_worlds=yes_worlds,
        no_worlds=no_worlds,
        domain_min=domain_min,
        domain_max=domain_max,
        unknown_rate=unknown_rate,
    )


def validate_generated_records(records: Iterable[Dict[str, Any]], *, timeout_ms: int = 30000) -> List[str]:
    """Validate that each generated record is solved by its planted formula."""
    errors: List[str] = []
    for record in records:
        formula = record["problemDescription"]["hiddenTarget"]["formula"]
        result = evaluate_prediction(
            record,
            {
                "instanceId": record["instanceId"],
                "model": "gold",
                "extractedFormula": formula,
                "success": True,
            },
            run_id="generator_validation",
            timeout_ms=timeout_ms,
            compute_diagnostics=False,
        )
        if not result["parse_ok"] or not result["valid"]:
            errors.append(
                f"{record['instanceId']}: parse_ok={result['parse_ok']} valid={result['valid']} "
                f"reason={result['evaluation'].get('failureExplanation')}"
            )
    return errors


def write_records(records: Sequence[Dict[str, Any]], output_path: str | Path) -> None:
    """Write generated records to YAML or YAML.GZ."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.name.endswith(".gz"):
        with gzip.open(output, "wt", encoding="utf-8") as handle:
            yaml.safe_dump(list(records), handle, sort_keys=False, allow_unicode=True)
    else:
        with output.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(list(records), handle, sort_keys=False, allow_unicode=True)
