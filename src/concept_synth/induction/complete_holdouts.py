"""Deterministic, model-blind completion of the frozen Challenge holdouts.

No model clients, predictions, or evaluation caches are read by this module.
Holdouts remain separate from the solver-facing benchmark files.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import random
from pathlib import Path

import yaml

from concept_synth.fol.model import FiniteModel
from concept_synth.sexpr_parser import parse_sexpr_formula
from concept_synth.target import compute_target_extension
from .evaluator import build_model_from_world, match_world


def stable(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def read_dataset(path):
    with gzip.open(path, "rt") as handle:
        return yaml.safe_load(handle)


def structure(world):
    model = build_model_from_world(world)
    return stable({"domain": model.domain_constants(),
                   "unary": {k: sorted(v) for k, v in model.unary.items() if v},
                   "binary": {k: sorted(v) for k, v in model.binary.items() if v}})


def draw(rng, n, stage, policy):
    unary = {k: set(rng.sample(range(n), rng.randint(max(1, int(n*.15)), min(n-1, int(n*.85)))))
             for k in ("P", "Q")}
    if stage == "regular":
        binary = {k: {(i, j) for i in range(n) for j in rng.sample(range(n), min(2, n))}
                  for k in ("R", "S")}
    else:
        probabilities = {k: rng.choice(policy["binary_probability_grid"]) for k in ("R", "S")}
        binary = {k: {(i, j) for i in range(n) for j in range(n) if rng.random() < probabilities[k]}
                  for k in ("R", "S")}
    return FiniteModel(n, unary=unary, binary=binary)


def world_dict(model, truth, world_id):
    return {"worldId": world_id, "domain": model.domain_constants(), "domainSize": model.n,
            "observationMode": "full",
            "predicates": {**{k: {"true": [model.index_to_const(i) for i in sorted(model.unary[k])]}
                              for k in ("P", "Q")},
                           **{k: {"true": [f"({model.index_to_const(i)}, {model.index_to_const(j)})"
                                            for i, j in sorted(model.binary[k])]}
                              for k in ("R", "S")}},
            "targetExtension": {"T_true": sorted(truth)}}


def complete_task(record, legacy, policy):
    tid = record["instanceId"]
    assert legacy["task_id"] == tid
    worlds = copy.deepcopy(legacy["worlds"])
    target_count = policy["worlds_per_task"]
    if len(worlds) > target_count:
        raise ValueError(f"Too many legacy worlds: {tid}")
    train = record["problem"]["worlds"]
    seen = {structure(w) for w in train + worlds}
    gold = parse_sexpr_formula(record["problemDescription"]["hiddenTarget"]["formula"])
    sizes = sorted({len(w["domain"]) for w in train})
    assert sizes and min(sizes) >= 2
    trace = []
    counts = {}
    lo, hi = policy["target_positive_fraction"]
    for stage in ("regular", "fallback"):
        counts[stage] = {"proposals": 0, "balance_rejected": 0, "duplicate_rejected": 0, "accepted": 0}
        seed_text = policy["seed_namespace"] + ":" + tid + ":" + stage
        seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16)
        rng = random.Random(seed)
        budget = policy[stage + "_budget_per_incomplete_task"]
        if stage == "regular" and tid in policy["structural_regular_skip"]:
            budget = 0
        for attempt in range(1, budget + 1):
            if len(worlds) == target_count:
                break
            counts[stage]["proposals"] += 1
            model = draw(rng, rng.choice(sizes), stage, policy)
            truth = compute_target_extension(model, gold).T_true
            if not lo <= len(truth)/model.n <= hi:
                counts[stage]["balance_rejected"] += 1
                continue
            suffix = 1
            used_ids = {w['worldId'] for w in worlds}
            while f"holdout_v2_added_{suffix}" in used_ids:
                suffix += 1
            world = world_dict(model, truth, f"holdout_v2_added_{suffix}")
            key = structure(world)
            if key in seen:
                counts[stage]["duplicate_rejected"] += 1
                continue
            seen.add(key)
            worlds.append(world)
            counts[stage]["accepted"] += 1
            trace.append({"world_id": world["worldId"], "stage": stage, "proposal": attempt,
                          "seed": seed, "world_sha256": hashlib.sha256(stable(world).encode()).hexdigest()})
    if len(worlds) != target_count:
        raise RuntimeError(f"Incomplete holdouts for {tid}: {len(worlds)}/{target_count}; no release written")
    assert worlds[:len(legacy["worlds"])] == legacy["worlds"]
    return {"task_id": tid, "requested_worlds": target_count, "generated_worlds": len(worlds),
            "worlds": worlds}, {"task_id": tid, "preserved": len(legacy["worlds"]),
                                "added": len(trace), "sampling": counts, "accepted": trace}


def validate(records, legacy_rows, completed):
    assert len(records) == len(legacy_rows) == len(completed)
    old = {r["task_id"]: r for r in legacy_rows}
    assert len(old) == len(records)
    total = added = 0
    for record, row in zip(records, completed):
        tid = record["instanceId"]
        assert row["task_id"] == tid and len(row["worlds"]) == 5
        previous = old[tid]["worlds"]
        assert row["worlds"][:len(previous)] == previous
        assert len({w["worldId"] for w in row["worlds"]}) == 5
        train_keys = {structure(w) for w in record["problem"]["worlds"]}
        holdout_keys = [structure(w) for w in row["worlds"]]
        assert len(set(holdout_keys)) == 5 and not train_keys.intersection(holdout_keys)
        gold = parse_sexpr_formula(record["problemDescription"]["hiddenTarget"]["formula"])
        for world in row["worlds"]:
            assert world["domainSize"] == len(world["domain"])
            assert match_world(world, gold)[0], (tid, world["worldId"])
            assert .15 <= len(world["targetExtension"]["T_true"])/len(world["domain"]) <= .85
        total += 5
        added += 5-len(previous)
    return {"tasks": len(records), "worlds": total, "added": added,
            "preserved": total-added, "all_gold_labels_exact": True,
            "training_overlap": 0, "duplicate_holdouts": 0, "model_api_calls": 0}


def build(dataset, projection, legacy_path, policy_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    records, c64 = read_dataset(dataset), read_dataset(projection)
    assert len(records) == 100 and len(c64) == 64 and records[:64] == c64
    legacy = read_jsonl(legacy_path)
    lookup = {r["task_id"]: r for r in legacy}
    assert len(lookup) == 100 and set(lookup) == {r["instanceId"] for r in records}
    policy = json.loads(Path(policy_path).read_text())
    sources = {p.name: sha(p) for p in (dataset, projection, legacy_path, policy_path)}
    completed, traces = [], []
    for record in records:
        row, trace = complete_task(record, lookup[record["instanceId"]], policy)
        completed.append(row); traces.append(trace)
        if trace["added"]:
            print(json.dumps({k: trace[k] for k in ("task_id", "preserved", "added", "sampling")}), flush=True)
    checks = validate(records, legacy, completed)
    assert checks["worlds"] == 500 and checks["preserved"] == 449 and checks["added"] == 51
    assert all(sha(p) == sources[p.name] for p in (dataset, projection, legacy_path, policy_path))
    output.mkdir(parents=True)
    # Identical solver inputs, independently versioned bundle paths. Do not embed holdouts in prompts.
    artifacts = {
        "induction_fullobs_challenge100_v2.yaml.gz": Path(dataset).read_bytes(),
        "induction_fullobs_challenge64_v2.yaml.gz": Path(projection).read_bytes(),
        "challenge100_holdouts_v2.jsonl": ("\n".join(stable(r) for r in completed)+"\n").encode(),
        "challenge64_holdouts_v2.jsonl": ("\n".join(stable(r) for r in completed[:64])+"\n").encode(),
        "generation_trace.json": (json.dumps(traces, indent=2, sort_keys=True)+"\n").encode(),
        "validation.json": (json.dumps(checks, indent=2, sort_keys=True)+"\n").encode(),
        "policy.json": Path(policy_path).read_bytes(),
    }
    for name, data in artifacts.items():
        with (output/name).open("xb") as handle:
            handle.write(data)
    manifest = {"version": policy["version"], "sources": sources,
                "generator_sha256": sha(__file__), "validation": checks,
                "training_inputs_byte_identical": True, "challenge64_exact_projection": True,
                "holdouts_are_solver_inputs": False, "reevaluation_started": False,
                "artifacts": {name: {"sha256": sha(output/name), "bytes": len(data)}
                              for name, data in artifacts.items()}}
    with (output/"manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True); handle.write("\n")
    print(json.dumps(checks), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("dataset", "projection", "legacy", "policy", "output"):
        parser.add_argument("--"+arg, type=Path, required=True)
    args = parser.parse_args()
    build(args.dataset, args.projection, args.legacy, args.policy, args.output)


if __name__ == "__main__":
    main()
