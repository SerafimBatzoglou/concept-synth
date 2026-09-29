import gzip
import json
from pathlib import Path


BENCH = Path(__file__).resolve().parents[1] / "benchmarks/induction"


def test_sonnet55_final_stored_scores_and_projection():
    report = json.loads((BENCH / "eval/claude_sonnet_5_5_challenge100_round1_report.json").read_text())
    assert report["model"] == "claude-sonnet-5-5"
    assert report["publication_status"] == "final_four_verified_physical_passes"
    with gzip.open(BENCH / "data/induction_fullobs_challenge100_v1.yaml.gz", "rt") as f:
        import yaml
        ids = [r["instanceId"] for r in yaml.safe_load(f)]
    assert [t["task_id"] for t in report["tasks"]] == ids
    assert len(set(ids)) == 100
    assert report["frontiers"] == {"post_round1": 100, "symbolic_candidates": 0}
    assert (report["overall"]["evaluable"], report["overall"]["correct"]) == (100, 44)
    assert (report["overall"]["strict_as_submitted_evaluable"], report["overall"]["strict_as_submitted_correct"]) == (100, 44)
    assert report["overall"]["parser_normalized_responses"] == 0
    assert (report["overall"]["holdout"]["correct"], report["overall"]["holdout"]["available"]) == (37, 42)
    assert len(report["non_evaluable_task_ids"]) == 0
    assert [c["generation_calls"]["submitted_requests"] for c in report["components"]] == [100, 69, 57, 3]
    assert [c["overall"]["evaluable"] for c in report["components"]] == [31, 12, 54, 3]
    assert [c["overall"]["correct"] for c in report["components"]] == [31, 10, 3, 0]
    assert [c["provider_terminal"]["refusals"] for c in report["components"]] == [0, 0, 0, 0]
    assert [c["provider_terminal"]["output_cap"] for c in report["components"]] == [69, 57, 3, 0]
    calls = report["generation_calls"]
    assert calls["submitted_requests"] == calls["attempted_confirmed"] == calls["accepted_confirmed"] == calls["terminal_records"] == 229
    assert calls["batch_submissions"] == 4 and calls["retries"] == calls["acceptance_unknown"] == calls["attempted_unknown"] == 0
    assert report["settings"]["effort_sequence"] == ["max", "xhigh", "high", "medium"]
    assert report["settings"]["max_tokens"] == 300000
    assert report["all_physical_response_token_usage"]["total"]["sum"] == 57586546
    assert report["overall"]["token_usage"]["total"]["sum"] == 17997746
    assert report["overall"]["token_usage"]["reasoning"]["count"] == 0
    assert sum(v["selected_evaluable"] for v in report["selected_component_counts"].values()) == 100
    assert sum(v["selected"] for v in report["selected_component_counts"].values()) == 100
    assert len({t["category_or_slice"] for t in report["tasks"] if t["subset"] == "new36"}) == 10
    rows = {r["instance_id"]: r for r in map(json.loads, (BENCH / "eval/induction_challenge64_round1_eval_cache_v1.jsonl").read_text().splitlines()) if r["model_id"] == report["model"]}
    assert len(rows) == 64
    assert sum(r["parse_ok"] for r in rows.values()) == 64
    assert sum(r["valid"] for r in rows.values()) == 32
    for t in report["tasks"][:64]:
        assert rows[t["task_id"]]["parse_ok"] == t["evaluable"]
        assert rows[t["task_id"]]["valid"] == t["correct"]
    for t in report["tasks"]:
        assert bool(t["formula"]) == t["evaluable"]
    serialized = json.dumps(report)
    for forbidden in ["/Users/", "pipeline.sqlite", "reasoningTrace", "provider_call_meta", "candidate_id", "ANTHROPIC_API_KEY", "msgbatch_"]:
        assert forbidden not in serialized

    assert report["monetary_cost"] is None
    assert [c["overall"]["token_usage"]["total"]["sum"] for c in report["components"]] == [25453965, 20288395, 11710907, 133279]
    assert report["all_physical_response_token_usage"]["reasoning"]["sum"] is None
    assert report["final_combined_report_sha256"] == "49b0ecdf12aa834941b6f6b44b0b540588537b7cbc6b298765441b047d2ed67f"
    assert report["campaign_completion_sha256"] == "0b1db04e3c5b8abcb17488d52ab6d79f1d727d0660ee80175aab269b0bbf7070"
    holds = [json.loads(line) for line in (BENCH / "eval/induction_challenge64_round1_holdout_eval_cache_v1.jsonl").read_text().splitlines()]
    holds = [r for r in holds if r["model_id"] == report["model"]]
    assert len(holds) == 64
    assert sum(r["completed"] for r in holds) == 31
    assert sum(r["valid"] for r in holds) == 26
    with gzip.open(BENCH / "predictions/induction_challenge64_round1_predictions_v1.jsonl.gz", "rt") as f:
        preds = [json.loads(line) for line in f]
    preds = [r for r in preds if r.get("model_id", r.get("model")) == report["model"]]
    assert len(preds) == 64
    for row in preds:
        assert row.get("rawResponse") is None and row.get("thinking") is None
