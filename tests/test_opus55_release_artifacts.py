import gzip
import json
from pathlib import Path


BENCH = Path(__file__).resolve().parents[1] / "benchmarks/induction"


def test_opus55_final_stored_scores_and_projection():
    report = json.loads((BENCH / "eval/claude_opus_5_5_challenge100_round1_report.json").read_text())
    assert report["model"] == "claude-opus-5-5"
    assert report["publication_status"] == "final_four_verified_physical_passes"
    with gzip.open(BENCH / "data/induction_fullobs_challenge100_v1.yaml.gz", "rt") as f:
        import yaml
        ids = [r["instanceId"] for r in yaml.safe_load(f)]
    assert [t["task_id"] for t in report["tasks"]] == ids
    assert len(set(ids)) == 100
    assert report["frontiers"] == {"post_round1": 100, "symbolic_candidates": 0}
    assert (report["overall"]["evaluable"], report["overall"]["correct"]) == (91, 65)
    assert (report["overall"]["strict_as_submitted_evaluable"], report["overall"]["strict_as_submitted_correct"]) == (90, 64)
    assert report["overall"]["parser_normalized_responses"] == 1
    assert (report["overall"]["holdout"]["correct"], report["overall"]["holdout"]["available"]) == (50, 60)
    assert len(report["non_evaluable_task_ids"]) == 9
    assert [c["generation_calls"]["submitted_requests"] for c in report["components"]] == [100, 64, 49, 31]
    assert [c["overall"]["evaluable"] for c in report["components"]] == [36, 15, 18, 22]
    assert [c["overall"]["correct"] for c in report["components"]] == [36, 15, 10, 4]
    assert [c["provider_terminal"]["refusals"] for c in report["components"]] == [31, 18, 8, 9]
    assert [c["provider_terminal"]["output_cap"] for c in report["components"]] == [33, 31, 23, 0]
    calls = report["generation_calls"]
    assert calls["submitted_requests"] == calls["attempted_confirmed"] == calls["accepted_confirmed"] == calls["terminal_records"] == 244
    assert calls["batch_submissions"] == 4 and calls["retries"] == calls["acceptance_unknown"] == calls["attempted_unknown"] == 0
    assert report["settings"]["effort_sequence"] == ["max", "xhigh", "high", "medium"]
    assert report["settings"]["max_tokens"] == 300000
    assert report["all_physical_response_token_usage"]["total"]["sum"] == 52259750
    assert report["overall"]["token_usage"]["total"]["sum"] == 18718136
    assert report["overall"]["token_usage"]["reasoning"]["count"] == 0
    assert sum(v["selected_evaluable"] for v in report["selected_component_counts"].values()) == 91
    assert sum(v["selected"] for v in report["selected_component_counts"].values()) == 100
    assert len({t["category_or_slice"] for t in report["tasks"] if t["subset"] == "new36"}) == 10
    rows = {r["instance_id"]: r for r in map(json.loads, (BENCH / "eval/induction_challenge64_round1_eval_cache_v1.jsonl").read_text().splitlines()) if r["model_id"] == report["model"]}
    assert len(rows) == 64
    assert sum(r["parse_ok"] for r in rows.values()) == 59
    assert sum(r["valid"] for r in rows.values()) == 46
    for t in report["tasks"][:64]:
        assert rows[t["task_id"]]["parse_ok"] == t["evaluable"]
        assert rows[t["task_id"]]["valid"] == t["correct"]
    for t in report["tasks"]:
        assert bool(t["formula"]) == t["evaluable"]
    serialized = json.dumps(report)
    for forbidden in ["/Users/", "pipeline.sqlite", "reasoningTrace", "provider_call_meta", "candidate_id", "ANTHROPIC_API_KEY", "msgbatch_"]:
        assert forbidden not in serialized
