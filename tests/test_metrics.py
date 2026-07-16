"""src/eval/metrics_*.py 유닛테스트 — 목(mock) QAResult/IndexStats 기반.
TODO.md Milestone 3: "서브1/서브3 실제 데이터 없이도 통과해야 함."
"""

import json

import pytest

from src.eval.interface import Evidence, IndexStats, QAResult
from src.eval.metrics_cost import aggregate_ours_gpu_hours, compare_costs, compare_total_costs
from src.eval.metrics_coverage import load_agreement_report
from src.eval.metrics_gold_accuracy import load_gold_accuracy_report
from src.eval.metrics_hallucination import hallucination_rate
from src.eval.metrics_qa import compute_em_f1, compute_llm_as_judge, merge_with_anchor


# ---- metrics_cost ----

def test_compare_costs_sorts_by_gpu_hours_and_includes_all_fields():
    stats = {
        "ms_graphrag": IndexStats(wall_clock_sec=3600, gpu_hours=5.0, llm_calls=1000, node_count=10, edge_count=20),
        "ours": IndexStats(wall_clock_sec=600, gpu_hours=0.2, llm_calls=0, node_count=8, edge_count=15),
    }
    rows = compare_costs(stats)
    assert [r["method"] for r in rows] == ["ours", "ms_graphrag"]
    assert rows[0]["llm_calls"] == 0


def test_compare_costs_optional_api_cost_estimate():
    stats = {"ours": IndexStats(wall_clock_sec=600, gpu_hours=2.0, llm_calls=0)}
    rows = compare_costs(stats, api_cost_per_gpu_hour=1.5)
    assert rows[0]["api_cost_estimate"] == pytest.approx(3.0)


def test_compare_costs_empty_stats_returns_empty_list():
    assert compare_costs({}) == []


def test_aggregate_ours_gpu_hours_sums_stages():
    stage_hours = {"data_gen": 1.5, "distill": 3.0, "graph_construction": 0.2}
    assert aggregate_ours_gpu_hours(stage_hours) == pytest.approx(4.7)


def test_aggregate_ours_gpu_hours_empty_stages_is_zero():
    assert aggregate_ours_gpu_hours({}) == 0.0


def test_compare_total_costs_sorts_ours_against_baselines():
    stage_hours = {"data_gen": 1.0, "distill": 2.0, "graph_construction": 0.5}
    baseline_stats = {
        "ms_graphrag": IndexStats(wall_clock_sec=3600, gpu_hours=10.0, llm_calls=1000),
        "litesemrag": IndexStats(wall_clock_sec=60, gpu_hours=0.01, llm_calls=0),
    }
    rows = compare_total_costs(stage_hours, baseline_stats)

    assert [r["method"] for r in rows] == ["litesemrag", "ours", "ms_graphrag"]
    ours_row = next(r for r in rows if r["method"] == "ours")
    assert ours_row["gpu_hours"] == pytest.approx(3.5)
    assert ours_row["stage_breakdown"] == stage_hours
    baseline_row = next(r for r in rows if r["method"] == "ms_graphrag")
    assert baseline_row["stage_breakdown"] is None


# ---- metrics_coverage / metrics_gold_accuracy ----

@pytest.fixture
def sub3_report_path(tmp_path):
    report = {
        "agreement": {"teacher_student_agreement": 0.74},
        "gold_accuracy": {"precision": 0.71, "recall": 0.66},
    }
    p = tmp_path / "sub3_phase3_7.json"
    p.write_text(json.dumps(report), encoding="utf-8")
    return p


def test_load_agreement_report(sub3_report_path):
    agreement = load_agreement_report(sub3_report_path)
    assert agreement == {"teacher_student_agreement": 0.74}


def test_load_gold_accuracy_report(sub3_report_path):
    gold_accuracy = load_gold_accuracy_report(sub3_report_path)
    assert gold_accuracy == {"precision": 0.71, "recall": 0.66}


def test_load_agreement_report_rejects_wrong_schema(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"unrelated": 1}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_agreement_report(p)


# ---- metrics_qa ----

def _mock_result(answer: str) -> QAResult:
    return QAResult(answer=answer, evidence=[], latency_ms=10.0, indexing_llm_calls=0)


def test_compute_em_f1_perfect_match():
    predictions = [_mock_result("Paris"), _mock_result("the Eiffel Tower")]
    references = [["Paris"], ["Eiffel Tower"]]
    metrics = compute_em_f1(predictions, references)
    assert metrics["em"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["n"] == 2


def test_compute_em_f1_partial_overlap():
    predictions = [_mock_result("Paris is the capital")]
    references = [["Paris is the capital of France"]]
    metrics = compute_em_f1(predictions, references)
    assert metrics["em"] == 0.0
    assert 0.0 < metrics["f1"] < 1.0


def test_compute_em_f1_length_mismatch_raises():
    with pytest.raises(ValueError):
        compute_em_f1([_mock_result("a")], [])


def test_compute_em_f1_empty_lists_returns_zero_n():
    metrics = compute_em_f1([], [])
    assert metrics == {"em": 0.0, "f1": 0.0, "n": 0}


def test_compute_llm_as_judge_empty_lists_returns_zero_n():
    result = compute_llm_as_judge([], [], judge_fn=lambda pred, refs: 1.0)
    assert result == {"llm_judge_score": 0.0, "n": 0}


def test_compute_llm_as_judge_uses_injected_judge_fn():
    predictions = [_mock_result("Paris")]
    references = [["Paris"]]
    judge_fn = lambda pred, refs: 1.0 if pred in refs else 0.0
    result = compute_llm_as_judge(predictions, references, judge_fn)
    assert result == {"llm_judge_score": 1.0, "n": 1}


def test_merge_with_anchor():
    merged = merge_with_anchor({"em": 0.5}, {"em": 0.6})
    assert merged == {"em": 0.5, "original_paper_anchor": {"em": 0.6}}

    merged_null = merge_with_anchor({"em": 0.5}, None)
    assert merged_null["original_paper_anchor"] is None


# ---- metrics_hallucination ----

def test_hallucination_rate_all_grounded():
    source_documents = {"doc1": "Steve Jobs founded Apple in Cupertino."}
    results = [
        QAResult(
            answer="Apple",
            evidence=[Evidence(source_span="Steve Jobs founded Apple", doc_id="doc1")],
            latency_ms=5.0,
            indexing_llm_calls=0,
        )
    ]
    assert hallucination_rate(results, source_documents) == 0.0


def test_hallucination_rate_detects_ungrounded_span():
    source_documents = {"doc1": "Steve Jobs founded Apple in Cupertino."}
    results = [
        QAResult(
            answer="Apple",
            evidence=[
                Evidence(source_span="Steve Jobs founded Apple", doc_id="doc1"),
                Evidence(source_span="totally fabricated sentence", doc_id="doc1"),
            ],
            latency_ms=5.0,
            indexing_llm_calls=0,
        )
    ]
    assert hallucination_rate(results, source_documents) == pytest.approx(0.5)


def test_hallucination_rate_no_evidence_returns_zero():
    results = [QAResult(answer="", evidence=[], latency_ms=1.0, indexing_llm_calls=0)]
    assert hallucination_rate(results, {}) == 0.0


def test_hallucination_rate_empty_results_list_returns_zero():
    assert hallucination_rate([], {}) == 0.0
