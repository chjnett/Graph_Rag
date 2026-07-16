"""scripts/throughput_pilot.py 유닛테스트 — TODO.md Milestone 1.

실제 vLLM 엔드포인트/GPU/UltraDomain 코퍼스 없이도 순수 로직(샘플링, 리포트 저장,
corpus_scope 의사결정)은 검증 가능하다. 네트워크·인덱싱 호출부는 mock/에러 경로만 확인한다.
"""

import json

import pytest

from scripts.throughput_pilot import (
    DocTiming,
    ThroughputReport,
    _write_report,
    check_endpoint,
    decide_corpus_scope,
    index_one_doc,
    sample_docs,
)


# ---- sample_docs ----

def test_sample_docs_raises_when_not_enough_files(tmp_path):
    (tmp_path / "doc1.txt").write_text("a", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        sample_docs(tmp_path, n=3)


def test_sample_docs_returns_requested_count(tmp_path):
    for i in range(5):
        (tmp_path / f"doc{i}.txt").write_text("a", encoding="utf-8")
    docs = sample_docs(tmp_path, n=3)
    assert len(docs) == 3
    assert len(set(docs)) == 3  # 중복 없이 샘플링


def test_sample_docs_deterministic_with_same_seed(tmp_path):
    for i in range(10):
        (tmp_path / f"doc{i}.txt").write_text("a", encoding="utf-8")
    first = sample_docs(tmp_path, n=4, seed=42)
    second = sample_docs(tmp_path, n=4, seed=42)
    assert first == second


# ---- check_endpoint ----

def test_check_endpoint_returns_false_on_connection_error(monkeypatch):
    import requests

    def _raise(*args, **kwargs):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "get", _raise)
    assert check_endpoint("http://localhost:8000/v1") is False


def test_check_endpoint_returns_true_when_ok(monkeypatch):
    class _FakeResp:
        ok = True

    monkeypatch.setattr("requests.get", lambda *a, **k: _FakeResp())
    assert check_endpoint("http://localhost:8000/v1") is True


# ---- index_one_doc ----

def test_index_one_doc_unknown_method_raises_value_error(tmp_path):
    doc = tmp_path / "doc.txt"
    doc.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        index_one_doc("unknown_method", doc, "http://localhost:8000/v1", "some-model")


def test_index_one_doc_raises_runtime_error_when_package_missing(tmp_path):
    doc = tmp_path / "doc.txt"
    doc.write_text("x", encoding="utf-8")
    with pytest.raises(RuntimeError):
        index_one_doc("ms_graphrag", doc, "http://localhost:8000/v1", "some-model")


# ---- _write_report ----

def test_write_report_creates_json_file(tmp_path):
    report = ThroughputReport(
        method="ms_graphrag",
        stage=2,
        doc_count=2,
        doc_timings=[DocTiming(doc_id="d1", wall_clock_sec=10.0, llm_calls=5)],
        avg_sec_per_doc=10.0,
        avg_llm_calls_per_doc=5.0,
        extrapolated_full_corpus_days=0.05,
    )
    out_path = _write_report(report, reports_dir=tmp_path)

    assert out_path == tmp_path / "throughput_pilot_ms_graphrag_stage2.json"
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["method"] == "ms_graphrag"
    assert saved["doc_timings"][0]["doc_id"] == "d1"


# ---- decide_corpus_scope ----

def _write_stage2_report(reports_dir, method: str, extrapolated_days: float):
    reports_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": method,
        "stage": 2,
        "doc_count": 20,
        "doc_timings": [],
        "avg_sec_per_doc": 1.0,
        "avg_llm_calls_per_doc": 1.0,
        "extrapolated_full_corpus_days": extrapolated_days,
    }
    (reports_dir / f"throughput_pilot_{method}_stage2.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _base_config():
    return {"throughput_pilot": {"extrapolation_threshold_days": 14}}


def test_decide_corpus_scope_raises_when_report_missing(tmp_path):
    with pytest.raises(SystemExit):
        decide_corpus_scope(
            _base_config(), reports_dir=tmp_path, eval_config_path=tmp_path / "eval.yaml"
        )


def test_decide_corpus_scope_picks_subset_when_over_threshold(tmp_path):
    _write_stage2_report(tmp_path, "ms_graphrag", extrapolated_days=20.0)
    _write_stage2_report(tmp_path, "lightrag", extrapolated_days=5.0)
    eval_config_path = tmp_path / "eval.yaml"

    scope = decide_corpus_scope(_base_config(), reports_dir=tmp_path, eval_config_path=eval_config_path)

    assert scope == "subset"


def test_decide_corpus_scope_picks_full_when_under_threshold(tmp_path):
    _write_stage2_report(tmp_path, "ms_graphrag", extrapolated_days=3.0)
    _write_stage2_report(tmp_path, "lightrag", extrapolated_days=5.0)
    eval_config_path = tmp_path / "eval.yaml"

    scope = decide_corpus_scope(_base_config(), reports_dir=tmp_path, eval_config_path=eval_config_path)

    assert scope == "full"


def test_decide_corpus_scope_persists_decision_to_eval_config(tmp_path):
    import yaml

    _write_stage2_report(tmp_path, "ms_graphrag", extrapolated_days=20.0)
    _write_stage2_report(tmp_path, "lightrag", extrapolated_days=5.0)
    eval_config_path = tmp_path / "eval.yaml"
    config = _base_config()
    config["corpus_scope"] = "subset"
    config["corpus_scope_decided_at"] = None

    decide_corpus_scope(config, reports_dir=tmp_path, eval_config_path=eval_config_path)

    saved = yaml.safe_load(eval_config_path.read_text(encoding="utf-8"))
    assert saved["corpus_scope"] == "subset"
    assert saved["corpus_scope_decided_at"] is not None
