"""src/eval/benchmark.py 유닛테스트 — TODO.md Milestone 4."""

import pytest

from src.eval.benchmark import sequential_run, serialize_result_row
from src.eval.interface import GraphRAGMethod, IndexStats, QAResult


class _RecordingMethod(GraphRAGMethod):
    """index() 호출을 순서/시점과 함께 공유 리스트에 기록하는 mock — 직렬 실행 검증용."""

    def __init__(self, name: str, call_log: list[str]):
        self.name = name
        self.call_log = call_log

    def index(self, corpus_path: str, scope: str) -> IndexStats:
        self.call_log.append(f"{self.name}:start")
        self.call_log.append(f"{self.name}:end")
        return IndexStats(wall_clock_sec=1.0, gpu_hours=0.01, llm_calls=0)

    def query(self, question: str, top_k: int = 5) -> QAResult:
        return QAResult(answer="mock", evidence=[], latency_ms=1.0, indexing_llm_calls=0)


def test_sequential_run_processes_methods_one_at_a_time_in_order():
    call_log: list[str] = []
    methods = {
        "a": _RecordingMethod("a", call_log),
        "b": _RecordingMethod("b", call_log),
    }
    results = sequential_run(methods, corpus_path="dummy", scope="subset")

    assert list(results.keys()) == ["a", "b"]
    # a의 start/end가 b의 start보다 먼저 — 동시 실행이 아니라 완전 직렬임을 증명
    assert call_log == ["a:start", "a:end", "b:start", "b:end"]


def _base_row_kwargs(**overrides):
    kwargs = dict(
        method="ms_graphrag",
        dataset="graphrag_bench",
        difficulty="multi_hop",
        em=0.5,
        f1=0.6,
        ci95=(0.4, 0.6),
        teacher_student_agreement=None,
        gold_accuracy=None,
        hallucination_rate=0.1,
        gpu_hours=5.0,
        indexing_llm_calls=1000,
        original_paper_anchor={"em": 0.55},
        corpus_scope="subset",
    )
    kwargs.update(overrides)
    return kwargs


def test_serialize_result_row_baseline_matches_spec_schema():
    row = serialize_result_row(**_base_row_kwargs())
    assert row["gold_accuracy"] is None
    assert row["original_paper_anchor"] == {"em": 0.55}
    assert row["ci95"] == [0.4, 0.6]


def test_serialize_result_row_ours_can_have_gold_accuracy_and_null_anchor():
    row = serialize_result_row(
        **_base_row_kwargs(
            method="ours",
            indexing_llm_calls=0,
            gold_accuracy={"precision": 0.71, "recall": 0.66},
            original_paper_anchor=None,
        )
    )
    assert row["gold_accuracy"] == {"precision": 0.71, "recall": 0.66}
    assert row["original_paper_anchor"] is None


def test_serialize_result_row_rejects_baseline_with_gold_accuracy():
    with pytest.raises(AssertionError):
        serialize_result_row(
            **_base_row_kwargs(gold_accuracy={"precision": 0.9, "recall": 0.9})
        )


@pytest.mark.parametrize("method", ["ours", "litesemrag", "nollmrag"])
def test_serialize_result_row_llm_free_guard_rejects_nonzero_calls(method):
    with pytest.raises(AssertionError):
        serialize_result_row(
            **_base_row_kwargs(method=method, indexing_llm_calls=3, original_paper_anchor=None)
        )


def test_serialize_result_row_llm_free_zero_calls_ok():
    row = serialize_result_row(
        **_base_row_kwargs(method="litesemrag", indexing_llm_calls=0, original_paper_anchor=None)
    )
    assert row["indexing_llm_calls"] == 0
