"""통합 러너 — spec.md §5/§6/§8 (TODO.md Milestone 4).

`configs/eval.yaml`의 `methods`를 순차 실행(동시 실행 금지 — spec.md §2 "단일 GPU 제약")하고
결과를 spec.md §8 JSON 스키마로 직렬화한다.

🔒 실제 end-to-end 실행(전체 QA 벤치마크, Phase 4.1~4.3)은 Milestone 1~3 완료 +
`graphrag_03` Phase 3.35(검색 인터페이스)/Phase 3.7(일치율·실정확도) 산출물이 있어야 가능하다.
이 모듈은 그 실행을 오케스트레이션하는 골격(순차 실행 루프, 결과 직렬화, 스키마 가드)까지
제공한다 — 실제 데이터셋 반복 채점 루프는 그 산출물이 준비된 뒤 `run_qa_pass`를 호출하는
쪽에서 채운다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.eval.interface import GraphRAGMethod, IndexStats, QAResult

# spec.md §8: "우리 방법·LiteSemRAG·NoLLMRAG는 항상 0이어야 하고, 0이 아니면 버그"
LLM_FREE_METHODS = {"ours", "litesemrag", "nollmrag"}


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def sequential_run(
    methods: dict[str, GraphRAGMethod], corpus_path: str, scope: str
) -> dict[str, IndexStats]:
    """method 하나씩 직렬로 index() 실행 — VRAM 경합 방지(spec.md §2).

    `methods`는 dict 순서(= config.yaml의 `methods` 리스트 순서)대로 순차 처리되며,
    이전 method의 index()가 끝나야 다음 method가 시작된다(동시 실행 금지).
    """
    results: dict[str, IndexStats] = {}
    for name, method in methods.items():
        results[name] = method.index(corpus_path, scope)
    return results


def run_qa_pass(
    method: GraphRAGMethod, questions: list[str], top_k: int = 5
) -> list[QAResult]:
    """단일 method에 대해 질문 리스트를 순서대로 query() — Milestone 5(Phase 4.1~4.3)에서
    실제 데이터셋과 함께 호출된다. 여기서는 재사용 가능한 헬퍼만 제공."""
    return [method.query(q, top_k=top_k) for q in questions]


def serialize_result_row(
    *,
    method: str,
    dataset: str,
    difficulty: str,
    em: float,
    f1: float,
    ci95: tuple[float, float],
    teacher_student_agreement: float | None,
    gold_accuracy: dict | None,
    hallucination_rate: float,
    gpu_hours: float,
    indexing_llm_calls: int,
    original_paper_anchor: dict | None,
    corpus_scope: str,
) -> dict[str, Any]:
    """spec.md §8 JSON 스키마로 결과 한 행을 직렬화하고, 스키마 불변식을 가드한다.

    가드:
      1. LLM_FREE_METHODS(ours/litesemrag/nollmrag)는 indexing_llm_calls가 반드시 0.
      2. gold_accuracy는 method == "ours"일 때만 채워질 수 있다(baseline은 None).
    """
    if method in LLM_FREE_METHODS and indexing_llm_calls != 0:
        raise AssertionError(
            f"'{method}'는 LLM-free 계열인데 indexing_llm_calls={indexing_llm_calls} (버그 — spec.md §8)"
        )

    if method != "ours" and gold_accuracy is not None:
        raise AssertionError(
            f"'{method}'는 baseline이라 gold_accuracy는 null이어야 합니다(spec.md §8) — "
            "순환논증 방지 지표는 우리 방법 전용"
        )

    return {
        "method": method,
        "dataset": dataset,
        "difficulty": difficulty,
        "em": em,
        "f1": f1,
        "ci95": list(ci95),
        "teacher_student_agreement": teacher_student_agreement,
        "gold_accuracy": gold_accuracy,
        "hallucination_rate": hallucination_rate,
        "gpu_hours": gpu_hours,
        "indexing_llm_calls": indexing_llm_calls,
        "original_paper_anchor": original_paper_anchor,
        "corpus_scope": corpus_scope,
    }
