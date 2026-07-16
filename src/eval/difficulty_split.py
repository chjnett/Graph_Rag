"""GraphRAG-Bench 난이도별 질문 분리 — spec.md Phase 4.1-a (TODO_mac.md #3).

GraphRAG-Bench는 4단계 난이도 태깅을 제공한다(reports/sub3_phase3_6c_anchor.json
조사 과정에서 확인된 arXiv:2506.05690 기준 난이도명과 동일):
fact_retrieval(사실검색) · complex_reasoning(복합추론) ·
contextual_summarization(요약) · creative_generation(생성)
"""

from __future__ import annotations

KNOWN_DIFFICULTIES = (
    "fact_retrieval",
    "complex_reasoning",
    "contextual_summarization",
    "creative_generation",
)


def split_by_difficulty(records: list[dict]) -> dict[str, list[dict]]:
    """`difficulty` 필드를 가진 질문 레코드 리스트를 난이도별로 분리.

    각 레코드는 최소 {"difficulty": str, ...} 형태를 가정한다(질문 텍스트·정답 등
    나머지 필드는 그대로 보존). 알 수 없는 난이도 값이 있으면 데이터 문제를 조기에
    드러내기 위해 즉시 실패한다.
    """
    buckets: dict[str, list[dict]] = {d: [] for d in KNOWN_DIFFICULTIES}
    for rec in records:
        difficulty = rec.get("difficulty")
        if difficulty not in KNOWN_DIFFICULTIES:
            raise ValueError(
                f"알 수 없는 difficulty 값: {difficulty!r} (허용: {KNOWN_DIFFICULTIES})"
            )
        buckets[difficulty].append(rec)
    return buckets
