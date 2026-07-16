"""Evidence.source_span 대조 기반 환각률 — spec.md §5 `metrics_hallucination.py`
(TODO.md Milestone 3).
"""

from __future__ import annotations

from src.eval.interface import QAResult


def _span_is_grounded(span: str, doc_id: str, source_documents: dict[str, str]) -> bool:
    source_text = source_documents.get(doc_id)
    if source_text is None:
        return False
    return span.strip() in source_text


def hallucination_rate(
    results: list[QAResult], source_documents: dict[str, str]
) -> float:
    """전체 evidence span 중 원문(`source_documents[doc_id]`)에서 실제로 찾을 수 없는 비율.

    QAResult.answer 자체가 아니라 evidence.source_span의 근거 존재 여부를 검사한다
    (spec.md: "Evidence.source_span을 원문과 대조").
    """
    all_spans = [
        (ev.source_span, ev.doc_id) for r in results for ev in r.evidence
    ]
    if not all_spans:
        return 0.0

    ungrounded = sum(
        1 for span, doc_id in all_spans if not _span_is_grounded(span, doc_id, source_documents)
    )
    return ungrounded / len(all_spans)
