"""QAResult 리스트 → EM/F1, LLM-as-judge, 원 논문 앵커 병기 — spec.md §5 `metrics_qa.py`
(TODO.md Milestone 3).
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Callable

from src.eval.interface import QAResult


def _normalize(text: str) -> str:
    """SQuAD 스타일 정규화: 소문자화, 구두점/관사 제거, 공백 정리."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, references: list[str]) -> float:
    norm_pred = _normalize(prediction)
    return float(any(norm_pred == _normalize(ref) for ref in references))


def f1_score(prediction: str, references: list[str]) -> float:
    pred_tokens = _normalize(prediction).split()
    best = 0.0
    for ref in references:
        ref_tokens = _normalize(ref).split()
        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(ref_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def compute_em_f1(
    predictions: list[QAResult], references: list[list[str]]
) -> dict:
    """predictions[i].answer vs references[i](정답 후보 리스트)로 EM/F1 평균 계산."""
    if len(predictions) != len(references):
        raise ValueError("predictions와 references 길이가 다릅니다.")
    if not predictions:
        return {"em": 0.0, "f1": 0.0, "n": 0}

    ems = [exact_match(p.answer, refs) for p, refs in zip(predictions, references)]
    f1s = [f1_score(p.answer, refs) for p, refs in zip(predictions, references)]
    return {"em": sum(ems) / len(ems), "f1": sum(f1s) / len(f1s), "n": len(predictions)}


def compute_llm_as_judge(
    predictions: list[QAResult],
    references: list[list[str]],
    judge_fn: Callable[[str, list[str]], float],
) -> dict:
    """judge_fn(prediction, references) -> [0, 1] 점수를 매기는 외부(LLM) 채점기를 주입받아 평균 산출.

    Phase 4.1-b 미확정 사항(spec.md §9-3): 채점기 자체가 LLM 호출이라 별도 GPU-시간/비용
    집계 대상 — 이 함수는 채점 로직만 제공하고 비용 집계는 caller 책임.
    """
    if len(predictions) != len(references):
        raise ValueError("predictions와 references 길이가 다릅니다.")
    if not predictions:
        return {"llm_judge_score": 0.0, "n": 0}

    scores = [judge_fn(p.answer, refs) for p, refs in zip(predictions, references)]
    return {"llm_judge_score": sum(scores) / len(scores), "n": len(predictions)}


def merge_with_anchor(metrics: dict, original_paper_anchor: dict | None) -> dict:
    """spec.md §8 출력 스키마의 `original_paper_anchor` 컬럼을 병합."""
    return {**metrics, "original_paper_anchor": original_paper_anchor}
