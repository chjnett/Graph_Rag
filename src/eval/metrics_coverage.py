"""교사-학생 일치율 노출 — spec.md §5 `metrics_coverage.py` (TODO.md Milestone 3).

`graphrag_03` Phase 3.7 리포트(`{agreement: {...}, gold_accuracy: {...}}`)를 그대로
읽어서 노출한다. **재계산하지 않는다** — 순환논증 방지 지표는 서브3 소관 (spec.md §5 주석).
"""

from __future__ import annotations

import json
from pathlib import Path

REQUIRED_KEYS = {"agreement", "gold_accuracy"}


def load_agreement_report(path: str | Path) -> dict:
    """graphrag_03 Phase 3.7 리포트에서 `agreement` 필드만 그대로 반환."""
    with open(path, encoding="utf-8") as f:
        report = json.load(f)

    missing = REQUIRED_KEYS - report.keys()
    if missing:
        raise ValueError(
            f"{path}가 graphrag_03 Phase 3.7 스키마와 다릅니다. 누락된 키: {missing}"
        )
    return report["agreement"]
