"""골드셋 실제 정확도 노출 — spec.md §5 `metrics_gold_accuracy.py` (TODO.md Milestone 3).

`metrics_coverage.py`와 동일한 서브3 Phase 3.7 리포트를 소비하되 `gold_accuracy` 필드를
노출한다. **재계산하지 않는다.** baseline row에는 이 지표를 채우지 않는다(spec.md §8:
"baseline은 별도 골드셋 채점 대상이 아님 — 순환논증 방지 지표는 우리 방법 전용").
"""

from __future__ import annotations

import json
from pathlib import Path

REQUIRED_KEYS = {"agreement", "gold_accuracy"}


def load_gold_accuracy_report(path: str | Path) -> dict:
    """graphrag_03 Phase 3.7 리포트에서 `gold_accuracy`(precision/recall) 필드만 그대로 반환."""
    with open(path, encoding="utf-8") as f:
        report = json.load(f)

    missing = REQUIRED_KEYS - report.keys()
    if missing:
        raise ValueError(
            f"{path}가 graphrag_03 Phase 3.7 스키마와 다릅니다. 누락된 키: {missing}"
        )
    return report["gold_accuracy"]
