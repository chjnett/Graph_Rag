"""IndexStats → 비용 비교 테이블 — spec.md §5 `metrics_cost.py` (TODO.md Milestone 3)."""

from __future__ import annotations

from src.eval.interface import IndexStats


def compare_costs(
    stats_by_method: dict[str, IndexStats],
    api_cost_per_gpu_hour: float | None = None,
) -> list[dict]:
    """method별 IndexStats를 GPU-시간 오름차순 비교 테이블(dict 리스트)로 변환.

    api_cost_per_gpu_hour가 주어지면 참고용 API 환산 비용(`api_cost_estimate`)을 병기한다
    (spec.md §5: "선택" 항목).
    """
    rows = []
    for method, stats in stats_by_method.items():
        row = {
            "method": method,
            "wall_clock_sec": stats.wall_clock_sec,
            "gpu_hours": stats.gpu_hours,
            "llm_calls": stats.llm_calls,
            "node_count": stats.node_count,
            "edge_count": stats.edge_count,
        }
        if api_cost_per_gpu_hour is not None:
            row["api_cost_estimate"] = stats.gpu_hours * api_cost_per_gpu_hour
        rows.append(row)

    rows.sort(key=lambda r: r["gpu_hours"])
    return rows
