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


def aggregate_ours_gpu_hours(stage_gpu_hours: dict[str, float]) -> float:
    """서브1(데이터 생성)+서브2(증류)+서브3(그래프 구축) GPU-시간 합산 — spec.md Phase 4.4-a.

    baseline은 index() 한 번의 GPU-시간이 전부지만, 우리 방법은 증류 파이프라인
    전체(데이터 생성·SFT·그래프 구축)를 거치므로 동일 단위로 비교하려면 단계별
    GPU-시간을 먼저 합산해야 한다.
    """
    return sum(stage_gpu_hours.values())


def compare_total_costs(
    ours_stage_gpu_hours: dict[str, float], baseline_stats: dict[str, IndexStats]
) -> list[dict]:
    """우리 방법(단계 합산) vs baseline 각각(index() 1회)의 GPU-시간을 동일 단위로 비교 — Phase 4.4-b.

    baseline은 `stage_breakdown=None`(단일 인덱싱 호출), 우리 방법은 단계별 분해를
    `stage_breakdown`에 남겨 어느 단계가 비용을 지배하는지 추적 가능하게 한다.
    """
    rows = [
        {
            "method": "ours",
            "gpu_hours": aggregate_ours_gpu_hours(ours_stage_gpu_hours),
            "stage_breakdown": dict(ours_stage_gpu_hours),
        }
    ]
    for method, stats in baseline_stats.items():
        rows.append({"method": method, "gpu_hours": stats.gpu_hours, "stage_breakdown": None})

    rows.sort(key=lambda r: r["gpu_hours"])
    return rows
