"""논문용 종합 Table 초안 + 난이도별 성능 분해 — spec.md Phase 4.5-a/b (TODO_mac.md #3).

입력은 `benchmark.serialize_result_row()`가 만드는 spec.md §8 스키마 행(dict) 리스트.
실제 그래프 렌더링(matplotlib 등)은 스코프 밖 — 여기서는 난이도별로 그룹화된
데이터까지만 제공하고, 그리기는 이 데이터를 소비하는 쪽(노트북/스크립트)에서 한다.
"""

from __future__ import annotations


def build_markdown_table(rows: list[dict]) -> str:
    """spec.md §8 결과 행 리스트를 논문용 Table 초안(마크다운)으로 직렬화 — Phase 4.5-a.

    컬럼: method, dataset, difficulty, em, f1, ci95, gpu_hours, hallucination_rate,
    original_paper_anchor(있으면 요약 표시).
    """
    header = (
        "| method | dataset | difficulty | em | f1 | ci95 | gpu_hours | hallucination_rate | anchor |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    if not rows:
        return header

    lines = []
    for r in rows:
        ci95 = r.get("ci95")
        ci95_str = f"[{ci95[0]:.2f}, {ci95[1]:.2f}]" if ci95 else "-"
        anchor = r.get("original_paper_anchor")
        anchor_str = ", ".join(f"{k}={v}" for k, v in anchor.items()) if anchor else "-"
        lines.append(
            f"| {r['method']} | {r['dataset']} | {r['difficulty']} | "
            f"{r['em']:.3f} | {r['f1']:.3f} | {ci95_str} | "
            f"{r['gpu_hours']:.2f} | {r['hallucination_rate']:.3f} | {anchor_str} |"
        )
    return header + "\n".join(lines) + "\n"


def group_by_difficulty(rows: list[dict]) -> dict[str, list[dict]]:
    """결과 행을 `difficulty` 필드 기준으로 그룹화 — Phase 4.5-b 난이도별 성능 분해의 입력 데이터.

    실제 막대그래프/선그래프 렌더링은 이 함수가 반환한 그룹을 소비하는 쪽에서 수행한다.
    """
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["difficulty"], []).append(r)
    return groups
