"""src/eval/report_table.py 유닛테스트 — TODO_mac.md #3 (Phase 4.5-a/b)."""

from src.eval.report_table import build_markdown_table, group_by_difficulty

_ROW_A = {
    "method": "ours",
    "dataset": "graphrag_bench",
    "difficulty": "fact_retrieval",
    "em": 0.61,
    "f1": 0.68,
    "ci95": [0.58, 0.64],
    "gpu_hours": 3.5,
    "hallucination_rate": 0.09,
    "original_paper_anchor": None,
}

_ROW_B = {
    "method": "ms_graphrag",
    "dataset": "graphrag_bench",
    "difficulty": "complex_reasoning",
    "em": 0.50,
    "f1": 0.55,
    "ci95": [0.45, 0.55],
    "gpu_hours": 10.0,
    "hallucination_rate": 0.12,
    "original_paper_anchor": {"accuracy": 50.93},
}


def test_build_markdown_table_empty_rows_returns_header_only():
    table = build_markdown_table([])
    assert table.startswith("| method |")
    assert table.count("\n") == 2


def test_build_markdown_table_includes_all_rows():
    table = build_markdown_table([_ROW_A, _ROW_B])
    assert "ours" in table
    assert "ms_graphrag" in table
    assert "[0.58, 0.64]" in table
    assert "accuracy=50.93" in table


def test_group_by_difficulty_groups_correctly():
    groups = group_by_difficulty([_ROW_A, _ROW_B])
    assert set(groups.keys()) == {"fact_retrieval", "complex_reasoning"}
    assert groups["fact_retrieval"] == [_ROW_A]
    assert groups["complex_reasoning"] == [_ROW_B]


def test_group_by_difficulty_empty_rows_returns_empty_dict():
    assert group_by_difficulty([]) == {}
