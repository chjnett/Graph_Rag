"""src/eval/difficulty_split.py 유닛테스트 — TODO_mac.md #3 (Phase 4.1-a)."""

import pytest

from src.eval.difficulty_split import KNOWN_DIFFICULTIES, split_by_difficulty


def test_split_by_difficulty_groups_records():
    records = [
        {"q": "who founded Apple?", "difficulty": "fact_retrieval"},
        {"q": "why did X cause Y?", "difficulty": "complex_reasoning"},
        {"q": "summarize doc", "difficulty": "contextual_summarization"},
        {"q": "write a story", "difficulty": "creative_generation"},
        {"q": "another fact", "difficulty": "fact_retrieval"},
    ]
    buckets = split_by_difficulty(records)

    assert set(buckets.keys()) == set(KNOWN_DIFFICULTIES)
    assert len(buckets["fact_retrieval"]) == 2
    assert len(buckets["complex_reasoning"]) == 1
    assert buckets["fact_retrieval"][0]["q"] == "who founded Apple?"


def test_split_by_difficulty_empty_list_returns_empty_buckets():
    buckets = split_by_difficulty([])
    assert all(bucket == [] for bucket in buckets.values())


def test_split_by_difficulty_unknown_value_raises():
    with pytest.raises(ValueError):
        split_by_difficulty([{"q": "?", "difficulty": "impossible_level"}])


def test_split_by_difficulty_missing_field_raises():
    with pytest.raises(ValueError):
        split_by_difficulty([{"q": "no difficulty field"}])
