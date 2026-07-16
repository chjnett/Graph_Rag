"""src/graph_construction/entity_normalization.py 유닛테스트 — TODO_mac.md #7."""

from src.graph_construction.entity_normalization import normalize_entities

_TRIPLE = {
    "entity1_type": "PERSON",
    "relation": "founded",
    "entity2_type": "ORGANIZATION",
    "source_span": "x",
    "confidence": 0.9,
}


def test_normalize_entities_merges_case_and_whitespace_variants():
    triples = [
        {**_TRIPLE, "entity1": "Steve Jobs", "entity2": "Apple"},
        {**_TRIPLE, "entity1": "steve jobs", "entity2": "  Apple  ".strip()},
    ]
    normalized, alias_map = normalize_entities(triples)

    assert normalized[0]["entity1"] == normalized[1]["entity1"] == "Steve Jobs"
    assert alias_map["steve jobs"] == "Steve Jobs"
    assert alias_map["Steve Jobs"] == "Steve Jobs"


def test_normalize_entities_picks_longest_variant_as_representative():
    triples = [
        {**_TRIPLE, "entity1": "Apple", "entity2": "x"},
        {**_TRIPLE, "entity1": "APPLE INC", "entity2": "x"},
    ]
    # "Apple"과 "APPLE INC"는 canonical key가 달라(단어 수 다름) 별개 엔티티로 남아야 함
    normalized, alias_map = normalize_entities(triples)
    assert {t["entity1"] for t in normalized} == {"Apple", "APPLE INC"}


def test_normalize_entities_empty_list():
    normalized, alias_map = normalize_entities([])
    assert normalized == []
    assert alias_map == {}


def test_normalize_entities_does_not_merge_different_entities():
    triples = [
        {**_TRIPLE, "entity1": "Apple", "entity2": "x"},
        {**_TRIPLE, "entity1": "Google", "entity2": "x"},
    ]
    normalized, _ = normalize_entities(triples)
    assert {t["entity1"] for t in normalized} == {"Apple", "Google"}
