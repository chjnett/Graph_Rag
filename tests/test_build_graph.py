"""src/graph_construction/build_graph.py 유닛테스트 — TODO_mac.md #7.

`tests/fixtures/mock_triples.json`(교사 모델 출력 없이 만든 목 triple 6개, 두 개의
서로 무관한 클러스터: Apple/Jobs 쪽과 Wimbledon 쪽)로 인덱싱 시 LLM 호출 0회를
증명하고, 그래프/커뮤니티 요약이 정상적으로 나오는지 확인한다.
"""

import json
from pathlib import Path

import pytest

from src.graph_construction.build_graph import (
    build_graph,
    index_from_triples,
    summarize_communities,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def mock_triples():
    return json.loads((FIXTURES_DIR / "mock_triples.json").read_text(encoding="utf-8"))


def test_build_graph_creates_nodes_and_edges(mock_triples):
    graph = build_graph(mock_triples)
    assert graph.number_of_nodes() == 8  # Steve Jobs, Apple, Macintosh, Tim Cook, Serena Williams, Wimbledon, London, Roger Federer
    assert graph.number_of_edges() == 6
    assert graph.nodes["Apple"]["type"] == "ORGANIZATION"
    assert graph["Steve Jobs"]["Apple"]["relation"] == "founded"


def test_build_graph_missing_field_raises():
    with pytest.raises(ValueError):
        build_graph([{"entity1": "A", "entity2": "B"}])


def test_build_graph_empty_triples_returns_empty_graph():
    graph = build_graph([])
    assert graph.number_of_nodes() == 0


def test_summarize_communities_finds_two_clusters(mock_triples):
    graph = build_graph(mock_triples)
    summaries = summarize_communities(graph)

    assert len(summaries) == 2  # Apple/Jobs 클러스터 + Wimbledon 클러스터
    all_summary_sentences = [s for sentences in summaries.values() for s in sentences]
    assert any("Wimbledon" in s for s in all_summary_sentences)
    assert any("Apple" in s or "Jobs" in s for s in all_summary_sentences)


def test_index_from_triples_llm_calls_always_zero(mock_triples):
    graph, summaries, stats = index_from_triples(mock_triples)

    assert stats.llm_calls == 0
    assert stats.node_count == graph.number_of_nodes()
    assert stats.edge_count == graph.number_of_edges()
    assert len(summaries) >= 1
