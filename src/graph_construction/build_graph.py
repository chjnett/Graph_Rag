"""LLM-free 그래프 구축 파이프라인 — spec.md Phase 3.0~3.5 프로토타입, TODO_mac.md #7.

목(mock) triple 세트(spec.md §1 triple 스키마: entity1, entity1_type, relation,
entity2, entity2_type, source_span, confidence)를 받아 엔티티 정규화 → 그래프
생성 → Leiden 커뮤니티 탐지 → TextRank 커뮤니티 요약까지 전부 LLM 호출 없이
수행한다.

이 프로토타입은 `GraphRAGMethod.query()`(검색, Phase 3.35)는 구현하지 않는다 —
그건 서브3 자체 문서(graphrag_03_graph_construction.md) 범위이며, 실제 학생
모델이 만든 triple(교사 모델 출력)이 있어야 의미가 있다.
"""

from __future__ import annotations

import time

import networkx as nx

from src.eval.interface import IndexStats
from src.graph_construction.community_detection import detect_communities
from src.graph_construction.community_summary import summarize_community
from src.graph_construction.entity_normalization import normalize_entities

REQUIRED_TRIPLE_FIELDS = (
    "entity1",
    "entity1_type",
    "relation",
    "entity2",
    "entity2_type",
    "source_span",
    "confidence",
)


def _validate_triples(triples: list[dict]) -> None:
    for t in triples:
        missing = [f for f in REQUIRED_TRIPLE_FIELDS if f not in t]
        if missing:
            raise ValueError(f"triple에 필수 필드 누락: {missing} (spec.md §1 triple 스키마)")


def build_graph(triples: list[dict]) -> nx.DiGraph:
    """정규화된 triple들로 방향그래프 생성. 노드에 entity_type, 엣지에 relation/
    source_span/confidence를 속성으로 붙인다."""
    _validate_triples(triples)
    normalized, _alias_map = normalize_entities(triples)

    graph = nx.DiGraph()
    for t in normalized:
        graph.add_node(t["entity1"], type=t["entity1_type"])
        graph.add_node(t["entity2"], type=t["entity2_type"])
        graph.add_edge(
            t["entity1"],
            t["entity2"],
            relation=t["relation"],
            source_span=t["source_span"],
            confidence=t["confidence"],
        )
    return graph


def summarize_communities(graph: nx.DiGraph, top_k: int = 2) -> dict[int, list[str]]:
    """커뮤니티별로 소속 엣지의 source_span을 모아 TextRank 요약."""
    community_of = detect_communities(graph.to_undirected())

    sentences_by_community: dict[int, list[str]] = {}
    for u, v, data in graph.edges(data=True):
        cid = community_of.get(u, -1)
        sentences_by_community.setdefault(cid, []).append(data["source_span"])

    return {
        cid: summarize_community(sentences, top_k=top_k)
        for cid, sentences in sentences_by_community.items()
    }


def index_from_triples(triples: list[dict]) -> tuple[nx.DiGraph, dict[int, list[str]], IndexStats]:
    """목 triple 리스트 → (그래프, 커뮤니티별 요약, IndexStats).

    `IndexStats.llm_calls`는 정의상 항상 0 — 이 파이프라인은 교사/학생 모델을
    전혀 호출하지 않고, 이미 만들어진 triple만 소비한다.
    """
    start = time.monotonic()
    graph = build_graph(triples)
    summaries = summarize_communities(graph)

    stats = IndexStats(
        wall_clock_sec=time.monotonic() - start,
        gpu_hours=0.0,
        llm_calls=0,
        node_count=graph.number_of_nodes(),
        edge_count=graph.number_of_edges(),
    )
    return graph, summaries, stats
