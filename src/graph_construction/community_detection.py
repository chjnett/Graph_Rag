"""Leiden 커뮤니티 탐지 — spec.md Phase 3.4 대응, TODO_mac.md #7.

`python-igraph` + `leidenalg` 기반. 둘 중 하나라도 미설치면(선택 의존성)
networkx 내장 Louvain(`nx.community.louvain_communities`)으로 대체한다 —
TODO_mac.md에 명시된 "또는 networkx 대안". 커뮤니티 품질은 알고리즘마다
다를 수 있지만, "인덱싱 시 LLM 호출 0회"라는 이 프로젝트의 핵심 계약에는
어느 쪽을 쓰든 영향이 없다.
"""

from __future__ import annotations

import networkx as nx


def detect_communities(graph: nx.Graph) -> dict[str, int]:
    """노드 이름 → 커뮤니티 id 매핑. 빈 그래프는 빈 dict."""
    if graph.number_of_nodes() == 0:
        return {}
    try:
        return _detect_with_leiden(graph)
    except ImportError:
        return _detect_with_louvain(graph)


def _detect_with_leiden(graph: nx.Graph) -> dict[str, int]:
    import igraph as ig
    import leidenalg

    nodes = list(graph.nodes())
    index_of = {n: i for i, n in enumerate(nodes)}

    ig_graph = ig.Graph()
    ig_graph.add_vertices(len(nodes))
    ig_graph.add_edges([(index_of[u], index_of[v]) for u, v in graph.edges()])

    partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition)
    return {
        nodes[i]: community_id
        for community_id, community in enumerate(partition)
        for i in community
    }


def _detect_with_louvain(graph: nx.Graph) -> dict[str, int]:
    communities = nx.community.louvain_communities(graph, seed=13)
    return {node: idx for idx, community in enumerate(communities) for node in community}
