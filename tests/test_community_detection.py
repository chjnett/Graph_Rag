"""src/graph_construction/community_detection.py 유닛테스트 — TODO_mac.md #7."""

import networkx as nx

from src.graph_construction.community_detection import detect_communities


def test_detect_communities_empty_graph_returns_empty_dict():
    assert detect_communities(nx.Graph()) == {}


def test_detect_communities_splits_two_disconnected_clusters():
    graph = nx.Graph()
    graph.add_edges_from([("a", "b"), ("b", "c"), ("c", "a")])  # cluster 1
    graph.add_edges_from([("x", "y"), ("y", "z"), ("z", "x")])  # cluster 2 (연결 없음)

    communities = detect_communities(graph)

    assert set(communities.keys()) == {"a", "b", "c", "x", "y", "z"}
    assert communities["a"] == communities["b"] == communities["c"]
    assert communities["x"] == communities["y"] == communities["z"]
    assert communities["a"] != communities["x"]


def test_detect_communities_single_node():
    graph = nx.Graph()
    graph.add_node("solo")
    communities = detect_communities(graph)
    assert communities == {"solo": 0}
