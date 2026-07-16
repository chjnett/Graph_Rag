"""src/graph_construction/community_summary.py 유닛테스트 — TODO_mac.md #7."""

from src.graph_construction.community_summary import summarize_community


def test_summarize_community_empty_list():
    assert summarize_community([], top_k=2) == []


def test_summarize_community_fewer_sentences_than_top_k_returns_all():
    sentences = ["Apple makes iPhones.", "Apple makes iPhones."]  # 중복 포함
    result = summarize_community(sentences, top_k=3)
    assert result == ["Apple makes iPhones."]  # 중복 제거 후 1개뿐이라 그대로 반환


def test_summarize_community_picks_top_k_from_many_sentences():
    sentences = [
        "Steve Jobs founded Apple in Cupertino.",
        "Apple released the first Macintosh in 1984.",
        "Tim Cook succeeded Steve Jobs as chief executive.",
        "Serena Williams won Wimbledon multiple times.",
    ]
    result = summarize_community(sentences, top_k=2)
    assert len(result) == 2
    assert all(s in sentences for s in result)
