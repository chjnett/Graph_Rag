"""TextRank 기반 커뮤니티 요약 — spec.md Phase 3.5 대응, TODO_mac.md #7.

커뮤니티에 속한 edge들의 `source_span` 문장을 모아 TextRank(문장 유사도 그래프 위의
PageRank)로 대표 문장 몇 개를 뽑는다. TF-IDF(scikit-learn) + networkx.pagerank만
사용 — LLM 호출 없음.
"""

from __future__ import annotations

import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def summarize_community(sentences: list[str], top_k: int = 2) -> list[str]:
    """문장 리스트 중 TextRank 점수 상위 top_k개를 원래 순서 유지 없이 점수순으로 반환."""
    unique_sentences = list(dict.fromkeys(s.strip() for s in sentences if s.strip()))
    if not unique_sentences:
        return []
    if len(unique_sentences) <= top_k:
        return unique_sentences

    tfidf = TfidfVectorizer().fit_transform(unique_sentences)
    similarity = cosine_similarity(tfidf)

    graph = nx.from_numpy_array(similarity)
    scores = nx.pagerank(graph)

    ranked_indices = sorted(range(len(unique_sentences)), key=lambda i: scores[i], reverse=True)
    return [unique_sentences[i] for i in ranked_indices[:top_k]]
