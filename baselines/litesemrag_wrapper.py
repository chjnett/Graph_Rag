"""LiteSemRAG / NoLLMRAG wrapper — spec.md Phase 0.5-c/0.5-e (TODO.md Milestone 2).

둘 다 인덱싱 시점에 LLM을 전혀 호출하지 않는 계열(`indexing_llm_calls=0` 고정)이라
한 파일에 인접 배치했다 (TODO.md: "NoLLMRAG는 ... litesemrag_wrapper.py에 인접 배치 가능").

⚠️ 공개된 공식 구현체가 확인되지 않아, 두 클래스 모두 논문 설명(graphrag_00_overview.md
§1.4 4.3: "표면 통계에 의존")에 기반한 최소 재현이다:
  - NoLLMRAGWrapper: 순수 co-occurrence 빈도(표면 통계)만 사용 — 더 단순한 쪽
  - LiteSemRAGWrapper: TF-IDF 코사인 유사도로 관계 강도를 가중 — "semantic" 명칭에 맞춰
    NoLLMRAG보다 한 단계 더 정교한 lightweight 방식
공식 코드가 나중에 확인되면 이 최소 재현을 대체할 것.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict

import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.eval.interface import Evidence, GraphRAGMethod, IndexStats, QAResult

_ENTITY_PATTERN = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _extract_entities(sentence: str) -> list[str]:
    return sorted(set(_ENTITY_PATTERN.findall(sentence)))


class _CooccurrenceGraphBase(GraphRAGMethod):
    """두 baseline이 공유하는 co-occurrence 그래프 구축/질의 로직."""

    def __init__(self) -> None:
        self.graph: nx.Graph = nx.Graph()
        self._sentence_by_doc: dict[str, list[tuple[str, str]]] = {}  # doc_id -> [(chunk_id, sentence)]

    def index(self, corpus_path: str, scope: str) -> IndexStats:
        start = time.monotonic()
        docs = self._load_corpus(corpus_path)
        for doc_id, text in docs:
            sentences = _split_sentences(text)
            self._sentence_by_doc[doc_id] = [(f"{doc_id}_s{i}", s) for i, s in enumerate(sentences)]
            self._index_document(doc_id, sentences)

        return IndexStats(
            wall_clock_sec=time.monotonic() - start,
            gpu_hours=0.0,
            llm_calls=0,
            node_count=self.graph.number_of_nodes(),
            edge_count=self.graph.number_of_edges(),
        )

    def _load_corpus(self, corpus_path: str) -> list[tuple[str, str]]:
        from pathlib import Path

        p = Path(corpus_path)
        if p.is_file():
            return [(p.stem, p.read_text(encoding="utf-8"))]
        return [(f.stem, f.read_text(encoding="utf-8")) for f in sorted(p.glob("*.txt"))]

    def _index_document(self, doc_id: str, sentences: list[str]) -> None:
        raise NotImplementedError

    def query(self, question: str, top_k: int = 5) -> QAResult:
        start = time.monotonic()
        q_entities = set(_extract_entities(question))
        scored: list[tuple[float, str, str, dict]] = []
        for u, v, data in self.graph.edges(data=True):
            overlap = len({u, v} & q_entities)
            if overlap == 0 and q_entities:
                continue
            scored.append((data.get("weight", 1.0), u, v, data))
        scored.sort(key=lambda t: t[0], reverse=True)
        top = scored[:top_k]

        evidence = [
            Evidence(source_span=data.get("source_span", f"{u} - {v}"), doc_id=data.get("doc_id", "unknown"), chunk_id=data.get("chunk_id"))
            for _, u, v, data in top
        ]
        answer = "; ".join(f"{u} - {v}" for _, u, v, _ in top) if top else ""

        return QAResult(
            answer=answer,
            evidence=evidence,
            latency_ms=(time.monotonic() - start) * 1000,
            indexing_llm_calls=0,
        )


class NoLLMRAGWrapper(_CooccurrenceGraphBase):
    """Phase 0.5-e — 순수 co-occurrence 빈도 기반 (가장 단순한 표면 통계)."""

    def _index_document(self, doc_id: str, sentences: list[str]) -> None:
        for i, sentence in enumerate(sentences):
            entities = _extract_entities(sentence)
            chunk_id = f"{doc_id}_s{i}"
            for a in range(len(entities)):
                for b in range(a + 1, len(entities)):
                    u, v = entities[a], entities[b]
                    if self.graph.has_edge(u, v):
                        self.graph[u][v]["weight"] += 1.0
                    else:
                        self.graph.add_edge(
                            u, v, weight=1.0, doc_id=doc_id, chunk_id=chunk_id, source_span=sentence
                        )


class LiteSemRAGWrapper(_CooccurrenceGraphBase):
    """Phase 0.5-c — TF-IDF 코사인 유사도로 관계 강도를 가중하는 lightweight semantic 방식."""

    def _index_document(self, doc_id: str, sentences: list[str]) -> None:
        if not sentences:
            return
        vectorizer = TfidfVectorizer()
        try:
            tfidf = vectorizer.fit_transform(sentences)
        except ValueError:
            return  # 빈 vocabulary (전부 불용어 등) — 스킵
        sim_matrix = cosine_similarity(tfidf)

        entity_sentence_idx: dict[str, list[int]] = defaultdict(list)
        for i, sentence in enumerate(sentences):
            for entity in _extract_entities(sentence):
                entity_sentence_idx[entity].append(i)

        entities = list(entity_sentence_idx)
        for a in range(len(entities)):
            for b in range(a + 1, len(entities)):
                u, v = entities[a], entities[b]
                best_sim, best_i, best_j = 0.0, None, None
                for i in entity_sentence_idx[u]:
                    for j in entity_sentence_idx[v]:
                        if sim_matrix[i, j] > best_sim:
                            best_sim, best_i, best_j = sim_matrix[i, j], i, j
                if best_i is None or best_sim <= 0:
                    continue
                chunk_id = f"{doc_id}_s{best_i}"
                if self.graph.has_edge(u, v):
                    self.graph[u][v]["weight"] = max(self.graph[u][v]["weight"], float(best_sim))
                else:
                    self.graph.add_edge(
                        u, v, weight=float(best_sim), doc_id=doc_id, chunk_id=chunk_id,
                        source_span=sentences[best_i],
                    )
