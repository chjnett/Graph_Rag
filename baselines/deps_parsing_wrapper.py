"""Dependency-parsing 기반 그래프 구축 wrapper — spec.md Phase 0.5-d (TODO.md Milestone 2).

대상 논문: arXiv:2507.03226.

⚠️ 미확정 사항 (spec.md §9-2): 이 논문의 공개 코드가 있는지 확인되지 않았다.
확인 전까지는 논문 기술(의존구문분석 기반 SVO 트리플 추출)에 대한 표준적 최소
재현으로 스코프를 축소한다 — spaCy 의존구문분석 트리로 (주어, 동사, 목적어) 트리플을
뽑는 통상적인 접근. 공개 코드가 확인되면 이 최소 재현을 대체할 것.

LLM을 전혀 호출하지 않으므로 indexing_llm_calls는 항상 0.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import networkx as nx

from src.eval.interface import Evidence, GraphRAGMethod, IndexStats, QAResult

_SPACY_MODEL = "en_core_web_sm"


def _load_spacy():
    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError(
            "spacy 미설치. `pip install spacy && python -m spacy download "
            f"{_SPACY_MODEL}` 후 재시도하세요."
        ) from exc
    try:
        return spacy.load(_SPACY_MODEL)
    except OSError as exc:
        raise RuntimeError(
            f"spacy 모델 {_SPACY_MODEL} 미설치. `python -m spacy download {_SPACY_MODEL}` 실행하세요."
        ) from exc


def _extract_svo_triples(doc) -> list[tuple[str, str, str, str]]:
    """spaCy Doc에서 (subject, verb, object, source_sentence) 트리플 추출."""
    triples = []
    for sent in doc.sents:
        for token in sent:
            if token.pos_ != "VERB":
                continue
            subjects = [c for c in token.children if c.dep_ in ("nsubj", "nsubjpass")]
            objects = [c for c in token.children if c.dep_ in ("dobj", "pobj", "attr")]
            for subj in subjects:
                for obj in objects:
                    triples.append((subj.text, token.lemma_, obj.text, sent.text.strip()))
    return triples


class DepsParsingWrapper(GraphRAGMethod):
    """GraphRAGMethod 구현 — spaCy 의존구문분석 기반 SVO 트리플 추출."""

    def __init__(self) -> None:
        self._nlp = None
        self.graph: nx.DiGraph = nx.DiGraph()

    def index(self, corpus_path: str, scope: str) -> IndexStats:
        start = time.monotonic()
        self._nlp = _load_spacy()

        p = Path(corpus_path)
        docs = [(p.stem, p.read_text(encoding="utf-8"))] if p.is_file() else [
            (f.stem, f.read_text(encoding="utf-8")) for f in sorted(p.glob("*.txt"))
        ]

        for doc_id, text in docs:
            spacy_doc = self._nlp(text)
            for i, (subj, verb, obj, sentence) in enumerate(_extract_svo_triples(spacy_doc)):
                chunk_id = f"{doc_id}_t{i}"
                self.graph.add_edge(
                    subj, obj, relation=verb, doc_id=doc_id, chunk_id=chunk_id, source_span=sentence
                )

        return IndexStats(
            wall_clock_sec=time.monotonic() - start,
            gpu_hours=0.0,
            llm_calls=0,
            node_count=self.graph.number_of_nodes(),
            edge_count=self.graph.number_of_edges(),
        )

    def query(self, question: str, top_k: int = 5) -> QAResult:
        start = time.monotonic()
        q_tokens = set(re.findall(r"\w+", question.lower()))

        scored = []
        for u, v, data in self.graph.edges(data=True):
            overlap = len({u.lower(), v.lower()} & q_tokens)
            scored.append((overlap, u, v, data))
        scored.sort(key=lambda t: t[0], reverse=True)
        top = [t for t in scored if t[0] > 0][:top_k] or scored[:top_k]

        evidence = [
            Evidence(source_span=data["source_span"], doc_id=data["doc_id"], chunk_id=data["chunk_id"])
            for _, _, _, data in top
        ]
        answer = "; ".join(f"{u} {data['relation']} {v}" for _, u, v, data in top) if top else ""

        return QAResult(
            answer=answer,
            evidence=evidence,
            latency_ms=(time.monotonic() - start) * 1000,
            indexing_llm_calls=0,
        )
