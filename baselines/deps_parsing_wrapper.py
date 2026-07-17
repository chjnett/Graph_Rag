"""Dependency-parsing 기반 그래프 구축 wrapper — spec.md Phase 0.5-d (TODO.md Milestone 2).

대상 논문: arXiv:2507.03226, "Towards Practical GraphRAG: Efficient Knowledge Graph
Construction and Hybrid Retrieval at Scale" (Min et al.). 아키텍처 상세는 spec.md §9-2.

⚠️ 미확정 사항 (spec.md §9-2): 이 논문의 공개 코드가 있는지 확인되지 않았다(2026-07 기준).
확인 전까지는 논문 본문 기술 내용에 대한 재현으로 스코프를 한정한다:
  - 전처리: 동사구 없는 문장 필터링
  - 추출: spaCy 의존구문분석 트리로 (주어, 동사, 목적어) 트리플 추출, 명사구(noun_chunk)
    단위로 다중토큰 엔티티 병합
  - 후처리: 수동태 정규화(by-구 있는 경우만 능동태로 스왑, 없으면 스킵), 짧은
    엔티티(2자 미만)·불용어 제거, 모든 엔티티에 `type="Concept"` 부여
  - **스코프 밖**: coreference resolution — 논문은 언급하지만 별도 라이브러리(예:
    fastcoref/neuralcoref) 필요, 이번 재현에서는 구현하지 않음(TODO_mac.md 1-2 참고)

Docling 기반 포맷 무관 파싱, 계층적 청킹(2048자 상한)은 재현하지 않고 입력을
plain text 파일로 가정한다(§5 모듈 구조 범위 밖).

LLM을 전혀 호출하지 않으므로 indexing_llm_calls는 항상 0.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import networkx as nx

from src.eval.interface import Evidence, GraphRAGMethod, IndexStats, QAResult

_SPACY_MODEL = "en_core_web_sm"
_MIN_ENTITY_LEN = 2


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


def _build_noun_chunk_map(doc) -> dict[int, str]:
    """토큰 인덱스 → 그 토큰을 포함하는 명사구(noun_chunk) 전체 텍스트. 다중토큰 엔티티 병합용."""
    chunk_map: dict[int, str] = {}
    for chunk in doc.noun_chunks:
        for tok in chunk:
            chunk_map[tok.i] = chunk.text
    return chunk_map


def _is_valid_entity(token, text: str) -> bool:
    """짧은 엔티티(2자 미만)·불용어 제거."""
    stripped = text.strip()
    if len(stripped) < _MIN_ENTITY_LEN:
        return False
    if token.is_stop:
        return False
    return True


def _extract_svo_triples(doc) -> list[tuple[str, str, str, str]]:
    """spaCy Doc에서 (subject, verb, object, source_sentence) 트리플 추출.

    논문 기술 재현: 동사구 없는 문장 필터링 + 수동태 정규화(by-구 있을 때만 능동태로
    스왑) + 명사구 단위 다중토큰 엔티티 병합 + 짧은 엔티티/불용어 제거.
    """
    triples = []
    chunk_map = _build_noun_chunk_map(doc)

    def expand(token):
        return chunk_map.get(token.i, token.text)

    for sent in doc.sents:
        if not any(tok.pos_ == "VERB" for tok in sent):
            continue  # 동사구 없는 문장 필터링

        for token in sent:
            if token.pos_ != "VERB":
                continue

            active_subjects = [c for c in token.children if c.dep_ == "nsubj"]
            passive_subjects = [c for c in token.children if c.dep_ == "nsubjpass"]
            objects = [c for c in token.children if c.dep_ in ("dobj", "pobj", "attr")]

            if passive_subjects:
                # 수동태 정규화: "X was verb-ed by Y" → (Y, verb, X). agent(by-구)가
                # 없으면 진짜 주어를 알 수 없으므로 논문 취지에 따라 생성하지 않고 스킵.
                agents = [c for c in token.children if c.dep_ == "agent"]
                for patient in passive_subjects:
                    for agent in agents:
                        for real_subj in (c for c in agent.children if c.dep_ == "pobj"):
                            subj_text, obj_text = expand(real_subj), expand(patient)
                            if _is_valid_entity(real_subj, subj_text) and _is_valid_entity(patient, obj_text):
                                triples.append((subj_text, token.lemma_, obj_text, sent.text.strip()))
                continue

            for subj in active_subjects:
                for obj in objects:
                    subj_text, obj_text = expand(subj), expand(obj)
                    if _is_valid_entity(subj, subj_text) and _is_valid_entity(obj, obj_text):
                        triples.append((subj_text, token.lemma_, obj_text, sent.text.strip()))
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
                # 논문 §9-2: 도메인별 세분화 없이 전부 type="Concept"로 단일 처리
                self.graph.add_node(subj, type="Concept")
                self.graph.add_node(obj, type="Concept")
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
