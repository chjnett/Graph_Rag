"""공통 QA 인터페이스 — spec.md §4.

5개 baseline wrapper(baselines/*.py) + 우리 방법(src/graph_construction 기반)이
전부 이 계약을 구현한다. benchmark.py는 이 인터페이스에만 의존하고 구현 세부는 모른다.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Evidence:
    source_span: str
    doc_id: str
    chunk_id: str | None = None


@dataclass
class QAResult:
    answer: str
    evidence: list[Evidence]
    latency_ms: float
    indexing_llm_calls: int  # 인덱싱 시점 LLM 호출 수. 우리 방법/LiteSemRAG/NoLLMRAG는 항상 0


@dataclass
class IndexStats:
    wall_clock_sec: float
    gpu_hours: float
    llm_calls: int
    node_count: int | None = None
    edge_count: int | None = None


class GraphRAGMethod(ABC):
    """모든 wrapper(baselines/*.py, src/graph_construction 기반 우리 방법)가 구현하는 공통 인터페이스."""

    @abstractmethod
    def index(self, corpus_path: str, scope: str) -> IndexStats:
        """코퍼스를 인덱싱. scope: 'full' | 'subset'. 반환값에 GPU-시간/LLM 호출 수 포함."""
        raise NotImplementedError

    @abstractmethod
    def query(self, question: str, top_k: int = 5) -> QAResult:
        """단일 질의에 대해 검색+응답 생성까지 end-to-end 수행."""
        raise NotImplementedError
