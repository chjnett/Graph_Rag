"""Microsoft GraphRAG wrapper — spec.md Phase 0.5-a (TODO.md Milestone 2).

`graphrag` 패키지를 로컬 vLLM 엔드포인트(OpenAI 호환)로 설정해 표준 인덱싱/질의를 재현한다.
인덱싱 LLM 호출 지점을 실제 로컬 엔드포인트로 연결하는 작업은 `graphrag` 설치 후
그 버전의 config 스키마를 보고 채워야 하므로, 이 파일은 GraphRAGMethod 계약을 만족하는
뼈대 + 설정 전달까지만 구현하고 실제 인덱싱/질의 호출부는 TODO로 명시한다.

VRAM 안전장치(graphrag_00 Phase 0.0-e)를 그대로 따른다: gpu_memory_utilization=0.90,
chunk_size는 500~1,000자.
"""

from __future__ import annotations

import time

from src.eval.interface import GraphRAGMethod, IndexStats, QAResult


class MSGraphRAGWrapper(GraphRAGMethod):
    def __init__(
        self,
        teacher_endpoint: str = "http://localhost:8000/v1",
        teacher_model: str = "Qwen2.5-32B-Instruct-AWQ",
        gpu_memory_utilization: float = 0.90,
        chunk_size_chars: tuple[int, int] = (500, 1000),
    ) -> None:
        self.teacher_endpoint = teacher_endpoint
        self.teacher_model = teacher_model
        self.gpu_memory_utilization = gpu_memory_utilization
        self.chunk_size_chars = chunk_size_chars
        self._indexed = False

    def _require_graphrag(self):
        try:
            import graphrag  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "graphrag 미설치. `pip install graphrag` 후 재시도하세요. "
                "(Phase 0.5-a: MS GraphRAG 표준 인덱싱 재현)"
            ) from exc
        return graphrag

    def index(self, corpus_path: str, scope: str) -> IndexStats:
        self._require_graphrag()
        start = time.monotonic()

        # TODO(Phase 0.5-a): graphrag 설치 후 실제 설치 버전 config 스키마에 맞춰
        # LLM 엔드포인트를 self.teacher_endpoint / self.teacher_model로,
        # chunk 크기를 self.chunk_size_chars로 설정하고 인덱싱 파이프라인을 호출한다.
        # 예: graphrag.index.run_pipeline(config=...) 형태 (버전별 API 상이).
        raise NotImplementedError(
            "graphrag 인덱싱 연동 미구현 — graphrag 설치 후 이 함수를 실제 API로 채우세요."
        )

    def query(self, question: str, top_k: int = 5) -> QAResult:
        if not self._indexed:
            raise RuntimeError("index()를 먼저 호출해야 합니다.")
        # TODO(Phase 0.5-a): graphrag의 local/global search API 호출 후 QAResult로 변환.
        raise NotImplementedError(
            "graphrag 질의 연동 미구현 — graphrag 설치 후 이 함수를 실제 API로 채우세요."
        )
