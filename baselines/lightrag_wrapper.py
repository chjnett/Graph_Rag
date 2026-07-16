"""LightRAG wrapper — spec.md Phase 0.5-b (TODO.md Milestone 2).

`lightrag-hku` 패키지를 로컬 vLLM 엔드포인트(OpenAI 호환)로 설정해 표준 인덱싱/질의를 재현한다.
ms_graphrag_wrapper.py와 동일한 이유로, 실제 인덱싱/질의 호출부는 lightrag 설치 후
그 버전 API를 보고 채워야 하므로 TODO로 명시한다.
"""

from __future__ import annotations

import time

from src.eval.interface import GraphRAGMethod, IndexStats, QAResult


class LightRAGWrapper(GraphRAGMethod):
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

    def _require_lightrag(self):
        try:
            import lightrag  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "lightrag 미설치. `pip install lightrag-hku` 후 재시도하세요. "
                "(Phase 0.5-b: LightRAG 표준 인덱싱 재현)"
            ) from exc
        return lightrag

    def index(self, corpus_path: str, scope: str) -> IndexStats:
        self._require_lightrag()
        start = time.monotonic()

        # TODO(Phase 0.5-b): lightrag 설치 후 실제 설치 버전 API에 맞춰
        # LLM 엔드포인트를 self.teacher_endpoint / self.teacher_model로 설정하고
        # LightRAG(...).insert(...) 형태의 인덱싱을 호출한다 (버전별 API 상이).
        raise NotImplementedError(
            "lightrag 인덱싱 연동 미구현 — lightrag 설치 후 이 함수를 실제 API로 채우세요."
        )

    def query(self, question: str, top_k: int = 5) -> QAResult:
        if not self._indexed:
            raise RuntimeError("index()를 먼저 호출해야 합니다.")
        # TODO(Phase 0.5-b): LightRAG(...).query(...) 호출 후 QAResult로 변환.
        raise NotImplementedError(
            "lightrag 질의 연동 미구현 — lightrag 설치 후 이 함수를 실제 API로 채우세요."
        )
