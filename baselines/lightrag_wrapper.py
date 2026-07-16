"""LightRAG wrapper — spec.md Phase 0.5-b (TODO.md Milestone 2, TODO_mac.md #4).

`lightrag-hku`(1.x, Python API: `lightrag.LightRAG` + `lightrag.llm.openai.openai_complete_if_cache`/
`openai_embed`)를 로컬 OpenAI 호환 엔드포인트(vLLM 또는 테스트용 더미 서버)로 설정해
표준 인덱싱/질의를 재현한다.

LightRAG는 MS GraphRAG와 달리 자체 `llm_model_func`/`embedding_func` 콜백을 주입받는
구조라, litellm 없이 이 콜백 호출 횟수를 직접 세면 된다(ms_graphrag_wrapper.py처럼
litellm monkeypatch가 필요 없음).

⚠️ 이 wrapper는 실제 vLLM(GPU) 없이 더미 HTTP 서버(tests/fixtures/dummy_llm_server.py)
대상으로 배선만 검증됐다 — GPU 복구 후 실제 vLLM 엔드포인트로 재검증 필요
(TODO_mac.md #4 "GPU 복구 후 처리할 것" 참고).
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path

from src.eval.interface import Evidence, GraphRAGMethod, IndexStats, QAResult

_DUMMY_EMBEDDING_DIM = 8  # 더미 서버 응답 차원(tests/fixtures/dummy_llm_server.py)과 일치.
# 실제 vLLM 엔드포인트로 전환 시 해당 임베딩 모델의 실제 차원으로 바꿔야 한다.


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
        self._working_dir: Path | None = None
        self._rag = None
        self._index_llm_calls = 0

    def _require_lightrag(self):
        try:
            import lightrag  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "lightrag 미설치. `pip install lightrag-hku` 후 재시도하세요. "
                "(Phase 0.5-b: LightRAG 표준 인덱싱 재현)"
            ) from exc
        return lightrag

    def _build_rag(self, working_dir: Path, call_counter: dict[str, int]):
        from lightrag import LightRAG
        from lightrag.llm.openai import openai_complete_if_cache, openai_embed
        from lightrag.utils import EmbeddingFunc

        async def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
            call_counter["count"] += 1
            return await openai_complete_if_cache(
                self.teacher_model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                api_key="not-needed",
                base_url=self.teacher_endpoint,
                **kwargs,
            )

        async def embedding_func(texts):
            call_counter["count"] += 1
            return await openai_embed(
                texts,
                model=self.teacher_model,
                api_key="not-needed",
                base_url=self.teacher_endpoint,
            )

        return LightRAG(
            working_dir=str(working_dir),
            llm_model_func=llm_model_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=_DUMMY_EMBEDDING_DIM, max_token_size=8192, func=embedding_func
            ),
            # Phase 0.0-e VRAM 안전장치와 같은 취지 — 토큰 단위라 문자 상한과 정확히
            # 대응하진 않지만 과도하게 큰 컨텍스트로 인덱싱하지 않도록 제한.
            chunk_token_size=self.chunk_size_chars[1],
        )

    @staticmethod
    def _read_corpus_text(corpus_path: str) -> str:
        p = Path(corpus_path)
        if p.is_file():
            return p.read_text(encoding="utf-8")
        texts = [f.read_text(encoding="utf-8") for f in sorted(p.glob("*.txt"))]
        if not texts:
            raise FileNotFoundError(f"{corpus_path}에서 .txt 문서를 찾지 못했습니다.")
        return "\n\n".join(texts)

    def _read_graph_counts(self, working_dir: Path) -> tuple[int, int]:
        import networkx as nx

        graphml_files = sorted(working_dir.glob("*.graphml"))
        if not graphml_files:
            return 0, 0
        g = nx.read_graphml(graphml_files[0])
        return g.number_of_nodes(), g.number_of_edges()

    def index(self, corpus_path: str, scope: str) -> IndexStats:
        self._require_lightrag()
        from lightrag.kg.shared_storage import initialize_pipeline_status

        start = time.monotonic()
        working_dir = Path(tempfile.mkdtemp(prefix="lightrag_"))
        call_counter = {"count": 0}
        rag = self._build_rag(working_dir, call_counter)
        text = self._read_corpus_text(corpus_path)

        async def _run():
            await rag.initialize_storages()
            await initialize_pipeline_status()
            await rag.ainsert(text)

        asyncio.run(_run())

        self._working_dir = working_dir
        self._rag = rag
        self._index_llm_calls = call_counter["count"]
        self._indexed = True

        node_count, edge_count = self._read_graph_counts(working_dir)

        return IndexStats(
            wall_clock_sec=time.monotonic() - start,
            gpu_hours=0.0,  # GPU-시간 자체는 benchmark.py가 wall_clock 기준으로 별도 집계(spec.md §2)
            llm_calls=call_counter["count"],
            node_count=node_count,
            edge_count=edge_count,
        )

    def query(self, question: str, top_k: int = 5) -> QAResult:
        if not self._indexed or self._rag is None:
            raise RuntimeError("index()를 먼저 호출해야 합니다.")

        from lightrag import QueryParam

        start = time.monotonic()
        answer = asyncio.run(
            self._rag.aquery(question, param=QueryParam(mode="hybrid", top_k=top_k))
        )
        latency_ms = (time.monotonic() - start) * 1000

        answer = answer if isinstance(answer, str) else str(answer)
        return QAResult(
            answer=answer,
            evidence=[Evidence(source_span=answer[:200], doc_id="lightrag_context")] if answer else [],
            latency_ms=latency_ms,
            indexing_llm_calls=self._index_llm_calls,
        )
