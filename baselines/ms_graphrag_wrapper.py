"""Microsoft GraphRAG wrapper — spec.md Phase 0.5-a (TODO.md Milestone 2, TODO_mac.md #4).

`graphrag`(3.x, Python API: `graphrag.cli.initialize.initialize_project_at` +
`graphrag.config.load_config` + `graphrag.api.build_index`/`local_search`)를 로컬
OpenAI 호환 엔드포인트(vLLM 또는 테스트용 더미 서버)로 설정해 표준 인덱싱/질의를
재현한다.

VRAM 안전장치(graphrag_00 Phase 0.0-e)를 그대로 따른다: gpu_memory_utilization=0.90,
chunk_size는 500~1,000자 상한(`chunking.size`로 반영).

⚠️ 이 wrapper는 실제 vLLM(GPU) 없이 더미 HTTP 서버(tests/fixtures/dummy_llm_server.py)
대상으로 배선만 검증됐다 — GPU 복구 후 실제 vLLM 엔드포인트로 재검증 필요
(TODO_mac.md #4 "GPU 복구 후 처리할 것" 참고).
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import time
from pathlib import Path

from src.eval.interface import Evidence, GraphRAGMethod, IndexStats, QAResult

_DEFAULT_COMMUNITY_LEVEL = 1
_DEFAULT_RESPONSE_TYPE = "multiple paragraphs"


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
        self._root_dir: Path | None = None
        self._config = None
        self._index_llm_calls = 0

    def _require_graphrag(self):
        try:
            import graphrag  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "graphrag 미설치. `pip install graphrag` 후 재시도하세요. "
                "(Phase 0.5-a: MS GraphRAG 표준 인덱싱 재현)"
            ) from exc
        return graphrag

    def _build_config(self, root_dir: Path):
        from graphrag.cli.initialize import initialize_project_at
        from graphrag.config.load_config import load_config

        initialize_project_at(
            root_dir, force=True, model=self.teacher_model, embedding_model=self.teacher_model
        )

        overrides = {
            "completion_models": {
                "default_completion_model": {
                    "model_provider": "openai",
                    "model": self.teacher_model,
                    "api_base": self.teacher_endpoint,
                    "api_key": "not-needed",
                }
            },
            "embedding_models": {
                "default_embedding_model": {
                    "model_provider": "openai",
                    "model": self.teacher_model,
                    "api_base": self.teacher_endpoint,
                    "api_key": "not-needed",
                }
            },
            # Phase 0.0-e VRAM 안전장치: 청크 상한
            "chunking": {"size": self.chunk_size_chars[1], "overlap": 0},
        }
        return load_config(root_dir, cli_overrides=overrides)

    @staticmethod
    def _copy_corpus_into(corpus_path: str, input_dir: Path) -> None:
        p = Path(corpus_path)
        docs = [p] if p.is_file() else sorted(p.glob("*.txt"))
        if not docs:
            raise FileNotFoundError(f"{corpus_path}에서 .txt 문서를 찾지 못했습니다.")
        for f in docs:
            shutil.copy(f, input_dir / f.name)

    @staticmethod
    def _read_parquet_or_empty(output_dir: Path, name: str):
        import pandas as pd

        path = output_dir / f"{name}.parquet"
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()

    def index(self, corpus_path: str, scope: str) -> IndexStats:
        self._require_graphrag()
        from graphrag.api import build_index

        start = time.monotonic()
        root_dir = Path(tempfile.mkdtemp(prefix="ms_graphrag_"))
        config = self._build_config(root_dir)
        self._copy_corpus_into(corpus_path, root_dir / "input")

        counter = _LLMCallCounter()
        originals = _install_litellm_counter(counter)
        try:
            # graphrag CLI가 `graphrag index` 실행 전 수행하는 연결성 확인 + 벡터스토어
            # 차원 자동 동기화. Python API(build_index)는 이걸 자동으로 안 해줘서
            # 직접 호출해야 한다 — 안 하면 임베딩 차원 불일치로 벡터스토어 적재가 죽는다.
            from graphrag.index.validate_config import validate_config_names

            validate_config_names(config)

            results = asyncio.run(build_index(config))
        finally:
            _restore_litellm(originals)

        errors = [r for r in results if r.error is not None]
        if errors:
            raise RuntimeError(
                f"graphrag 인덱싱 파이프라인 실패 (workflow={errors[0].workflow}): {errors[0].error}"
            )

        self._root_dir = root_dir
        self._config = config
        self._index_llm_calls = counter.count
        self._indexed = True

        output_dir = root_dir / "output"
        entities = self._read_parquet_or_empty(output_dir, "entities")
        relationships = self._read_parquet_or_empty(output_dir, "relationships")

        return IndexStats(
            wall_clock_sec=time.monotonic() - start,
            gpu_hours=0.0,  # GPU-시간 자체는 benchmark.py가 wall_clock 기준으로 별도 집계(spec.md §2)
            llm_calls=counter.count,
            node_count=len(entities),
            edge_count=len(relationships),
        )

    def query(self, question: str, top_k: int = 5) -> QAResult:
        if not self._indexed or self._root_dir is None:
            raise RuntimeError("index()를 먼저 호출해야 합니다.")

        from graphrag.api import local_search

        output_dir = self._root_dir / "output"
        entities = self._read_parquet_or_empty(output_dir, "entities")
        communities = self._read_parquet_or_empty(output_dir, "communities")
        community_reports = self._read_parquet_or_empty(output_dir, "community_reports")
        text_units = self._read_parquet_or_empty(output_dir, "text_units")
        relationships = self._read_parquet_or_empty(output_dir, "relationships")

        start = time.monotonic()
        response, _context = asyncio.run(
            local_search(
                config=self._config,
                entities=entities,
                communities=communities,
                community_reports=community_reports,
                text_units=text_units,
                relationships=relationships,
                covariates=None,
                community_level=_DEFAULT_COMMUNITY_LEVEL,
                response_type=_DEFAULT_RESPONSE_TYPE,
                query=question,
            )
        )
        latency_ms = (time.monotonic() - start) * 1000

        answer = response if isinstance(response, str) else str(response)
        return QAResult(
            answer=answer,
            evidence=[Evidence(source_span=answer[:200], doc_id="ms_graphrag_context")] if answer else [],
            latency_ms=latency_ms,
            indexing_llm_calls=self._index_llm_calls,
        )


class _LLMCallCounter:
    """실제 LLM 호출 수(completion+embedding)를 센다.

    litellm의 success_callback/CustomLogger 훅은 버전에 따라 호출이 안 되는 경우가
    있어 신뢰할 수 없었다(직접 확인함) — 대신 litellm.completion/acompletion/
    embedding/aembedding을 직접 monkeypatch해서 호출 시점에 카운트한다.
    """

    def __init__(self) -> None:
        self.count = 0


def _install_litellm_counter(counter: _LLMCallCounter) -> list[tuple[object, str, object]]:
    """4개 함수를 카운팅 래퍼로 교체하고, 복원용 (모듈, 이름, 원본함수) 리스트를 반환."""
    import litellm

    originals: list[tuple[object, str, object]] = []
    for name in ("completion", "acompletion", "embedding", "aembedding"):
        original = getattr(litellm, name)
        originals.append((litellm, name, original))

        if name.startswith("a"):

            async def _async_wrapped(*args, _orig=original, **kwargs):
                counter.count += 1
                return await _orig(*args, **kwargs)

            setattr(litellm, name, _async_wrapped)
        else:

            def _sync_wrapped(*args, _orig=original, **kwargs):
                counter.count += 1
                return _orig(*args, **kwargs)

            setattr(litellm, name, _sync_wrapped)

    return originals


def _restore_litellm(originals: list[tuple[object, str, object]]) -> None:
    for module, name, original in originals:
        setattr(module, name, original)
