"""GPU-backed baseline wrapper(ms_graphrag/lightrag) 배선 검증 — TODO_mac.md #4.

실제 vLLM/GPU 없이 로컬 더미 OpenAI 호환 서버(tests/fixtures/dummy_llm_server.py)를
대상으로 index()/query()가 예외 없이 끝까지 도는지만 확인한다. 패키지가 설치돼
있지 않으면 skip — CI/다른 환경에서 무조건 실패하지 않도록.
"""

import pytest

from tests.fixtures.dummy_llm_server import run_dummy_server

SAMPLE_TEXT = (
    "Steve Jobs founded Apple in Cupertino. Apple released the first Macintosh in 1984. "
    "Tim Cook succeeded Steve Jobs as chief executive."
)


@pytest.fixture
def sample_corpus(tmp_path):
    corpus_file = tmp_path / "doc1.txt"
    corpus_file.write_text(SAMPLE_TEXT, encoding="utf-8")
    return corpus_file


def test_ms_graphrag_wrapper_index_and_query_against_dummy_server(sample_corpus):
    pytest.importorskip("graphrag")
    from baselines.ms_graphrag_wrapper import MSGraphRAGWrapper
    from src.eval.interface import IndexStats, QAResult

    with run_dummy_server() as (base_url, server):
        wrapper = MSGraphRAGWrapper(teacher_endpoint=base_url, teacher_model="dummy-model")
        stats = wrapper.index(str(sample_corpus), scope="subset")

        assert isinstance(stats, IndexStats)
        assert stats.llm_calls > 0
        assert server.request_count > 0  # 더미 서버까지 실제로 도달했는지 확인

        result = wrapper.query("Who founded Apple?")
        assert isinstance(result, QAResult)
        assert result.indexing_llm_calls == stats.llm_calls


def test_lightrag_wrapper_index_and_query_against_dummy_server(sample_corpus):
    pytest.importorskip("lightrag")
    from baselines.lightrag_wrapper import LightRAGWrapper
    from src.eval.interface import IndexStats, QAResult

    with run_dummy_server() as (base_url, server):
        wrapper = LightRAGWrapper(teacher_endpoint=base_url, teacher_model="dummy-model")
        stats = wrapper.index(str(sample_corpus), scope="subset")

        assert isinstance(stats, IndexStats)
        assert stats.llm_calls > 0
        assert server.request_count > 0

        result = wrapper.query("Who founded Apple?")
        assert isinstance(result, QAResult)
        assert result.indexing_llm_calls == stats.llm_calls
