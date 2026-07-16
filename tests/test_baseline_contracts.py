"""Baseline wrapper 계약 테스트 — TODO.md Milestone 2 마지막 항목.

5개 wrapper 전부:
  1) GraphRAGMethod를 구현하는지 (isinstance)
  2) query()가 QAResult를 반환하는지 — LLM-free 계열(litesemrag/nollmrag/deps_parsing)은
     실제로 index()+query()를 실행해 검증. GPU/vLLM이 필요한 ms_graphrag/lightrag는
     패키지 미설치 시 명확한 RuntimeError로 실패하는지만 검증(서브4 spec.md §6: 이 두
     wrapper의 실제 end-to-end 실행은 인프라가 갖춰진 뒤).
"""

import pytest

from baselines.deps_parsing_wrapper import DepsParsingWrapper
from baselines.lightrag_wrapper import LightRAGWrapper
from baselines.litesemrag_wrapper import LiteSemRAGWrapper, NoLLMRAGWrapper
from baselines.ms_graphrag_wrapper import MSGraphRAGWrapper
from src.eval.interface import GraphRAGMethod, QAResult

SAMPLE_TEXT = (
    "Steve Jobs founded Apple in Cupertino. Apple released the first Macintosh in 1984. "
    "Tim Cook succeeded Steve Jobs as chief executive."
)

ALL_WRAPPER_CLASSES = [
    MSGraphRAGWrapper,
    LightRAGWrapper,
    LiteSemRAGWrapper,
    NoLLMRAGWrapper,
    DepsParsingWrapper,
]


@pytest.mark.parametrize("wrapper_cls", ALL_WRAPPER_CLASSES)
def test_implements_graphragmethod(wrapper_cls):
    instance = wrapper_cls()
    assert isinstance(instance, GraphRAGMethod)


@pytest.fixture
def sample_corpus(tmp_path):
    corpus_file = tmp_path / "doc1.txt"
    corpus_file.write_text(SAMPLE_TEXT, encoding="utf-8")
    return corpus_file


@pytest.mark.parametrize("wrapper_cls", [LiteSemRAGWrapper, NoLLMRAGWrapper])
def test_llm_free_cooccurrence_wrappers_end_to_end(wrapper_cls, sample_corpus):
    wrapper = wrapper_cls()
    stats = wrapper.index(str(sample_corpus), scope="subset")
    assert stats.llm_calls == 0
    assert stats.node_count is not None and stats.node_count > 0

    result = wrapper.query("What did Steve Jobs found?", top_k=3)
    assert isinstance(result, QAResult)
    assert result.indexing_llm_calls == 0
    assert isinstance(result.evidence, list)


def test_deps_parsing_wrapper_end_to_end(sample_corpus):
    pytest.importorskip("spacy")
    wrapper = DepsParsingWrapper()
    stats = wrapper.index(str(sample_corpus), scope="subset")
    assert stats.llm_calls == 0

    result = wrapper.query("Who founded Apple?", top_k=3)
    assert isinstance(result, QAResult)
    assert result.indexing_llm_calls == 0


@pytest.mark.parametrize(
    "wrapper_cls,package_name",
    [(MSGraphRAGWrapper, "graphrag"), (LightRAGWrapper, "lightrag")],
)
def test_gpu_backed_wrappers_fail_clearly_without_package(wrapper_cls, package_name, sample_corpus):
    wrapper = wrapper_cls()
    try:
        __import__(package_name)
        package_installed = True
    except ImportError:
        package_installed = False

    if package_installed:
        pytest.skip(f"{package_name} 설치됨 — 실제 API 연동 후 별도 통합 테스트 필요")

    with pytest.raises(RuntimeError):
        wrapper.index(str(sample_corpus), scope="subset")
