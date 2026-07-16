"""scripts/prepare_corpus.py 유닛테스트 — TODO_mac.md #6."""

import json

import pytest

from scripts.prepare_corpus import chunk_text, load_unique_documents, prepare_domain


def test_chunk_text_short_input_single_chunk():
    text = "짧은 문서입니다." * 10  # 500자 미만
    chunks = chunk_text(text, min_chars=500, max_chars=1000)
    assert len(chunks) == 1


def test_chunk_text_never_exceeds_max_chars():
    text = " ".join(f"word{i}" for i in range(2000))  # 충분히 긴 텍스트
    chunks = chunk_text(text, min_chars=500, max_chars=1000)
    assert all(len(c) <= 1000 for c in chunks)


def test_chunk_text_merges_short_tail_when_it_fits():
    # 900자 청크 하나 + 50자 꼬리 → 합쳐도 1000자 이내라 병합돼야 함
    text = ("a" * 900) + " " + ("b" * 50)
    chunks = chunk_text(text, min_chars=500, max_chars=1000)
    assert len(chunks) == 1
    assert len(chunks[0]) <= 1000


def test_chunk_text_leaves_short_tail_when_merge_would_exceed_max():
    # 900자 청크 하나 + 900자 꼬리 → 합치면 1000자를 넘으므로 병합하면 안 됨
    text = ("a" * 900) + " " + ("b" * 900)
    chunks = chunk_text(text, min_chars=500, max_chars=1000)
    assert len(chunks) == 2
    assert all(len(c) <= 1000 for c in chunks)


def test_chunk_text_empty_input():
    assert chunk_text("", min_chars=500, max_chars=1000) == []
    assert chunk_text("   ", min_chars=500, max_chars=1000) == []


def test_load_unique_documents_dedups_shared_context(tmp_path):
    path = tmp_path / "domain.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"input": "Q1", "context": "doc A text"}),
                json.dumps({"input": "Q2", "context": "doc A text"}),
                json.dumps({"input": "Q3", "context": "doc B text"}),
            ]
        ),
        encoding="utf-8",
    )
    docs = load_unique_documents(path)
    assert docs == ["doc A text", "doc B text"]


def test_prepare_domain_produces_expected_fields(tmp_path):
    path = tmp_path / "domain.jsonl"
    long_text = " ".join(f"word{i}" for i in range(400))  # 500자 초과 보장
    path.write_text(json.dumps({"input": "Q", "context": long_text}), encoding="utf-8")

    records = prepare_domain(path, domain="testdomain", min_chars=500, max_chars=1000)
    assert len(records) >= 1
    for i, rec in enumerate(records):
        assert rec["domain"] == "testdomain"
        assert rec["doc_id"] == "testdomain_000"
        assert rec["chunk_id"] == f"testdomain_000_c{i:04d}"
        assert rec["char_len"] == len(rec["text"])
        assert rec["char_len"] <= 1000
