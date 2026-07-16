"""UltraDomain 원문 코퍼스 → 500~1,000자 청크 분할 — TODO_mac.md #6 (서브1 대응).

입력: UltraDomain 도메인별 jsonl (예: `TommyChien/UltraDomain`의 `mix.jsonl`).
각 라인은 {"input": 질문, "context": 원문 문서} 형태이고, 여러 라인이 같은 문서를
공유할 수 있어 `context` 기준으로 문서를 중복 제거한 뒤 청크로 나눈다.

청크 크기 상한은 Phase 0.0-e VRAM 안전장치(spec.md §2)와 동일하게 500~1,000자.
문서의 마지막 청크는 자연히 500자 미만일 수 있다(문서 끝에서 남는 조각) — 이는
허용되는 예외이며 분포 검증 시 별도로 취급한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_MIN_CHARS = 500
DEFAULT_MAX_CHARS = 1000


def load_unique_documents(input_path: str | Path) -> list[str]:
    """UltraDomain jsonl에서 `context` 필드 기준으로 중복 제거한 문서 텍스트 리스트."""
    seen: dict[str, None] = {}
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ctx = rec["context"]
            seen.setdefault(ctx, None)
    return list(seen.keys())


def chunk_text(
    text: str, min_chars: int = DEFAULT_MIN_CHARS, max_chars: int = DEFAULT_MAX_CHARS
) -> list[str]:
    """텍스트를 500~1,000자 청크로 분할. 단어 경계에서 자르고, 마지막 청크가
    너무 짧으면(min_chars 미만) 직전 청크와 합친다(합쳐도 max_chars 이내일 때만 —
    상한은 VRAM 안전장치의 실제 제약이라 하한 보정보다 우선).
    """
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            break_point = text.rfind(" ", start + min_chars, end)
            if break_point == -1:
                break_point = end
            end = break_point
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end

    if len(chunks) >= 2 and len(chunks[-1]) < min_chars:
        merged = (chunks[-2] + " " + chunks[-1]).strip()
        # 병합해도 상한(max_chars)을 넘지 않을 때만 병합 — VRAM 안전장치(spec.md §2)의
        # 실제 제약은 하한이 아니라 상한이므로, 못 넘기면 짧은 꼬리 청크를 그대로 둔다.
        if len(merged) <= max_chars:
            chunks[-2:] = [merged]

    return chunks


def prepare_domain(
    input_path: str | Path,
    domain: str,
    min_chars: int = DEFAULT_MIN_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[dict]:
    """도메인 jsonl 파일 하나를 청크 레코드 리스트로 변환.

    각 레코드: {chunk_id, doc_id, domain, text, char_len} — spec.md §1의
    고정 평가 서브셋 스키마(chunk_id, domain 필드)와 호환.
    """
    documents = load_unique_documents(input_path)
    records = []
    for doc_idx, doc_text in enumerate(documents):
        doc_id = f"{domain}_{doc_idx:03d}"
        for chunk_idx, chunk in enumerate(chunk_text(doc_text, min_chars, max_chars)):
            records.append(
                {
                    "chunk_id": f"{doc_id}_c{chunk_idx:04d}",
                    "doc_id": doc_id,
                    "domain": domain,
                    "text": chunk,
                    "char_len": len(chunk),
                }
            )
    return records


def write_jsonl(records: list[dict], output_path: str | Path) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="UltraDomain 도메인 jsonl 경로")
    parser.add_argument("--domain", required=True, help="도메인 이름 (예: mix)")
    parser.add_argument("--out", required=True, help="출력 jsonl 경로")
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    args = parser.parse_args()

    records = prepare_domain(args.input, args.domain, args.min_chars, args.max_chars)
    write_jsonl(records, args.out)

    lens = [r["char_len"] for r in records]
    in_range = sum(1 for l in lens if args.min_chars <= l <= args.max_chars)
    print(f"문서 수(중복제거): {len({r['doc_id'] for r in records})}")
    print(f"청크 수: {len(records)}")
    print(f"범위 내({args.min_chars}~{args.max_chars}자): {in_range}/{len(records)}")
    print(f"범위 밖(주로 문서 끝 마지막 청크): {len(records) - in_range}/{len(records)}")


if __name__ == "__main__":
    main()
