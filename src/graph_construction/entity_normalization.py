"""엔티티 정규화(별칭 통합) — spec.md Phase 3.2 대응, TODO_mac.md #7.

교사/학생 모델이 뽑은 triple의 entity1/entity2 표기가 문서마다 다를 수 있다
(예: "Steve Jobs" vs "Jobs" vs "steve jobs"). 여기서는 대소문자/공백 정규화 +
같은 canonical key를 공유하는 표기들의 병합만 다룬다 — 임베딩 기반 유사도 병합
(다른 개체를 잘못 묶는 과묶임 방지, spec.md Phase 3.2-b2)은 이 프로토타입의
스코프 밖.
"""

from __future__ import annotations


def _canonical_key(name: str) -> str:
    return " ".join(name.strip().lower().split())


def normalize_entities(triples: list[dict]) -> tuple[list[dict], dict[str, str]]:
    """triple 리스트의 entity1/entity2를 정규화된 대표 표기로 치환.

    반환: (정규화된 triple 리스트, {원래 표기: 대표 표기} 별칭 맵).
    같은 canonical key(대소문자·공백 무시)를 가진 표기 중 가장 긴 표기를 대표로
    삼는다 — 약어보다 완전한 이름이 정보량이 많다는 단순 휴리스틱(예: "Jobs"/
    "steve jobs"/"Steve Jobs" → "Steve Jobs").
    """
    canonical_variants: dict[str, set[str]] = {}
    for t in triples:
        for name in (t["entity1"], t["entity2"]):
            canonical_variants.setdefault(_canonical_key(name), set()).add(name)

    alias_map: dict[str, str] = {}
    for variants in canonical_variants.values():
        representative = max(variants, key=len)
        for v in variants:
            alias_map[v] = representative

    normalized = [
        {**t, "entity1": alias_map[t["entity1"]], "entity2": alias_map[t["entity2"]]}
        for t in triples
    ]
    return normalized, alias_map
