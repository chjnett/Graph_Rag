# TODO_mac — 맥(GPU 없음) 전용 작업 목록

> 컨텍스트: 윈도우(RTX 3090) 원격 연결 끊김, 복구까지 약 1주일 예상.
> 이 문서는 그 기간 동안 GPU 없이 맥에서 진행할 서브프로젝트4 작업만 다룬다.
> `sub4-mac-noGPU` 브랜치에서 작업. 원본 마일스톤 정의는 `TODO.md`/`spec.md` 참고.
> 우선순위는 난이도(쉬운 것부터) 기준. 항목 간 의존관계 없음 — 순서는 참고용, 막히면 다음 항목으로 넘어가도 됨.
> 범위 밖(제외): 클라우드 GPU 대여를 통한 `throughput_pilot` 실행 — 이건 별도 결정/트래킹으로 관리.
> 5~7번은 원래 서브4가 아니라 서브1/서브3 소관이지만, "GPU로 만든 산출물이 없어도 되는" 항목이라 이번 주에 한해 여기 포함. GPU 복구 후 각 서브프로젝트 문서로 옮겨서 정합성 확인할 것.

## 1. spaCy 스킵 테스트 해결 (Milestone 2 보충) — ✅ 완료 (2026-07-16)

- [x] `.venv` 생성 + `pip install -r requirements.txt` + `pip install spacy` + `python -m spacy download en_core_web_sm`
- [x] Done when: `tests/test_baseline_contracts.py::test_deps_parsing_wrapper_end_to_end`가 skip 없이 pass — 10/10 pass 확인

## 1-2. `deps_parsing_wrapper.py` 논문(arXiv:2507.03226) 충실도 개선 — ✅ 완료 (2026-07-16)

> spec.md §9-2에 논문 아키텍처 정리 완료. 코드는 `baselines/deps_parsing_wrapper.py`.

- [x] 전처리: 동사구 없는 문장 필터링 (spaCy `sent`에 `VERB` 토큰 없으면 skip)
- [x] 후처리: 수동태 정규화 (`nsubjpass` + `agent`(by-구) 있을 때만 능동태로 swap, 없으면 스킵)
- [x] 후처리: 다중토큰 엔티티 병합 (`noun_chunks` 기반으로 subj/obj를 명사구 전체로 교체)
- [x] 후처리: 짧은 엔티티(2자 미만)·불용어 제거
- [x] 엔티티 타입: 모든 노드에 `type="Concept"` 속성 부여
- [x] 샘플 문장(능동태 "SAP launched Joule for Consultants.", 수동태 "The new product was developed by the engineering team.")으로 직접 실행 확인 — 수동태가 `(the engineering team) --[develop]--> (The new product)`로 정상 정규화됨
- [x] `tests/` 전체 43개 pass 유지 확인
- [ ] (범위 밖, 명시적 스킵) coreference resolution — 별도 라이브러리(fastcoref 등) 필요, 이번 재현에서 구현 안 함
- [ ] (후속, 🔒) 논문 보고 수치(Context Precision 61.07%, Semantic Alignment 61.87%, Full Coverage 51.08%) 대비 재현 결과 비교 스크립트 — 서브1 골드셋 필요해 지금은 스킵

- [ ] `tests/test_metrics.py` — 빈 `QAResult` 리스트, 정답 없음 등 경계값 케이스 추가
- [ ] `tests/test_benchmark.py` — 빈 `methods` 리스트, `indexing_llm_calls` 위반 케이스 추가 확인
- [ ] `tests/test_throughput_pilot.py` — 0건/1건 샘플링 등 경계값 케이스 추가
- [ ] Done when: 추가한 케이스 포함해 전체 테스트 스위트 pass

## 3. Milestone 5/6 스크립트 골격 + 목데이터 테스트 선작성

- [ ] `4.1-a` GraphRAG-Bench 난이도 분리 스크립트 — 함수 시그니처 + 목데이터 유닛테스트
- [ ] `4.4-a/b` GPU-시간 집계 스크립트 — 함수 시그니처 + 목데이터 유닛테스트
- [ ] `4.5-a/b` Table 초안 / 난이도별 성능 분해 그래프 스크립트 — 함수 시그니처 + 목데이터 유닛테스트
- [ ] Done when: 각 스크립트 최소 1개 이상 유닛테스트 pass (실제 데이터 연동은 여전히 🔒, 서브1/서브3 산출물 필요)

## 4. `ms_graphrag_wrapper.py` / `lightrag_wrapper.py` 실연동 (가장 오래 걸림)

- [ ] `pip install graphrag lightrag-hku`, 설치된 버전의 실제 config/API 확인
- [ ] 로컬 더미 HTTP 서버(OpenAI 호환 응답 흉내) 작성 — 실제 vLLM 없이 배선만 검증
- [ ] `ms_graphrag_wrapper.py`의 `index()`/`query()`를 실제 API 호출로 구현 (더미 서버 대상)
- [ ] `lightrag_wrapper.py`의 `index()`/`query()`를 실제 API 호출로 구현 (더미 서버 대상)
- [ ] `tests/test_baseline_contracts.py::test_gpu_backed_wrappers_fail_clearly_without_package`를 "설치 시 더미 서버로 정상 동작" 케이스까지 확장
- [ ] Done when: 두 wrapper 모두 `NotImplementedError` 제거, 더미 서버 대상 `index()`+`query()` 성공. 실제 vLLM(GPU) 대상 검증은 원격 연결 복구 후 별도 확인

## 5. 원 논문 앵커 수치 수집 (서브3 Phase 3.6-c 대응, 가장 쉬움)

- [ ] MS GraphRAG / LightRAG 원 논문에서 GPT-4(o) 기준 보고 정확도·비용 수치 정리
- [ ] `configs/eval.yaml`의 `original_paper_anchor_path`(`reports/sub3_phase3_6c_anchor.json`) 스키마에 맞춰 초안 작성
- [ ] Done when: 두 baseline 논문의 anchor 수치가 JSON으로 정리되고 출처(논문/표 번호) 주석 포함

## 6. INPUT — 코퍼스 준비 (서브1 대응)

- [ ] UltraDomain 등 원문 코퍼스 확보 경로 확인(HuggingFace/공식 repo 등) 및 다운로드
- [ ] 500~1,000자 청크 분할 스크립트 작성 (Phase 0.0-e 청크 크기 상한과 동일 기준)
- [ ] Done when: 최소 1개 도메인 문서 세트가 청크 단위 JSONL로 로컬에 존재, 청크 크기 분포가 500~1,000자 범위 내

## 7. INDEX — LLM-free 그래프 구축 프로토타입 (서브3 대응, 가장 오래 걸림)

- [ ] spec.md §4 triple 스키마(`entity1, entity1_type, relation, entity2, entity2_type, source_span, confidence`)를 따르는 목(mock) triple 세트 작성 — 교사 모델 출력 없이 프로토타입 목적
- [ ] Leiden 커뮤니티 탐지 구현/연동 (예: `python-igraph` + `leidenalg`, 또는 `networkx` 대안)
- [ ] 엔티티 정규화(별칭 통합) 로직 작성
- [ ] TextRank 기반 커뮤니티 요약 구현
- [ ] Done when: 목 triple 세트를 입력으로 그래프(NetworkX pickle 등)가 만들어지고, 인덱싱 과정에 LLM 호출이 0회임을 테스트로 증명 (spec.md §4 `IndexStats.llm_calls == 0`)

---

## GPU 복구 후 처리할 것 (참고용, 지금 실행 안 함)

- 위 4개 항목 결과를 `main`으로 merge하기 전, `TODO.md`의 해당 체크박스 상태와 충돌 없는지 확인
- `ms_graphrag_wrapper.py`/`lightrag_wrapper.py`를 실제 vLLM 엔드포인트 대상으로 재검증
