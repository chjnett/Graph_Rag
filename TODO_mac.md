# TODO_mac — 맥(GPU 없음) 전용 작업 목록

> 컨텍스트: 윈도우(RTX 3090) 원격 연결 끊김, 복구까지 약 1주일 예상.
> 이 문서는 그 기간 동안 GPU 없이 맥에서 진행할 서브프로젝트4 작업만 다룬다.
> `sub4-mac-noGPU` 브랜치에서 작업. 원본 마일스톤 정의는 `TODO.md`/`spec.md` 참고.
> 우선순위는 난이도(쉬운 것부터) 기준. 항목 간 의존관계 없음 — 순서는 참고용, 막히면 다음 항목으로 넘어가도 됨.
> 범위 밖(제외): 클라우드 GPU 대여를 통한 `throughput_pilot` 실행 — 이건 별도 결정/트래킹으로 관리.

## 1. spaCy 스킵 테스트 해결 (Milestone 2 보충)

- [ ] `pip install spacy` + `python -m spacy download en_core_web_sm`
- [ ] Done when: `tests/test_baseline_contracts.py::test_deps_parsing_wrapper_end_to_end`가 skip 없이 pass

## 2. 기존 테스트 엣지케이스 보강 (전 Milestone 공통)

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

---

## GPU 복구 후 처리할 것 (참고용, 지금 실행 안 함)

- 위 4개 항목 결과를 `main`으로 merge하기 전, `TODO.md`의 해당 체크박스 상태와 충돌 없는지 확인
- `ms_graphrag_wrapper.py`/`lightrag_wrapper.py`를 실제 vLLM 엔드포인트 대상으로 재검증
