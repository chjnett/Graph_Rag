# TODO_mac — 맥(GPU 없음) 전용 작업 목록

> 컨텍스트: 윈도우(RTX 3090) 원격 연결 끊김, 복구까지 약 1주일 예상.
> 이 문서는 그 기간 동안 GPU 없이 맥에서 진행할 서브프로젝트4 작업만 다룬다.
> `sub4-mac-noGPU` 브랜치에서 작업. 원본 마일스톤 정의는 `TODO.md`/`spec.md` 참고.
> 우선순위는 난이도(쉬운 것부터) 기준. 항목 간 의존관계 없음 — 순서는 참고용, 막히면 다음 항목으로 넘어가도 됨.
> 범위 밖(제외): 클라우드 GPU 대여를 통한 `throughput_pilot` 실행 — 이건 별도 결정/트래킹으로 관리.
> 5~7번은 원래 서브4가 아니라 서브1/서브3 소관이지만, "GPU로 만든 산출물이 없어도 되는" 항목이라 이번 주에 한해 여기 포함. GPU 복구 후 각 서브프로젝트 문서로 옮겨서 정합성 확인할 것.
>
> **✅ 1~7번 전부 완료 (2026-07-16).** 남은 건 파일 하단 "GPU 복구 후 처리할 것" 뿐 — 원격 연결이 돌아오면 그것부터 확인.

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

## 2. 기존 테스트 엣지케이스 보강 (전 Milestone 공통) — ✅ 완료 (2026-07-16)

- [x] `tests/test_metrics.py` — 빈 `QAResult`/예측·정답 리스트, `compare_costs({})`, `hallucination_rate([], {})` 등 경계값 케이스 추가
- [x] `tests/test_benchmark.py` — `sequential_run({}, ...)` 빈 methods 케이스 추가. `indexing_llm_calls` 위반 케이스는 기존 파라미터화 테스트로 이미 커버됨을 확인
- [x] `tests/test_throughput_pilot.py` — `sample_docs(n=0)`, `sample_docs(n=1)` 경계값 케이스 추가
- [x] Done when: 추가한 케이스 포함해 전체 테스트 스위트 pass — 57/57 pass

## 3. Milestone 5/6 스크립트 골격 + 목데이터 테스트 선작성 — ✅ 완료 (2026-07-16)

- [x] `4.1-a` GraphRAG-Bench 난이도 분리 — `src/eval/difficulty_split.py`(`split_by_difficulty`), 난이도명은 원 논문 앵커 조사(arXiv:2506.05690) 때 확인한 4종(fact_retrieval/complex_reasoning/contextual_summarization/creative_generation) 그대로 사용. `tests/test_difficulty_split.py`(4개)
- [x] `4.4-a/b` GPU-시간 집계 — `src/eval/metrics_cost.py`에 `aggregate_ours_gpu_hours`(서브1+2+3 단계 합산)·`compare_total_costs`(baseline 단일 index()와 동일 단위 비교) 추가. `tests/test_metrics.py`에 3개 추가
- [x] `4.5-a/b` Table 초안 / 난이도별 성능 분해 — `src/eval/report_table.py`(`build_markdown_table`, `group_by_difficulty`). 실제 그래프 렌더링(matplotlib 등)은 스코프 밖 — 그룹화된 데이터까지만 제공. `tests/test_report_table.py`(4개)
- [x] Done when: 각 스크립트 최소 1개 이상 유닛테스트 pass — 68/68 전체 pass (실제 데이터 연동은 여전히 🔒, 서브1/서브3 산출물 필요)

## 4. `ms_graphrag_wrapper.py` / `lightrag_wrapper.py` 실연동 — ✅ 완료 (2026-07-16)

> ⚠️ `graphrag`(3.x)는 Python 3.14를 지원하지 않아 `.venv`를 python3.12로 재생성함(requirements.txt에 메모).

- [x] `pip install graphrag lightrag-hku` — graphrag 3.1.0, lightrag-hku 1.5.4 설치 확인
- [x] 로컬 더미 HTTP 서버 — `tests/fixtures/dummy_llm_server.py`. graphrag는 프롬프트 내용(entity 추출/community report 마커)에 따라 형식이 맞는 응답을 골라줘야 함(그냥 아무 문자열이면 `ValueError`/`JSONSchemaValidationError`로 죽음) — 실제로 부딪혀서 알아낸 것들:
  - entity 추출 실패 시 즉시 `ValueError("No entities detected")`로 하드 크래시(개별 문서 단위는 관대해도 전체가 0건이면 크래시)
  - community_reports는 litellm이 구조화 출력 스키마(JSON) 검증까지 함 — 형식 안 맞으면 `JSONSchemaValidationError`
  - `graphrag.api.build_index()`는 CLI가 자동으로 해주는 `validate_config_names()`(연결성 확인 + 임베딩 차원 자동 동기화)를 안 해줌 — 직접 호출 안 하면 임베딩 차원 불일치로 벡터스토어 적재가 죽음
  - LightRAG는 자체 콜백 구조(`llm_model_func`/`embedding_func`)라 형식에 훨씬 관대함 — 더미 응답 그대로 통과
- [x] `ms_graphrag_wrapper.py` 실연동 — `graphrag.cli.initialize.initialize_project_at` + `graphrag.config.load_config` + `graphrag.api.build_index`/`local_search`. LLM 호출 수는 litellm 콜백(`.callbacks`/`.success_callback`) 둘 다 실제로는 호출 안 되는 걸 확인해서, 대신 `litellm.completion/acompletion/embedding/aembedding`을 직접 monkeypatch해서 셈
- [x] `lightrag_wrapper.py` 실연동 — `lightrag.LightRAG` + `openai_complete_if_cache`/`openai_embed` 콜백 주입. 자체 콜백 구조라 LLM 호출 수는 monkeypatch 없이 직접 카운트
- [x] `tests/test_gpu_backed_wrappers_integration.py` 신규 작성(2개) — 기존 `test_baseline_contracts.py`의 "미설치 시 실패" 테스트는 그대로 두고, "설치 시 정상 동작"은 별도 통합테스트 파일로 분리(관심사 분리가 명확해서 이 편이 낫다고 판단)
- [x] Done when: 두 wrapper 모두 `NotImplementedError` 제거, 더미 서버 대상 `index()`+`query()` 성공 — 확인됨(전체 70개 중 68 pass + 2 skip, skip은 패키지가 이제 설치돼서 "미설치" 테스트가 정상적으로 건너뛰는 것). 실제 vLLM(GPU) 대상 검증은 원격 연결 복구 후 별도 확인

## 5. 원 논문 앵커 수치 수집 (서브3 Phase 3.6-c 대응) — ✅ 완료 (2026-07-16)

- [x] MS GraphRAG / LightRAG 원 논문 조사 — **설계 재검토 발견**: 두 원 논문 다 EM/F1이 아니라 GPT-4(o) 심사 기반 LLM-as-judge 승률(comprehensiveness/diversity/empowerment)을 보고, 평가 코퍼스도 우리와 다름(팟캐스트·뉴스 / UltraDomain 일부)
- [x] 대안 발견 및 채택: GraphRAG-Bench 자체 논문(arXiv:2506.05690)이 같은 GraphRAG-Bench 벤치마크로 MS GraphRAG/LightRAG를 accuracy(%)로 재평가한 Table 2 — 이걸 1차 앵커로 사용
- [x] `reports/sub3_phase3_6c_anchor.json` 작성 — GraphRAG-Bench Table 2 수치(1차 앵커) + 두 원 논문 win-rate(qualitative_notes, 참고용) + 출처/판별 근거 전부 명시
- [x] `.gitignore`의 `reports/` 패턴이 이 파일까지 무시하는 문제 발견 → `reports/*` + `!reports/sub3_phase3_6c_anchor.json` 예외로 수정
- [ ] (후속) hotpotqa/multihop_rag용 앵커는 못 찾음 — 두 원 논문 다 이 데이터셋 평가 안 함, 추가 조사 필요(JSON에 `_todo`로 기록)
- [x] Done when: anchor 수치가 JSON으로 정리되고 출처(논문/표 번호) 주석 포함 — 단, 수치는 arXiv HTML 파싱 기반이라 인용 확정 전 PDF 원문 Table 2 대조 재확인 필요(JSON `verification_needed` 필드에 기록)

## 6. INPUT — 코퍼스 준비 (서브1 대응) — ✅ 완료 (2026-07-16)

- [x] UltraDomain 확보 경로 확인 — HuggingFace `TommyChien/UltraDomain`, 18개 도메인 jsonl(agriculture/art/.../technology). 크기가 가장 작은 `mix.jsonl`(5.7MB, 61개 문서, LightRAG 논문 Mix 서브셋과 일치)을 `data/raw/mix.jsonl`로 다운로드
- [x] `scripts/prepare_corpus.py` 작성 — `context` 필드 기준 문서 중복 제거 → 500~1,000자 청크 분할. 상한(1,000자)이 VRAM 안전장치의 실제 제약이라 하한 미달보다 우선 — 병합해도 상한을 넘으면 병합하지 않고 짧은 꼬리 청크를 그대로 둠
- [x] `tests/test_prepare_corpus.py` 작성(7개, 전부 pass) — 상한 미준수, 병합 경계, 빈 입력, 중복 제거 케이스 포함
- [x] Done when: mix 도메인 61개 문서 → 2,676개 청크로 `data/processed/mix_chunks.jsonl` 생성, 2,642/2,676(98.7%)이 500~1,000자 범위 내, 나머지 34개는 전부 문서 끝 500자 미만 꼬리 청크(허용된 예외, 상한 위반 0건)

## 7. INDEX — LLM-free 그래프 구축 프로토타입 (서브3 대응) — ✅ 완료 (2026-07-16)

- [x] 목(mock) triple 세트 작성 — `tests/fixtures/mock_triples.json`(spec.md §1 triple 스키마 준수, 교사 모델 출력 없이 손으로 작성). triple 6개, 노드 8개, 서로 무관한 두 클러스터(Apple/Steve Jobs 쪽, Wimbledon 쪽)로 구성해 커뮤니티 탐지가 실제로 나뉘는지 검증 가능하게 함
- [x] Leiden 커뮤니티 탐지 — `src/graph_construction/community_detection.py`. `python-igraph`+`leidenalg` 설치해서 진짜 Leiden 사용, 둘 중 하나라도 없으면 `networkx.community.louvain_communities`로 자동 대체(ImportError 캐치)
- [x] 엔티티 정규화 — `src/graph_construction/entity_normalization.py`. 대소문자/공백만 다른 표기를 canonical key로 병합(예: "steve jobs" → "Steve Jobs"), 완전히 다른 표기(약어 등) 병합은 스코프 밖으로 명시
- [x] TextRank 기반 커뮤니티 요약 — `src/graph_construction/community_summary.py`. TF-IDF(scikit-learn) + `networkx.pagerank`만 사용, LLM 호출 없음
- [x] `src/graph_construction/build_graph.py` — 위 세 모듈을 엮은 오케스트레이션(`build_graph`/`summarize_communities`/`index_from_triples`). `GraphRAGMethod.query()`(검색, Phase 3.35)는 이 프로토타입 스코프 밖 — 실제 학생 모델 triple이 있어야 의미 있어서 서브3 자체 작업으로 남김
- [x] 테스트 15개 신규(`test_entity_normalization.py`, `test_community_detection.py`, `test_community_summary.py`, `test_build_graph.py`) — 전체 85개 중 83 pass + 2 skip
- [x] Done when: 목 triple 세트를 입력으로 그래프가 만들어지고, 인덱싱 과정에 LLM 호출이 0회임을 테스트로 증명 — `test_index_from_triples_llm_calls_always_zero`로 확인, 커뮤니티도 실제로 2개로 분리됨 확인

---

## GPU 복구 후 처리할 것 (참고용, 지금 실행 안 함)

- 위 4개 항목 결과를 `main`으로 merge하기 전, `TODO.md`의 해당 체크박스 상태와 충돌 없는지 확인
- `ms_graphrag_wrapper.py`/`lightrag_wrapper.py`를 실제 vLLM 엔드포인트 대상으로 재검증
