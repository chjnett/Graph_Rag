# TODO — 서브프로젝트 4 개발 착수용 (spec.md 기준)

> 체크박스 = 실제 코딩 단위. (S)/(M)/(L) = 원문서(`graphrag_04_evaluation.md`) 사이즈 태그 그대로.
> "🔓 즉시 가능" = 다른 서브프로젝트 산출물 없이 지금 시작 가능. "🔒 블록" = 명시된 산출물이 있어야 실행(코딩은 미리 가능, 실행/검증은 블록).

## Milestone 0 — 인프라 확인 (🔓, 최우선)

- [ ] Phase 0.0 vLLM 엔드포인트가 이미 떠 있는지 확인 (`graphrag_00` 담당, 서브4는 소비만)
- [ ] `configs/eval.yaml` 스킵톤 작성 (spec.md §7 스키마 그대로) — `corpus_scope: subset` 기본값으로 시작
- [ ] `GraphRAGMethod` / `QAResult` / `Evidence` / `IndexStats` 데이터클래스 구현 (spec.md §4) — 이후 모든 wrapper·metrics가 이 계약에 의존

## Milestone 1 — Phase 0.5-a0: 처리량 사전 체크 (🔓, 가장 리스크 큰 항목)

- [ ] (S) 1단계: UltraDomain 무작위 3개 문서로 MS GraphRAG 표준 인덱싱을 로컬 엔드포인트로 실행, OOM/설정 문제 확인
- [ ] (S) 1단계 통과 시 2단계: 10~20개 문서로 확장, 문서당 소요시간·LLM 호출 수 로깅
- [ ] (S) LightRAG도 동일 절차로 별도 측정
- [ ] (S) 428개 전체 외삽 → 예상 소요일수 산출, `throughput_pilot` 결과를 `configs/eval.yaml`에 기록
- [ ] **의사결정**: baseline당 외삽치가 2주 초과 → `corpus_scope: full` → `subset`으로 변경, 그렇지 않으면 `full` 유지
- [ ] Done when: 두 baseline의 처리량 벤치마크 리포트 + `corpus_scope` 확정값이 `eval.yaml`에 반영됨

## Milestone 2 — Phase 0.5: Baseline wrapper 5종 + 공통 인터페이스 (🔓)

- [ ] (L) `baselines/ms_graphrag_wrapper.py` — 설치, 인덱싱 LLM 호출 지점을 로컬 엔드포인트로 교체, `GraphRAGMethod` 구현
- [ ] (M) `baselines/lightrag_wrapper.py` — 동일하게 로컬 엔드포인트 연동
- [ ] (M) `baselines/litesemrag_wrapper.py` — LLM-free 계열, `indexing_llm_calls=0` 하드코딩 후 유닛테스트로 고정
- [ ] (M) `baselines/deps_parsing_wrapper.py` — arXiv:2507.03226 코드 공개 여부 먼저 확인 (spec.md §9-2 참고), 없으면 논문 기술 기반 최소 재현으로 스코프 축소하고 그 사실을 README에 기록
- [ ] (M) NoLLMRAG 재현 — wrapper 파일 위치 결정(별도 파일 or litesemrag_wrapper.py에 인접)
- [ ] (S) 5개 wrapper 전부 `GraphRAGMethod.query()` 반환 타입이 `QAResult`인지 확인하는 계약 테스트 작성
- [ ] Done when: 5개 baseline 전부 샘플 질의 1건에 정상 응답 + `indexing_llm_calls` 값이 각 방법론 특성과 일치(0 또는 >0)

## Milestone 3 — `src/eval/metrics_*.py` 구현 (🔓, 목 데이터로 선행 가능)

- [ ] (S) `metrics_cost.py` — `IndexStats` 리스트 → GPU-시간 비교 테이블, (선택) API 환산 비용 병기
- [ ] (M) `metrics_coverage.py` — `graphrag_03` Phase 3.7 일치율 리포트를 읽어 그대로 노출 (재계산 금지, 순환논증 지표는 서브3 소관)
- [ ] (M) `metrics_gold_accuracy.py` — 동일하게 서브3 Phase 3.7 골드셋 정확도 리포트 소비
- [ ] (M) `metrics_qa.py` — `QAResult` 리스트 + 정답 → EM/F1, LLM-as-judge 옵션, 원 논문 앵커 컬럼 병합
- [ ] (M) `metrics_hallucination.py` — `Evidence.source_span`을 원문과 대조해 환각률 계산
- [ ] 각 스크립트에 목(mock) `QAResult`/`IndexStats` 기반 유닛테스트 작성 — 서브1/서브3 실제 데이터 없이도 통과해야 함

## Milestone 4 — `benchmark.py` 통합 러너 (🔓 골격 / 🔒 실제 실행)

- [ ] (M) `configs/eval.yaml` 파싱 → `methods` 순차 실행 루프 (동시 실행 금지, VRAM 경합 방지)
- [ ] (S) 결과를 spec.md §8 JSON 스키마로 직렬화
- [ ] (S) `gold_accuracy`/`original_paper_anchor`가 우리 방법에만 채워지고 baseline에는 null인지 검증하는 테스트
- [ ] (S) `indexing_llm_calls != 0`인데 LLM-free 계열(litesemrag/nollmrag/ours)로 표시된 경우 assert로 실패시키는 가드 추가
- [ ] 🔒 실제 실행은 Milestone 1~3 + `graphrag_03` Phase 3.35/3.7 산출물 필요

## Milestone 5 — Phase 4.1~4.3: QA 벤치마크 실행 (🔒)

- [ ] (S) `4.1-a` GraphRAG-Bench 4단계 난이도 질문 분리 스크립트
- [ ] (M) `4.1-b` 전체 baseline + 우리 방법 순차 실행, 채점 기준(EM/F1 또는 LLM-as-judge) 확정
- [ ] (M) `4.1-c` 다중 시드(≥3) 또는 부트스트랩으로 신뢰구간 산출
- [ ] (M) `4.2-a/b` HotpotQA supporting facts 대비 인용 정확도 계산 + 실행
- [ ] (M) `4.3-a` MultiHop-RAG 뉴스 도메인 실행, UltraDomain 내부 성능과 격차 계산
- [ ] 블로킹: `graphrag_03` Phase 3.35(검색 인터페이스), Phase 3.7(일치율/실정확도) 완료 필요

## Milestone 6 — Phase 4.4~4.5: 비용 집계 & 종합 Table (🔒)

- [ ] (S) `4.4-a/b` 우리 파이프라인(서브1~3) + baseline(Phase 0.5) GPU-시간 집계, 동일 단위 비교
- [ ] (S, 선택) `4.4-c` 참고용 API 환산 비용 병기
- [ ] (S) `4.5-a` 논문용 Table 초안 (정확도/GPU-시간/신뢰구간)
- [ ] (S) `4.5-b` 난이도별 성능 분해 그래프
- [ ] (S) `4.5-c` `graphrag_03` Phase 3.6-c 원 논문 앵커 열 병합 (삼각비교 완성)
- [ ] 블로킹: Milestone 5 완료 + `graphrag_03` Phase 3.6-c

---

## 지금 바로 시작할 수 있는 것 (요약)

1. Milestone 0 (데이터클래스/config 스켈레톤)
2. Milestone 1 (처리량 사전 체크 — **최우선, 전체 일정의 최대 리스크**)
3. Milestone 2 (baseline wrapper 뼈대)
4. Milestone 3 (metrics 함수, 목 데이터 테스트)

Milestone 4의 골격까지는 병행 가능하나, 실제 end-to-end 실행과 Milestone 5~6은 서브1(`graphrag_01`)·서브3(`graphrag_03`) 산출물이 나온 뒤로 자연히 순연된다.
