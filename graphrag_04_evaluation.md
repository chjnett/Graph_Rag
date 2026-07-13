# 서브프로젝트 4 — Baseline 재현 & 다운스트림 QA 평가

> 리소스 전제는 `graphrag_00_overview.md` 참고.

## 목표
우리 파이프라인 vs LLM 기반 GraphRAG/LightRAG 등 5종 baseline을 실제 QA 정확도·비용으로 비교한다.

## 데이터셋
GraphRAG-Bench (난이도별 질문), HotpotQA (supporting facts), MultiHop-RAG (도메인 일반화, 뉴스 도메인)

## 설계 개요 — 베이스라인
- 대상: Microsoft GraphRAG, LightRAG, LiteSemRAG(LLM-free, 임베딩 기반), Dependency-parsing 방식(arXiv:2507.03226), NoLLMRAG
- **인덱싱 LLM 통일**: Microsoft GraphRAG/LightRAG는 원래 OpenAI API를 호출하지만, 본 프로젝트는 API 비용을 낼 수 없으므로 `graphrag_00_overview.md` Phase 0.0과 동일한 **로컬 vLLM 서빙 모델**을 OpenAI 호환 엔드포인트로 연결해 인덱싱시킨다. 이렇게 하면 baseline 간 "비용" 비교가 $ API 요금이 아니라 GPU-시간/wall-clock 기준으로 공정하게 통일됨
- Baseline 재현은 **일정상 최우선 트랙**으로 1주차부터 서브1(합성데이터)과 병렬 착수 (외부 코드 재현이 가장 시간 예측이 어려운 작업이므로 조기에 리스크를 드러내야 함)
- **원 논문 수치 앵커링 (methodology 리뷰 반영)**: baseline을 로컬 LLM으로 재현하면 "GPU-시간 기준 공정 비교"는 되지만, 원래 GraphRAG/LightRAG 논문이 GPT-4 계열로 보고한 정확도 수치와는 직접 비교가 아니게 된다. 따라서 최종 결과 Table에는 (1) 우리 방법, (2) 로컬 LLM 재현 baseline, (3) 원 논문 보고 수치(가능한 범위 내 인용) 세 지점을 나란히 병기해, "로컬 재현이 원 논문 대비 얼마나 괴리되는지"와 "우리 방법이 그 안에서 어디에 위치하는지"를 독자가 함께 판단할 수 있게 한다

## 설계 개요 — 평가 지표
> **개정 (리뷰 반영)**: 기존 "그래프 커버리지" 단일 지표는 학생 모델의 학습 신호(교사 출력)와 평가 정답(교사 출력)이 동일해 순환논증 위험이 있었음. `graphrag_03_graph_construction.md` Phase 3.7에서 "교사-학생 일치율"과 "실제 정확도(인간 검수)"를 분리했다.

| 축 | 지표 |
|---|---|
| 비용 | 인덱싱 GPU-시간(wall-clock), (참고용으로만) 동일 작업을 유료 API로 했을 때의 추정 $ 비용 |
| 품질 (교사-학생 일치) | 교사-학생 그래프 일치율 — 고정 평가 서브셋(150~200청크)에 대한 recall. **주의: 학생이 교사를 얼마나 잘 모방했는가를 재는 지표이며, 그래프가 실제로 정확한가의 증명은 아님** (상세: `graphrag_03_graph_construction.md` Phase 3.7) |
| 품질 (실제 정확도) | 인간 검수 골드셋(30~50 triple) 대비 정확도 — 위 일치율과의 괴리가 교사 모델 편향의 실질적 증거 |
| 다운스트림 | Multi-hop QA 정확도, 인용 근거 일치율. **원 논문(Microsoft GraphRAG/LightRAG 등)이 GPT-4 계열로 보고한 공개 수치를 참고 앵커로 최종 Table에 병기** — baseline도 로컬 LLM으로 재현하므로, 이 앵커 없이는 "LLM 기반과 동등한 품질"이라는 주장이 "동일한 약한 교사를 쓴 두 방법 간 비교"로 축소될 위험이 있음 |
| 안전성 | 환각률 — 추출된 triple 중 원문 `source_span`에서 실제로 지지되지 않는 비율 (그래프 존재 여부가 아니라 원문 대조로 정의) |
| 적응력 | 범용 vs 도메인 적응 모델의 triple 성능 격차 + 다운스트림 QA 정확도 격차 (`graphrag_05_domain_adaptation.md` Phase 5.4에서 QA 하네스 재사용) |

> **도메인별 세부 분석 유의**: 18개 도메인에 걸친 도메인별 recall/성능 분포는 도메인당 표본이 10개 내외로 작아 확증적 결론이 아니라 **탐색적(exploratory) 해석**으로 한정한다.

---

## Phase 0.5 — Baseline 5종 재현 착수 (1주차부터 서브1과 완전 병렬)
- **0.5-a0** (S, 신규 — 처리량 사전 체크, 다른 sub-step보다 먼저 실행) UltraDomain에서 무작위 10~20개 문서로 Microsoft GraphRAG 표준 인덱싱(엔티티 추출+gleaning 반복+계층적 커뮤니티 요약)을 Phase 0.0 로컬 엔드포인트로 실행, 문서당 소요시간·총 LLM 호출 수 측정 → 428개 전체 외삽 시 예상 소요일수 산출. LightRAG도 동일 절차로 별도 측정(커뮤니티 계층 요약이 없어 상대적으로 가벼울 수 있음)
  - **의사결정 규칙**: 외삽 추정치가 baseline 1종당 **약 2주**를 넘으면, Phase 4.1~4.5의 QA 비교 코퍼스를 UltraDomain 전체 428개가 아니라 GraphRAG-Bench/HotpotQA/MultiHop-RAG 질문이 실제로 걸쳐 있는 서브셋으로 축소한다. `graphrag_03_graph_construction.md` Phase 3.6의 구조 통계 비교(원 논문 보고 수치 대비)는 이 축소와 무관하게 전체 코퍼스 기준으로 유지 — 그쪽은 baseline을 로컬 재현하는 게 아니라 원 논문 발표 수치와 비교하는 것이라 코퍼스 축소의 영향을 받지 않음. 우리 방법(경량 학생 모델)의 Phase 3.1 전체 코퍼스 추출은 이 규칙과 무관하게 그대로 진행(병목이 아님)
  - Done when: 두 baseline(MS GraphRAG, LightRAG) 각각의 문서당 처리시간 + 전체 외삽 추정치(일 단위) + 코퍼스 축소 여부 결정 확정
  - **산출물**: baseline 처리량 벤치마크 리포트 + 코퍼스 범위 결정 기록
- **0.5-a** (L) Microsoft GraphRAG 설치, 인덱싱 LLM 호출 지점을 Phase 0.0의 로컬 vLLM 엔드포인트로 교체 (config의 API base URL 변경)
- **0.5-b** (M) LightRAG 설치, 동일하게 로컬 엔드포인트 연동
- **0.5-c** (M) LiteSemRAG 재현 (LLM-free 계열, 임베딩 모델만 필요)
- **0.5-d** (M) dependency-parsing(arXiv:2507.03226) 재현 — 코드 공개 여부 먼저 확인, 없으면 논문 기술만으로 최소 재현
- **0.5-e** (M) NoLLMRAG 재현
- **0.5-f** (S) 공통 QA 인터페이스(질의→검색→응답 함수 시그니처) 정의, 5개 wrapper 모두 이 인터페이스로 감싸기
- Done when: 5개 baseline 모두 샘플 질의 1건에 정상 응답, 재현 중 발견된 리스크(의존성 충돌 등) 로그로 기록, 0.5-a0의 코퍼스 범위 결정이 이후 Phase 4.1 실행 계획에 반영됨
- **산출물**: 5개 baseline wrapper (로컬 LLM 연동 포함) + 공통 QA 인터페이스 코드 + 처리량 벤치마크 리포트
- **왜 최우선인가**: 외부 코드 재현은 의존성 충돌, 미문서화된 하이퍼파라미터, 청킹 방식 차이 등으로 가장 예측 불가능한 작업일 뿐 아니라, MS GraphRAG/LightRAG는 원래 청크당 추출+커뮤니티 계층 요약마다 LLM을 호출하는 구조라 단일 3090으로 전체 코퍼스를 인덱싱하면 **순수 GPU-시간 자체가 며칠~몇 주로 불어날 위험**이 있음(0.5-a0에서 조기 실측). `graphrag_03_graph_construction.md` 완료를 기다리면 문제 발견이 프로젝트 후반부로 밀려 전체 일정이 위태로워짐

## Phase 4.1 — GraphRAG-Bench 난이도별 평가
- **4.1-a** (S) 4단계 난이도(사실검색/복합추론/맥락요약/창의생성) 질문 분리 스크립트
- **4.1-b** (M) 전체 baseline(Phase 0.5 wrapper) + 우리 방법(`graphrag_03_graph_construction.md` Phase 3.35 검색 인터페이스) 순차 실행, 채점 기준(EM/F1 or LLM-as-judge) 확정
- **4.1-c** (M) 다중 시드(최소 3회) 실행 또는 부트스트랩으로 신뢰구간 산출
- Done when: 난이도별 성능 표 + 신뢰구간 완성
- **산출물**: 난이도별 성능 원시 결과 + 신뢰구간

## Phase 4.2 — HotpotQA 인용 정확도
- **4.2-a** (M) supporting facts 대비 근거 문단(검색된 서브그래프의 source_span) 일치율 계산 로직
- **4.2-b** (S) 실행 및 결과 집계
- Done when: 인용/근거 일치율 리포트
- **산출물**: 인용/근거 일치율 리포트

## Phase 4.3 — MultiHop-RAG 도메인 일반화
- **4.3-a** (M) 뉴스 도메인 실행, UltraDomain 내부 성능과 격차 계산
- Done when: 도메인 일반화 리포트
- **산출물**: 도메인 일반화 성능 리포트

## Phase 4.4 — 비용(GPU-시간) 측정
- **4.4-a** (S) 우리 파이프라인 전체(서브1~3) GPU-시간 집계
- **4.4-b** (S) baseline(Phase 0.5, 로컬 LLM 인덱싱) GPU-시간 집계, 동일 단위로 비교
- **4.4-c** (S, 선택) 참고용으로 동일 작업을 유료 API로 했을 경우의 추정 $ 비용 병기
- Done when: 비용 비교 원자료(raw numbers) 확보
- **산출물**: 비용 비교 데이터

## Phase 4.5 — 종합 테이블 & 그래프 작성
- **4.5-a** (S) 논문용 Table 초안 (정확도/GPU-시간 통합, 신뢰구간 포함)
- **4.5-b** (S) 난이도별 성능 분해 그래프 생성
- **4.5-c** (S, 삼각비교) `graphrag_03_graph_construction.md` Phase 3.6-c에서 정리한 baseline 원 논문 보고 수치를 참고 앵커 열로 최종 Table에 병기
- Done when: 논문에 바로 삽입 가능한 Table(원 논문 앵커 포함) + Figure 완성
- **산출물**: 최종 비교 Table(원 논문 앵커 포함) + 난이도별 성능 분해 그래프

> Phase 4.1~4.5는 `graphrag_03_graph_construction.md`(특히 3.35, 3.7) 완료 이후 실행. Baseline 자체 재현(Phase 0.5)은 1주차에 이미 끝나 있어야 한다.

---

## 코드 매핑

```
baselines/
├── ms_graphrag_wrapper.py
├── lightrag_wrapper.py
├── litesemrag_wrapper.py
└── deps_parsing_wrapper.py     # arXiv:2507.03226 재현
src/eval/
├── benchmark.py           # 통합 평가 러너 — 모든 baseline + 우리 방법을 동일 데이터셋에 순차 실행, 결과를 단일 테이블로 자동 집계
├── metrics_cost.py        # GPU-시간/wall-clock 측정 (+ 참고용 API 환산 비용)
├── metrics_coverage.py    # 고정 평가 서브셋 기준 교사-학생 일치율(agreement) 계산 — 실제 정확도 아님(순환논증 주의)
├── metrics_gold_accuracy.py  # 인간 검수 골드셋(30~50 triple) 대비 precision/recall — 순환논증 방지용 실제 정확도 지표
├── metrics_qa.py          # QA 정확도 (LLM-as-judge or EM/F1), 원 논문 보고 수치 앵커 병기
└── metrics_hallucination.py  # source_span 대조 기반 환각률 계산
```

`benchmark.py`는 골드셋 기반 실제 정확도(`metrics_gold_accuracy.py`)와 교사-학생 일치율(`metrics_coverage.py`)을 별도 컬럼으로 병기해 순환논증 여부를 항상 함께 확인 가능하게 출력하며, baseline 비교 Table에는 원 논문 보고 수치(가능한 범위) 컬럼을 추가해 삼각비교를 구성한다.

---

## 이 파트의 한계
- **Baseline 비교의 삼각점 한계**: 원 논문(GraphRAG/LightRAG 등)의 GPT-4 기반 보고 수치를 앵커로 병기하더라도, 코드베이스/청킹 방식/평가 프로토콜 차이로 완전히 동일 조건의 비교는 아님 — 어디까지나 "참고 앵커"이지 통제된 재현이 아님을 명시
- 외부 코드 재현(Phase 0.5)은 의존성 충돌, 미문서화된 하이퍼파라미터 등으로 일정 지연 리스크가 가장 큰 항목

## 산출물 총괄
5개 baseline wrapper + 공통 QA 인터페이스 + 난이도별/도메인별 QA 성능표(신뢰구간 포함) + 비용 비교 데이터 + 최종 비교 Table(원 논문 앵커 포함)

## 다음으로 도와드릴 수 있는 것
- baseline wrapper 5종 중 우선순위 2종(Microsoft GraphRAG, LightRAG) 재현 스크립트 작성
- `eval/benchmark.py` 통합 평가 러너 코드 작성
