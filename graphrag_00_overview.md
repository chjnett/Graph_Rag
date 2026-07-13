# LLM-Free GraphRAG 인덱싱: 증류 기반 경량 추출 모델 — 프로젝트 개요

**한 줄 정의**: 교사 LLM으로 합성 (문서 → 엔티티/관계) 데이터를 생성하고, 이를 경량 NLP 모델에 SFT로 증류하여, 인덱싱 시점에 LLM 호출 없이 GraphRAG 그래프를 구축하는 프레임워크.

> **⚠️ 리소스 제약 (2026-07 기준 확정, 예산 갱신 반영)**: 실행 환경은 **RTX 3090 1장(24GB VRAM, 시간 무제한)**이고, **유료 API 예산은 약 10,000원(~$7) 한도로 확정**(대학생 개인 프로젝트). 이 소액 예산은 두 용도로만 배분한다: (1) GPT-4o로 고정 평가 서브셋(150~200청크) **전체**를 재추출해 로컬 교사 품질을 실측 비교(예상 ~1,300원, 상세: `graphrag_01_data_pipeline.md` Phase 1.6-e), (2) 인간 검수 골드셋(30~50 triple)의 2차 라벨러로 GPT-4o를 투입해 Cohen's κ로 inter-annotator agreement 확보(예상 ~300원, Phase 1.6-f). 나머지(~8,000원+)는 재시도/버퍼로만 보류하며 신규 용도에 배정하지 않는다. **"교사 LLM"(합성 데이터 생성 및 인덱싱 시점 실제 추론)은 여전히 로컬 오픈소스 강모델(RTX 3090 4bit 양자화, 예: Qwen2.5-32B-Instruct-AWQ)이며, 유료 API는 오직 사후 검증(sanity-check/2차 라벨러) 목적에만 국한된다** — "완전 LLM-free 인덱싱"이라는 핵심 주장은 인덱싱 단계에 어떤 LLM도 호출하지 않는다는 뜻이므로, 오프라인 검증 단계의 소액 유료 API 사용과 절대 혼동되지 않도록 코드/논문 모두에서 명확히 구분한다. baseline 인덱싱도 동일 로컬 모델로 재현한다.

---

## 문서 지도

이 프로젝트 문서는 아래 6개 파일로 나뉘어 있다. 각 파일은 해당 토픽의 목표·데이터셋·Phase/Sub-phase 실행 계획·코드 매핑·한계를 모두 담고 있으며, 리소스 전제나 Phase 0.0(공통 인프라) 같은 공통 요소는 이 개요 문서에만 존재한다.

| 문서 | 범위 |
|---|---|
| `graphrag_00_overview.md` (이 문서) | 논문 설계, 데이터셋 인벤토리, 전체 일정/의존관계, 공통 인프라(Phase 0.0), 리포지토리 구조, Limitations 총괄 |
| `graphrag_01_data_pipeline.md` | 서브프로젝트 1 — 합성 데이터 생성 & 검증 (Phase 1.0~1.6) |
| `graphrag_02_distillation.md` | 서브프로젝트 2 — 경량 추출 모델 SFT/증류 (Phase 2.0~2.6) |
| `graphrag_03_graph_construction.md` | 서브프로젝트 3 — 그래프 구축 & 커뮤니티 요약 (Phase 3.0~3.7) |
| `graphrag_04_evaluation.md` | 서브프로젝트 4 — Baseline 재현 & 다운스트림 평가 (Phase 0.5, 4.1~4.5) |
| `graphrag_05_domain_adaptation.md` | 서브프로젝트 5 — 도메인 적응 실험 (Phase 5.0~5.4) |

---

## 1. 논문 설계

### 1.1 제목 (후보)
- "Distilling GraphRAG: Zero-LLM-Call Knowledge Graph Construction via Synthetic Data Distillation"
- "SynthGraph: Training Lightweight Extractors for Cost-Free GraphRAG Indexing"
- "From Teacher to Graph: LLM-Distilled Lightweight Models for Scalable GraphRAG"

### 1.2 Abstract 구조 (5문장 템플릿)
1. 배경: GraphRAG는 멀티홉 추론에 강하지만 인덱싱 시 대량 LLM 호출로 비용/지연이 큼
2. 문제: 기존 LLM-free 방법(NoLLMRAG 등)은 표면 통계에 의존해 정확도 손실이 큼
3. 제안: 교사 LLM 합성 데이터 → 경량 모델 SFT 증류 → 인덱싱 시 LLM 0회
4. 결과: 인덱싱 비용 X% 절감, LLM 기반 대비 실제 정확도/QA 정확도 Y% 유지 (교사-학생 일치율이 아니라 인간 검수 골드셋 기준 — `graphrag_03_graph_construction.md` §Phase 3.7 참고)
5. 기여: 증류 파이프라인 + 도메인 적응 실험 + 신규 평가 벤치마크 공개

### 1.3 Contribution Bullets (Introduction 말미)
- (C1) 교사 LLM → 경량 모델 증류를 통한 **완전 LLM-free 인덱싱** 파이프라인
- (C2) 범용 vs 도메인 특화 파인튜닝 비교를 통한 **도메인 적응 전략** 제시
- (C3) 인덱싱 비용·실제 정확도·QA 정확도·환각률을 통합한 **자체 평가 벤치마크**
- (C4) 오픈소스 코드/데이터 공개

### 1.4 Related Work 구성 (4개 서브섹션)

| 서브섹션 | 대표 논문 | 우리와의 차별점 |
|---|---|---|
| 4.1 LLM 기반 GraphRAG | Microsoft GraphRAG, LightRAG, KAG | 인덱싱 LLM 의존 |
| 4.2 인덱싱 비용 절감 (LLM 유지) | LazyGraphRAG, KET-RAG | 여전히 부분적 LLM 사용 |
| 4.3 LLM-Free 그래프 구축 | NoLLMRAG, LiteSemRAG, ContextRAG, dependency-parsing(2507.03226) | 정확도 격차 존재 |
| 4.4 LLM 증류/합성데이터 (IE 일반) | Distill-SynthKG, MetaIE, Sub-Billion RE, Clinical Synthetic Distillation | GraphRAG 인덱싱에 특화 적용한 사례 부재 → **우리 논문의 갭** |

### 1.5 Method / 실험설계 섹션 매핑

논문의 Method(§5)·실험설계(§6) 각 하위 섹션이 실제로 어느 실행 문서에 상세히 기술되어 있는지는 아래를 참고. (본 개요 문서에는 각 항목의 전체 설명을 반복하지 않는다.)

| 논문 섹션(예정) | 내용 | 상세 문서 |
|---|---|---|
| Method — 합성 데이터 생성 / 필터링 | 교사 모델 호출, 스키마, self-consistency confidence, 인간 검수 골드셋 | `graphrag_01_data_pipeline.md` |
| Method — 경량 모델 SFT | 베이스 모델, QLoRA/LoRA, 보조 데이터 비율 실험 | `graphrag_02_distillation.md` |
| Method — 그래프 구축 / 커뮤니티 요약 | 엔티티 정규화, 그래프 DB, 검색 인터페이스, Leiden, TextRank | `graphrag_03_graph_construction.md` |
| Method — 도메인 적응 파인튜닝 | Zero-shot vs adapted 비교 | `graphrag_05_domain_adaptation.md` |
| 실험설계 — Baseline / 평가지표 / 다운스트림 QA | 5종 baseline 재현, 비용·품질·환각률 지표 정의, GraphRAG-Bench/HotpotQA/MultiHop-RAG | `graphrag_04_evaluation.md` |
| 실험설계 — Ablation(데이터양/교사모델크기/필터링 유무) | GPU-시간 기준 ablation | `graphrag_01_data_pipeline.md` |

---

## 2. 데이터셋 인벤토리

| 데이터셋 | 용도 | 접근 경로 | 비고 |
|---|---|---|---|
| **UltraDomain** | 원본 코퍼스 (그래프 구축 대상) | HuggingFace `TommyChien/UltraDomain` | 428개 대학교재, 18개 도메인, GraphRAG/LightRAG/MemoRAG 등에서 표준으로 쓰는 벤치마크라 baseline과 공정 비교 가능. Fin(금융)/Leg(법률) 서브셋 포함되어 있어 도메인 적응 실험에도 바로 활용 |
| **GraphRAG-Bench** | 통합 평가 (그래프 구축~QA 전체 파이프라인) | HuggingFace `GraphRAG-Bench/GraphRAG-Bench` | 난이도별 질문(사실검색/복합추론/요약/생성) 태깅 있어 세분화된 평가 가능 |
| **HotpotQA** | 멀티홉 QA 다운스트림 평가 | HuggingFace `hotpot_qa` | 정답/근거 문단(supporting facts) 포함 → 인용 정확도 평가에 유용 |
| **MultiHop-RAG** | 멀티홉 QA 평가 (뉴스 도메인) | HuggingFace `yixuantt/MultiHopRAG` | UltraDomain과 다른 도메인이라 일반화 검증용 |
| **DocRED** | 학생 모델 relation extraction 학습/평가 (정답 라벨 有) | GitHub `thunlp/DocRED`, HuggingFace `docred` | 원격지도(distant supervision) + 사람 검수 라벨 존재 → 증류 없이도 "정답 있는" 절대 성능 상한선 측정 가능. 96종 닫힌 Wikidata 관계 스키마라 교사 모델의 자유 관계어(open-vocabulary)와 직접 비교 불가 — 반드시 관계어 정렬(mapping) 단계를 거쳐야 유효한 비교 (`graphrag_01_data_pipeline.md` Phase 1.3 참고) |
| **SciERC** | 학생 모델 보조 학습 데이터 (일반화) | HuggingFace `sciERC` / 공식 사이트 | 과학 논문 도메인 엔티티/관계 라벨 |
| **WebNLG** | 학생 모델 보조 학습 데이터 | HuggingFace `web_nlg` | (엔티티, 관계, 텍스트) 트리플이 이미 정제되어 있어 필터링 로직 검증용으로도 사용 가능 |
| **FewRel** | 저자원 관계 유형 일반화 테스트 | HuggingFace `few_rel` | 학생 모델이 학습 데이터에 없던 관계 유형에 얼마나 강건한지 확인 |

**왜 이 조합이 유리한가**
1. **접근성**: 전부 HuggingFace/GitHub에서 로그인 없이 바로 받을 수 있음 (TACRED처럼 LDC 라이선스가 필요한 데이터는 의도적으로 제외)
2. **비교 가능성**: UltraDomain, GraphRAG-Bench는 이미 GraphRAG 계열 논문들의 사실상 표준 벤치마크라, 우리 결과를 기존 논문 수치와 나란히 놓고 비교하기 쉬움
3. **외부 검증 가능성**: DocRED처럼 정답 라벨이 있는 데이터를 끼워 넣어서, "우리 모델이 실제로 맞는 관계를 뽑는가"를 LLM-as-judge 없이도 객관적으로 증명 가능
4. **도메인 다양성 확보**: UltraDomain 자체에 이미 Fin/Leg 서브셋이 있어서 별도 도메인 데이터 수집 없이 도메인 적응 실험이 바로 가능

---

## 3. 전체 일정 (재산정 — 개인/대학생, 단일 RTX 3090 기준)

> 원래 10주 계획은 세부 Phase 분해(총 31개 phase) 기준으로 재계산하면 현실적으로 **4~6개월** 규모. GPU 시간은 무제한이지만 본인의 실제 가용 시간(주당 투입 가능 시간)에 따라 조정.

| 개월차 | 주요 내용 | 해당 Phase |
|---|---|---|
| 1개월차 | Related Work 정리, 데이터셋/스키마 확정, 로컬 교사 모델 서빙 환경 구축, baseline 5종 재현 착수(병렬) | Phase 0.0, 0.5 + 서브1 전체(1.0~1.6) |
| 2개월차 | 경량 모델 SFT + ablation, baseline 재현 계속 | 서브2 전체(2.0~2.6) |
| 3개월차 | 그래프 구축 + 검색(retrieval) + 커뮤니티 요약, baseline 재현 마무리 | 서브3 전체(3.0~3.7, 3.35 포함) |
| 4개월차 | 다운스트림 QA 평가 + baseline 비교, 도메인 적응은 이 시점부터 병렬 | 서브4(4.1~4.5) + 서브5(5.0~5.4) 병렬 |
| 5~6개월차 | 결과 분석, ablation 정리, 논문 작성/투고 (arXiv 프리프린트 우선) | — |

Phase 0.5(baseline 재현)의 실제 소요가 예상보다 길어질 경우 이 일정에서 가장 먼저 완충 여유를 줄 항목이므로, 1개월차 종료 시점에 baseline 재현 상태를 반드시 점검할 것. 특히 Phase 0.5-a0(처리량 사전 체크, `graphrag_04_evaluation.md` 참고)에서 MS GraphRAG/LightRAG의 전체 코퍼스 인덱싱 외삽 시간이 baseline 1종당 약 2주를 넘는 것으로 확인되면, QA 비교 코퍼스를 UltraDomain 전체(428개)가 아니라 벤치마크 질문이 실제로 걸쳐 있는 서브셋으로 축소하는 절충안을 조기에 결정할 것.

### 코드 마일스톤 체크리스트 (일정과 동기화)
1. Phase 0.0: 로컬 교사 모델 vLLM 서빙 기동 확인 (샘플 1건 응답)
2. `graphrag_01_data_pipeline.md` Phase 1.1 Done when 충족 (파일럿 파싱 성공률 ≥95%)
3. `graphrag_04_evaluation.md` Phase 0.5-a0(처리량 사전 체크, 문서 10~20개 샘플) 완료 후 Phase 0.5 Done when 충족 (baseline 5종 중 최소 2종 로컬 LLM 연동 확인) — **가장 리스크 큰 항목이므로 조기 확인**, 처리량 실측치에 따라 코퍼스 축소 여부도 이 시점에 결정
4. `graphrag_02_distillation.md` Phase 2.2 Done when 충족 (SFT 1 epoch 정상 종료)
5. `graphrag_03_graph_construction.md` Phase 3.35 Done when 충족 (쿼리 1건 end-to-end 테스트 통과)
6. `graphrag_04_evaluation.md` benchmark 러너로 전체 파이프라인 자동 비교 실행 (교사-학생 일치율 + 골드셋 실제 정확도 + baseline 원 논문 앵커 병기 포함)
7. Ablation 스크립트화 (`graphrag_01_data_pipeline.md` 참고)

---

## 4. 전체 Phase 의존관계

```
Phase 0:  0.0(로컬교사서빙) → 0.5(baseline 5종 재현, 로컬LLM 연동) ─────────────┐
                │                                                              │
서브1:          └→ 1.0→1.1→1.2→1.3→1.4→1.5→1.6                                │
                                  │                                            │
서브2:                            └→ 2.0→2.1→2.2→2.3→2.4→2.5→2.6              │
                                                            │                  │
서브3:                                                      └→ 3.0→3.1→3.2→3.3→3.35→3.4→3.5→3.6→3.7
                                                            │                        │        │
서브4:                                                      │                        │        └→ 4.1→4.2→4.3→4.4→4.5
                                                            │                        │                 ↑
서브5:                                                      └→ 5.0→5.1→5.2→5.3→5.4 ─┴─────────────────┘
                                                                                      (5.4는 서브4 하네스 재사용)
```

- **Phase 0(0.0, 0.5)은 1주차에 서브1과 완전 병렬로 착수** — baseline 재현 리스크를 최대한 일찍 드러내기 위함
- 서브1(1.0~1.6)과 서브2 초반(2.0~2.1) 준비는 코퍼스만 있으면 병렬 착수 가능
- 서브3은 서브2.6(최종 체크포인트) 확정 후 시작
- 서브3.35(검색 인터페이스)는 서브3.3(그래프 DB) 스키마 확정에 영향을 주므로 동시 설계 권장
- 서브3.7(교사-학생 일치율 + 골드셋 실제 정확도)은 서브1의 고정 평가 서브셋(150~200청크) 및 인간 검수 골드셋(Phase 1.6) 산출물이 모두 필요
- 서브5는 서브2.6 완료 즉시 서브3과 병렬 진행 가능 (인력 여유 시), 단 Phase 5.4는 서브4 하네스가 준비된 후 실행
- 서브4의 QA 실행(4.1~4.5)은 서브3.35(검색 인터페이스), 서브3.7(그래프 완성) 이후 시작 — baseline 자체 재현(Phase 0.5)은 이미 끝나 있어야 함

---

## 5. 공통 인프라 — Phase 0.0 로컬 교사 모델 서빙 환경 구축

서브1(합성 데이터 생성)과 서브4(baseline 재현) 양쪽에서 공유하는 유일한 선행 인프라이므로 이 개요 문서에만 둔다.

- **0.0-a** (M) vLLM 설치, CUDA/드라이버 호환성 확인 (RTX 3090, 24GB VRAM 기준)
- **0.0-b** (M) Qwen2.5-32B-Instruct-AWQ(또는 14B-Instruct, 속도 우선 시) 다운로드 및 4bit 양자화 로드 확인
- **0.0-c** (S) OpenAI 호환 엔드포인트로 vLLM 서버 기동, curl/Python 클라이언트로 샘플 요청 1건 응답 확인
- **0.0-d** (M) 처리량 벤치마크: 20개 청크로 초당 토큰 수, 청크당 평균 소요시간 측정 → 이후 데이터 생성/전체 코퍼스 추론 규모 산정의 기초 자료로 기록
- Done when: 로컬 엔드포인트가 안정적으로 응답하고, 처리량 수치가 문서화됨
- **산출물**: 로컬 서빙 기동 스크립트, 처리량 벤치마크 로그

---

## 6. 리포지토리 최상위 구조

```
llmfree-graphrag/
├── configs/
│   ├── data_gen.yaml          # graphrag_01_data_pipeline.md
│   ├── train_sft.yaml         # graphrag_02_distillation.md
│   ├── domain_adapt.yaml      # graphrag_05_domain_adaptation.md
│   └── eval.yaml              # graphrag_04_evaluation.md
├── data/
│   ├── raw_corpus/
│   ├── synthetic/           # 교사 LLM 생성 (문서, triple) 쌍 — graphrag_01
│   └── domain_corpus/       # 도메인 적응용 소량 데이터 — graphrag_05
├── src/
│   ├── data_gen/            # graphrag_01_data_pipeline.md
│   ├── training/            # graphrag_02_distillation.md, graphrag_05_domain_adaptation.md
│   ├── graph_construction/  # graphrag_03_graph_construction.md
│   ├── retrieval/           # graphrag_03_graph_construction.md
│   └── eval/                # graphrag_04_evaluation.md
├── baselines/                # graphrag_04_evaluation.md
├── scripts/
│   ├── run_data_gen.sh
│   ├── run_train.sh
│   └── run_eval.sh
└── notebooks/
    └── ablation_analysis.ipynb   # graphrag_01_data_pipeline.md
```

각 하위 디렉토리의 파일별 역할과 기술 스택은 해당 토픽 문서(`graphrag_0N_*.md`)의 "코드 매핑" 섹션에 상세히 기술되어 있으며, 이 문서에는 반복하지 않는다.

---

## 7. Limitations (총괄)

- **로컬 오픈소스 교사(32B급)는 GPT-4o 등 최상위 proprietary 모델보다 약할 수 있음** — 이전에는 가정으로만 명시했으나, 이제 고정 평가 서브셋(150~200청크) 전체에 대해 GPT-4o와 직접 비교해 격차를 실측한다(`graphrag_01_data_pipeline.md` Phase 1.6-e). 다만 이 비교는 검증용이며 합성 데이터 생성 자체(500~1,000청크)를 유료 API로 대체하는 것은 아니므로, 품질 상한이 낮을 가능성이라는 근본적 제약은 남는다. 이는 모든 서브프로젝트에 영향을 미침
- 합성 데이터의 교사 모델 편향이 학생 모델에 전이될 위험 — 상세: `graphrag_01_data_pipeline.md` §한계. 인간 검수 골드셋의 단일 annotator 한계는 GPT-4o 2차 라벨러(Phase 1.6-f) 도입으로 일부 완화(완전한 2인 human 교차검증은 아님)
- 초정밀 도메인(의료/법률)에서는 여전히 (로컬이든 API든) LLM 대비 격차 존재 가능성 — 상세: `graphrag_05_domain_adaptation.md` §한계
- TextRank 기반 커뮤니티 요약의 정성적 격차, 도메인별 세부 분석의 표본 크기 한계(탐색적 해석으로 한정) — 상세: `graphrag_03_graph_construction.md` §한계
- Baseline 비교의 삼각점(원 논문 GPT-4 수치) 한계 — 상세: `graphrag_04_evaluation.md` §한계

---

## 다음으로 도와드릴 수 있는 것
- 각 토픽 문서의 Sub-phase를 실제 이슈 트래커(GitHub Projects/Linear 등) 티켓으로 변환
- Phase 0.0(vLLM 로컬 서빙) 셋업 스크립트 실제 작성
- 논문 Abstract/Introduction 초안 텍스트 작성
