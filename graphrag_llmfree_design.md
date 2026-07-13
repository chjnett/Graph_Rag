# LLM-Free GraphRAG 인덱싱: 증류 기반 경량 추출 모델 — 전체 설계

**한 줄 정의**: 교사 LLM으로 합성 (문서 → 엔티티/관계) 데이터를 생성하고, 이를 경량 NLP 모델에 SFT로 증류하여, 인덱싱 시점에 LLM 호출 없이 GraphRAG 그래프를 구축하는 프레임워크.

> **⚠️ 리소스 제약 (2026-07 기준 확정)**: 실행 환경은 **RTX 3090 1장(24GB VRAM, 시간 무제한)**이고, **유료 API 예산은 사실상 $0**(대학생 개인 프로젝트, sanity-check용 소액만 선택적 허용). 이에 따라 원래 설계의 "GPT-4o급/Claude API 교사"와 "8노드 GPU 클러스터"는 전부 **로컬 오픈소스 강모델(RTX 3090 4bit 양자화 구동)**로 대체한다. 이 문서 전체에서 "교사 LLM"은 별도 언급이 없는 한 로컬 모델을 뜻한다. 상세 근거는 §5.1, §6.2, Part2 §3~4 참고.

---

# PART 1. 논문 설계

## 1. 제목 (후보)

- "Distilling GraphRAG: Zero-LLM-Call Knowledge Graph Construction via Synthetic Data Distillation"
- "SynthGraph: Training Lightweight Extractors for Cost-Free GraphRAG Indexing"
- "From Teacher to Graph: LLM-Distilled Lightweight Models for Scalable GraphRAG"

## 2. Abstract 구조 (5문장 템플릿)

1. 배경: GraphRAG는 멀티홉 추론에 강하지만 인덱싱 시 대량 LLM 호출로 비용/지연이 큼
2. 문제: 기존 LLM-free 방법(NoLLMRAG 등)은 표면 통계에 의존해 정확도 손실이 큼
3. 제안: 교사 LLM 합성 데이터 → 경량 모델 SFT 증류 → 인덱싱 시 LLM 0회
4. 결과: 인덱싱 비용 X% 절감, LLM 기반 대비 그래프 커버리지/QA 정확도 Y% 유지
5. 기여: 증류 파이프라인 + 도메인 적응 실험 + 신규 평가 벤치마크 공개

## 3. Contribution Bullets (Introduction 말미)

- (C1) 교사 LLM → 경량 모델 증류를 통한 **완전 LLM-free 인덱싱** 파이프라인
- (C2) 범용 vs 도메인 특화 파인튜닝 비교를 통한 **도메인 적응 전략** 제시
- (C3) 인덱싱 비용·그래프 커버리지·QA 정확도·환각률을 통합한 **자체 평가 벤치마크**
- (C4) 오픈소스 코드/데이터 공개

## 4. Related Work 구성 (4개 서브섹션)

| 서브섹션 | 대표 논문 | 우리와의 차별점 |
|---|---|---|
| 4.1 LLM 기반 GraphRAG | Microsoft GraphRAG, LightRAG, KAG | 인덱싱 LLM 의존 |
| 4.2 인덱싱 비용 절감 (LLM 유지) | LazyGraphRAG, KET-RAG | 여전히 부분적 LLM 사용 |
| 4.3 LLM-Free 그래프 구축 | NoLLMRAG, LiteSemRAG, ContextRAG, dependency-parsing(2507.03226) | 정확도 격차 존재 |
| 4.4 LLM 증류/합성데이터 (IE 일반) | Distill-SynthKG, MetaIE, Sub-Billion RE, Clinical Synthetic Distillation | GraphRAG 인덱싱에 특화 적용한 사례 부재 → **우리 논문의 갭** |

## 5. Method 섹션 구조

### 5.1 합성 데이터 생성 (Synthetic Data Generation)
- 교사 모델: **RTX 3090에서 4bit 양자화로 구동하는 로컬 오픈소스 강모델** (예: Qwen2.5-32B-Instruct-AWQ 또는 속도가 필요하면 Qwen2.5-14B-Instruct), vLLM으로 서빙해 OpenAI 호환 API처럼 호출. API 비용 $0, 대신 처리 시간은 GPU 순차 처리로 소요(무제한 GPU 시간 전제)
- few-shot 프롬프트로 (텍스트 청크 → 엔티티/관계 triple) 페어 생성
- 스키마(전 문서 통일, DocRED 정렬·hallucination 검증에 필요한 필드 모두 포함): `(entity1, entity1_type, relation, entity2, entity2_type, source_span, confidence)`
- 생성량: GPU-시간 기준으로 스코프 (파일럿 100 → 본생성 500~1,000 청크, §6.4 참고). 5K 규모 ablation은 시간 여유 있을 때만 선택적으로 시도
- (선택, 소액 예산 허용 시) GPT-4o-mini 등 저가 API로 50~100건 sanity-check만 수행해 로컬 교사 품질을 참고 비교 — 필수 아님

### 5.2 데이터 필터링/품질관리
- 중복 triple 제거, 문법적 일관성 체크(자체 룰 기반)
- 로컬 교사 모델의 self-consistency(동일 청크 2~3회 생성 후 일치도)로 confidence 산출 및 2차 필터링 — confidence는 모델 자기보고 점수가 아니라 **다중 샘플 일치율 기반**으로 정의(보정 문제 회피)
- 노이즈 비율 정량화 (필터링 전/후 비교 지표화)

### 5.3 경량 모델 SFT (지식 증류)
- 베이스 모델 후보: DeBERTa-v3 (span-based NER/RE), 또는 1B 이하 소형 생성 모델
- 학습 방식: QLoRA/LoRA 파인튜닝, label-to-span 방식(MetaIE 참고) 검토
- 하이퍼파라미터: lr, batch size, epoch 등 실험 로그화

### 5.4 도메인 적응 파인튜닝
- 범용 코퍼스로 1차 학습된 모델 → 특정 도메인(예: 기업 문서, 법률, 금융) 소량 데이터로 2차 파인튜닝
- Zero-shot(범용 모델) vs Domain-adapted 성능 비교가 핵심 실험

### 5.5 그래프 구축 파이프라인
- 경량 모델 추론 → triple 수집 → 엔티티 중복 제거(entity resolution, 임베딩 유사도 기반)
- NetworkX/Neo4j 그래프 저장
- (선택) 커뮤니티 탐지: Leiden/Louvain 알고리즘 (LLM 없이)

### 5.6 커뮤니티 요약 (LLM-free 확장 — 차별화 포인트)
- 기존 연구 대부분 이 단계는 LLM 의존적 → 우리는 추출적 요약(extractive summarization, TextRank 등) 또는 소형 모델로 대체 시도
- 이 부분이 "3번 갭"에서 언급한 니치 포인트

## 6. 실험 설계

### 6.1 데이터셋
- 공개 벤치마크: UltraDomain subset (ContextRAG, HiRAG에서 사용한 것과 동일 계열로 비교 가능)
- ~~자체 도메인 데이터셋 1개(엔터프라이즈 문서 시뮬레이션)~~ → **삭제**. 라이선스/PII 검토 부담 대비 실익이 낮고 실행 계획(서브프로젝트 5)에도 반영되지 않던 고아 항목이었음. 도메인 다양성은 UltraDomain 내 Fin/Leg 서브셋으로 충분히 확보(§5.4, 서브프로젝트 5)

### 6.2 베이스라인
- Microsoft GraphRAG, LightRAG, LiteSemRAG(LLM-free, 임베딩 기반), Dependency-parsing 방식(arXiv:2507.03226), NoLLMRAG
- **인덱싱 LLM 통일**: Microsoft GraphRAG/LightRAG는 원래 OpenAI API를 호출하지만, 본 프로젝트에서는 API 비용을 낼 수 없으므로 **§5.1과 동일한 로컬 vLLM 서빙 모델**을 OpenAI 호환 엔드포인트로 연결해 인덱싱시킨다. 이렇게 하면 baseline 간 "비용" 비교가 $ API 요금이 아니라 GPU-시간/wall-clock 기준으로 공정하게 통일됨(§6.3 참고)
- Baseline 재현은 **일정상 최우선 트랙**으로 1주차부터 서브1(합성데이터)과 병렬 착수 (§8, 이유는 phases 문서의 의존성 재검토 참고 — 외부 코드 재현이 가장 시간 예측이 어려운 작업이므로 조기에 리스크를 드러내야 함)
- **원 논문 수치 앵커링(리뷰 반영)**: baseline을 로컬 LLM으로 재현하면 "GPU-시간 기준 공정 비교"는 되지만, 원래 GraphRAG/LightRAG 논문이 GPT-4 계열로 보고한 정확도 수치와는 직접 비교가 아니게 됨. 따라서 최종 결과 Table에는 (1) 우리 방법, (2) 로컬 LLM 재현 baseline, (3) 원 논문 보고 수치(가능한 범위 내 인용) 세 지점을 나란히 병기해, "로컬 재현이 원 논문 대비 얼마나 괴리되는지"와 "우리 방법이 그 안에서 어디에 위치하는지"를 독자가 함께 판단할 수 있게 한다 (§6.3, Phase 3.6/4.5 참고)

### 6.3 평가 지표
> **개정 (리뷰 반영)**: 기존 "그래프 커버리지" 단일 지표는 학생 모델의 학습 신호(교사 출력)와 평가 정답(교사 출력)이 동일해 순환논증 위험이 있었음. 아래에서 "교사-학생 일치율"과 "실제 정확도(인간 검수)"를 분리했고, 고정 평가 서브셋도 도메인당 표본이 지나치게 작았던 문제(50~100청크/18도메인)를 반영해 150~200청크로 확대함. Baseline 비교에는 원 논문 보고 수치를 앵커로 추가.

| 축 | 지표 |
|---|---|
| 비용 | 인덱싱 GPU-시간(wall-clock), (참고용으로만) 동일 작업을 유료 API로 했을 때의 추정 $ 비용 |
| 품질 (교사-학생 일치) | 교사-학생 그래프 일치율(Teacher-Student Agreement) — **고정 평가 서브셋**(도메인별 균등 샘플 150~200청크, 도메인당 약 8~11청크)에 대해 로컬 교사 모델이 만든 참조 그래프 대비 학생 모델 그래프의 엔티티/관계 recall. **주의: 이 수치는 "학생이 교사를 얼마나 잘 모방했는가"를 재는 지표이며, 그 자체로 "그래프가 실제로 정확한가"의 증명은 아님** (교사가 체계적으로 틀리면 학생이 그 오류까지 복제해도 이 수치는 높게 나옴) |
| 품질 (실제 정확도) | **인간 검수 골드셋 대비 정확도** — 고정 평가 서브셋 중 30~50 triple을 저자가 직접 원문 `source_span` 대조로 수기 라벨링한 gold subset에 대해 학생 모델 precision/recall 계산. 위 "교사-학생 일치율"과 이 수치의 괴리 정도가 교사 모델 편향의 실질적 증거가 됨 (단일 annotator 한계는 §7 명시) |
| 다운스트림 | Multi-hop QA 정확도, 인용 근거 일치율. **원 논문(Microsoft GraphRAG/LightRAG 등)이 GPT-4 계열로 보고한 공개 수치를 참고 앵커로 최종 Table에 병기** — 우리는 baseline도 로컬 LLM으로 재현하므로(§6.2), 원 논문 수치라는 세 번째 기준점 없이는 "LLM 기반과 동등한 품질"이라는 주장이 "동일한 약한 교사를 쓴 두 방법 간 비교"로 축소될 위험이 있음 |
| 안전성 | 환각률 — **추출된 triple 중 원문 `source_span`에서 실제로 지지되지 않는 비율** (그래프 존재 여부가 아니라 원문 대조로 정의) |
| 적응력 | 범용 vs 도메인 적응 모델의 triple 성능 격차 **+ 다운스트림 QA 정확도 격차** (서브프로젝트 5.4에서 QA 하네스 재사용) |

> **도메인별 세부 분석 관련 유의**: 18개 도메인에 걸친 도메인별 recall/성능 분포(예: Phase 3.7-c)는 도메인당 표본이 여전히 10개 내외로 작아 확증적 결론이 아니라 **탐색적(exploratory) 해석**으로 한정한다. 통계적으로 유의한 도메인 간 차이를 주장하려면 표본을 추가 확대해야 함을 Limitations에 명시.

### 6.4 Ablation (GPU-시간 기준으로 스코프, API 비용 없음)
- 합성 데이터 양에 따른 성능 곡선 (100 / 500 / 1,000 pairs — 로컬 GPU 생성 속도 고려해 5K는 시간 여유 시 선택적 확장)
- 교사 모델 크기 변화 (로컬 14B vs 32B 양자화 모델이 만든 데이터 품질 비교 — 3090 한 장으로 실행 가능한 범위로 조정, 원래의 70B/8B는 단일 3090으로는 비현실적)
- 필터링 유무 비교

## 7. Limitations 섹션 (미리 준비)
- 초정밀 도메인(의료/법률)에서는 여전히 (로컬이든 API든) LLM 대비 격차 존재 가능성
- 합성 데이터의 교사 모델 편향이 학생 모델에 전이될 위험
- **로컬 오픈소스 교사(32B급)는 GPT-4o/Claude 등 최상위 proprietary 모델보다 약할 수 있어, 합성 데이터 품질 상한 자체가 원래 설계보다 낮을 가능성** — 이는 예산 제약에 따른 의도적 트레이드오프임을 논문에 명시
- TextRank 기반 커뮤니티 요약은 LLM 서술형 요약 대비 품질이 열위일 가능성이 높음(§5.6 참고, 커뮤니티는 "문장 집합"이 아니라 "엔티티/관계 집합"이라 TextRank 적용 방식 자체가 근사적임)
- **인간 검수 골드셋(30~50 triple)이 단일 annotator(저자 본인)로 라벨링됨** — inter-annotator agreement를 확보하지 못했으므로, 이 골드셋 대비 정확도 수치는 "저자 본인 판단 기준"이라는 한계를 명시. 여유가 있다면 2인 교차검증으로 보강
- **도메인별 세부 분석(18개 도메인)은 도메인당 표본이 10개 내외로 작아 탐색적(exploratory) 해석에 한정** — 도메인 간 차이에 대한 확증적 결론은 유보하고, 후속 연구에서 표본 확대가 필요함을 명시
- **Baseline 비교의 삼각점 한계**: 원 논문(GraphRAG/LightRAG 등)의 GPT-4 기반 보고 수치를 앵커로 병기하더라도, 코드베이스/청킹 방식/평가 프로토콜 차이로 완전히 동일 조건의 비교는 아님 — 어디까지나 "참고 앵커"이지 통제된 재현이 아님을 명시

## 8. 논문 작성 타임라인 (재산정 — 개인/대학생, 단일 RTX 3090 기준)
> 원래 10주 계획은 세부 Phase 분해(`graphrag_subprojects_phases.md`, 총 31개 phase) 기준으로 재계산하면 현실적으로 **4~6개월** 규모입니다. 아래는 그 기준의 재산정 타임라인이며, GPU 시간은 무제한이지만 본인의 실제 가용 시간(주당 투입 가능 시간)에 따라 조정하세요.

- 1개월차: Related Work 정리, 데이터셋/스키마 확정, **로컬 교사 모델 서빙 환경(vLLM) 구축**, baseline 5종 재현 착수(병렬 시작)
- 2개월차: 합성 데이터 생성 파이프라인 + DocRED 대조군 + 필터링(서브1 전체), baseline 재현 계속
- 3개월차: 경량 모델 SFT + ablation(서브2 전체), baseline 재현 마무리
- 4개월차: 그래프 구축 + 검색(retrieval) + 커뮤니티 요약(서브3 전체), 도메인 적응(서브5)은 이 시점부터 병렬 가능
- 5개월차: 다운스트림 QA 평가 + baseline 비교(서브4), 도메인 적응 QA 검증(서브5) 마무리
- 6개월차: 결과 분석, ablation 정리, 논문 작성/투고 (arXiv 프리프린트 우선)

---

# PART 2. 코드 설계

## 1. 리포지토리 구조

```
llmfree-graphrag/
├── configs/
│   ├── data_gen.yaml
│   ├── train_sft.yaml
│   ├── domain_adapt.yaml
│   └── eval.yaml
├── data/
│   ├── raw_corpus/
│   ├── synthetic/           # 교사 LLM 생성 (문서, triple) 쌍
│   └── domain_corpus/       # 도메인 적응용 소량 데이터
├── src/
│   ├── data_gen/
│   │   ├── teacher_prompt.py      # few-shot 프롬프트 템플릿
│   │   ├── teacher_serve.py       # 로컬 강모델 vLLM 서빙 기동 스크립트 (OpenAI 호환 엔드포인트)
│   │   ├── generate.py            # 로컬 교사 모델 호출 배치 스크립트 (vLLM 엔드포인트 대상)
│   │   ├── docred_align.py        # 교사 자유 관계어 ↔ DocRED 96종 스키마 정렬(매핑)
│   │   └── filter.py              # 중복/노이즈 필터링, self-consistency 기반 confidence 산출
│   ├── training/
│   │   ├── sft_train.py           # QLoRA/LoRA SFT 학습 루프
│   │   ├── model_def.py           # 학생 모델 아키텍처 (span-based or seq2seq)
│   │   └── domain_adapt.py        # 2차 도메인 파인튜닝
│   ├── graph_construction/
│   │   ├── extractor.py           # 학생 모델 추론 wrapper
│   │   ├── entity_resolution.py   # 임베딩 기반 엔티티 중복 제거
│   │   ├── graph_builder.py       # NetworkX/Neo4j 그래프 생성
│   │   └── community_summary.py   # LLM-free 요약 (TextRank 등)
│   ├── retrieval/
│   │   └── graph_retriever.py     # 쿼리 시 그래프 탐색/서브그래프 추출 (1-hop/2-hop, 랭킹)
│   └── eval/
│       ├── benchmark.py           # 통합 평가 러너
│       ├── metrics_cost.py        # GPU-시간/wall-clock 측정 (+ 참고용 API 환산 비용)
│       ├── metrics_coverage.py    # 고정 평가 서브셋 기준 교사-학생 일치율(agreement) 계산 — 실제 정확도 아님(순환논증 주의)
│       ├── metrics_gold_accuracy.py  # 인간 검수 골드셋(30~50 triple) 대비 precision/recall — 순환논증 방지용 실제 정확도 지표
│       ├── metrics_qa.py          # QA 정확도 (LLM-as-judge or EM/F1), 원 논문 보고 수치 앵커 병기
│       └── metrics_hallucination.py  # source_span 대조 기반 환각률 계산
├── baselines/
│   ├── ms_graphrag_wrapper.py
│   ├── lightrag_wrapper.py
│   ├── litesemrag_wrapper.py
│   └── deps_parsing_wrapper.py     # arXiv:2507.03226 재현
├── scripts/
│   ├── run_data_gen.sh
│   ├── run_train.sh
│   └── run_eval.sh
└── notebooks/
    └── ablation_analysis.ipynb
```

## 2. 파이프라인 단계별 설계

### Stage 0 — 환경/데이터 준비
- 코퍼스 확보 (UltraDomain subset + 자체 도메인 문서)
- 청크 분할 (기존 GraphRAG 방식과 동일한 chunk size로 맞춰 baseline과 공정 비교)

### Stage 1 — 합성 데이터 생성 (`data_gen/`)
- 입력: 텍스트 청크
- 로컬 강모델을 vLLM으로 서빙(`teacher_serve.py`) 후 OpenAI 호환 엔드포인트로 호출 — API 비용 없음, GPU 순차 처리 시간만 소요
- 프롬프트: few-shot으로 (entity1, entity1_type, relation, entity2, entity2_type, source_span) triple 추출 요청
- 출력 스키마 고정 (JSON), 파싱 실패 시 재시도 로직
- 배치 처리 + 캐싱 (같은 청크 재생성 방지)
- DocRED 문서에도 동일 파이프라인 적용 후 `docred_align.py`로 자유 관계어를 DocRED 96종 스키마에 정렬 — 이 정렬 없이는 교사 정확도 "상한선" 수치가 무의미함(원문 스키마 불일치 문제)

### Stage 2 — 필터링 (`data_gen/filter.py`)
- Rule 기반: 빈 triple, self-loop, 타입 불일치 제거
- 통계 기반: triple 빈도 이상치 제거
- confidence: 동일 청크를 2~3회 재생성해 self-consistency(일치도)로 산출 — 로컬 모델이라 반복 호출 비용이 GPU 시간만이므로 API 대비 부담 없음
- (선택, 소액 예산 시) 저가 API로 2차 검증 — 단, 이 부분은 "완전 LLM-free"라는 논문 주장과 충돌하지 않도록 **오프라인 학습 데이터 생성 단계**에만 (로컬이든 API든) 강모델을 쓰고, **인덱싱(추론) 단계**에는 절대 강모델을 쓰지 않는다는 경계를 코드/논문 모두에서 명확히 구분

### Stage 3 — 학생 모델 학습 (`training/`)
- 아키텍처 후보 A: span extraction (DeBERTa 기반 NER+RE 헤드)
- 아키텍처 후보 B: 소형 seq2seq (T5-small/1B급, "text → structured triple" 생성)
- 학습: HuggingFace `transformers` + `peft`(LoRA/QLoRA)
- 로깅: wandb 또는 자체 로그 CSV (loss, eval F1 등)

### Stage 4 — 도메인 적응 (`training/domain_adapt.py`)
- 1차 학습된 체크포인트 로드 → 도메인 소량 데이터로 추가 파인튜닝
- Freeze 전략 실험 (전체 vs 마지막 레이어만)

### Stage 5 — 그래프 구축 (`graph_construction/`)
- 학생 모델로 triple 추출 (배치 추론, GPU 활용)
- 엔티티 중복 제거: 문자열 유사도 + 임베딩 유사도 결합
- 그래프 저장: 소규모는 NetworkX, 대규모는 Neo4j
- 커뮤니티 탐지: Leiden 알고리즘 (`python-igraph` or `networkx` 확장)
- 커뮤니티 요약: LLM 없이 TextRank/추출 요약, 또는 학생 모델 재활용

### Stage 6 — 검색/응답 (`retrieval/`)
- 쿼리 → 관련 서브그래프 탐색 (1-hop/2-hop) → 컨텍스트 구성 → (최종 응답 생성은 LLM 사용 — 이 부분은 GraphRAG의 표준 구조와 동일, 인덱싱만 LLM-free임을 명확히)

### Stage 7 — 평가 (`eval/`)
- `benchmark.py`가 모든 baseline + 우리 방법을 동일 데이터셋에 대해 순차 실행
- 결과를 단일 테이블(csv/markdown)로 자동 집계 → 논문 Table 그대로 사용 가능하게 설계
- 골드셋 기반 실제 정확도(`metrics_gold_accuracy.py`)와 교사-학생 일치율(`metrics_coverage.py`)을 별도 컬럼으로 병기해 순환논증 여부를 항상 함께 확인 가능하게 출력
- baseline 비교 Table에는 원 논문 보고 수치(가능한 범위) 컬럼을 추가해 삼각비교 구성

## 3. 기술 스택 제안

| 영역 | 도구 |
|---|---|
| 학생 모델 학습 | HuggingFace `transformers`, `peft`, `bitsandbytes` (QLoRA, 단일 RTX 3090 기준) |
| 로컬 교사 모델 서빙 | `vllm` (OpenAI 호환 엔드포인트로 서빙, 4bit AWQ/GPTQ 양자화) |
| 그래프 저장/탐색 | `networkx` (프로토타입), `Neo4j` (스케일업, 로컬 docker) |
| 커뮤니티 탐지 | `python-igraph`(Leiden), `networkx`(Louvain) |
| 임베딩(엔티티 중복 제거용) | `sentence-transformers` (로컬 실행) |
| 평가/실험관리 | `wandb` 또는 `mlflow` (무료 tier) |
| 합성데이터 생성 | 로컬 vLLM 서빙 모델 (API 비용 없음). 유료 API는 sanity-check 소액 용도로만 선택적 사용 |

## 4. 인프라 관점 (단일 RTX 3090, 24GB VRAM 기준)
- ~~8노드 GPU 클러스터~~ 가정은 실제 환경과 맞지 않아 제거. 실제로는 **단일 GPU 순차 처리** 계획으로 재작성:
  - 로컬 교사 모델 추론(합성 데이터 생성, DocRED 대조군) → 학생 모델 SFT(QLoRA) → 학생 모델 배치 추론(전체 코퍼스) → baseline 인덱싱(로컬 LLM으로 재현) 이 네 가지가 **모두 같은 GPU를 순차 점유**하므로, 각 단계 실행 시간을 미리 벤치마크(소규모 파일럿)해서 전체 일정에 반영해야 함 (§8 타임라인)
  - baseline 재현(Microsoft GraphRAG/LightRAG 인덱싱)도 로컬 LLM을 태우므로 이 역시 GPU 시간 예산에 포함해서 계획
  - "8노드 클러스터/분산학습 역량 어필"용 문단은 삭제 — 실제 하드웨어와 다른 내용을 논문/코드 설계에 남겨두면 재현성 문서로서 오해를 유발함

## 5. 코드 마일스톤 (논문 타임라인과 동기화, §8 참고)
1. `data_gen/teacher_serve.py`로 로컬 교사 모델 vLLM 서빙 기동 확인 (샘플 1건 응답)
2. `data_gen` 파이프라인 프로토타입 (파일럿 100 샘플) + DocRED 정렬(`docred_align.py`) 동작 확인
3. baseline wrapper 5종 중 최소 2종(Microsoft GraphRAG, LightRAG) 로컬 LLM 연동 재현 확인 — **가장 리스크 큰 항목이므로 조기에 착수**
4. `training` SFT 최소 동작 확인 (1 epoch, 작은 데이터)
5. `graph_construction` 통합 → end-to-end 인덱싱 1개 문서로 smoke test (`retrieval/graph_retriever.py` 포함)
6. `eval/benchmark.py`로 전체 파이프라인 자동 비교 실행 (교사-학생 일치율 + 골드셋 실제 정확도 + baseline 원 논문 앵커 병기 포함)
7. Ablation 스크립트화 (`notebooks/ablation_analysis.ipynb`)

---

## 다음으로 도와드릴 수 있는 것
- `data_gen/teacher_prompt.py`용 실제 few-shot 프롬프트 초안 작성
- 학생 모델 학습 스크립트(`sft_train.py`) 실제 코드 작성
- 논문 Abstract/Introduction 초안 텍스트 작성
