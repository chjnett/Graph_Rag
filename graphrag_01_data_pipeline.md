# 서브프로젝트 1 — 합성 데이터 생성 & 검증 파이프라인

> 리소스 전제(RTX 3090 1장, API 예산 $0), 공통 인프라(Phase 0.0 로컬 교사 서빙)는 `graphrag_00_overview.md` 참고.

## 목표
교사 LLM으로 (문서 청크 → triple) 합성 데이터를 만들고, 품질을 정량적으로 검증한다.

## 데이터셋
- **UltraDomain** 원문 청크 — 교사 LLM 입력 (상세: `graphrag_00_overview.md` §2)
- **DocRED** — 교사 LLM이 만든 triple의 정확도를 비교할 "정답이 있는" 대조군. 96종 닫힌 스키마라 관계어 정렬(mapping) 단계가 반드시 필요

## 설계 개요
- 교사 모델: 로컬 vLLM 서빙 강모델(Qwen2.5-32B-Instruct-AWQ 등)을 few-shot 프롬프트로 호출해 (텍스트 청크 → 엔티티/관계 triple) 페어 생성
- Triple 스키마(전 phase 통일): `(entity1, entity1_type, relation, entity2, entity2_type, source_span, confidence)` — entity_type은 엔티티 정규화/그래프 노드 타입에, source_span은 환각률 검증에 필요
- 생성량: GPU-시간 기준 스코프 (파일럿 100 → 본생성 500~1,000 청크). 5K 규모 ablation은 시간 여유 있을 때만 선택적
- **(확정) API 예산 ~10,000원(~$7) 배분**: (a) GPT-4o로 고정 평가 서브셋(150~200청크, Phase 1.2-f) 전체를 재추출해 로컬 교사(Qwen2.5-32B-AWQ) 출력과 비교, 품질 격차 정량화(예상 ~1,300원, Phase 1.6-e). (b) GPT-4o를 인간 검수 골드셋(Phase 1.6-d, 30~50 triple)의 2차 라벨러로 투입해 저자 라벨과의 Cohen's κ 산출(예상 ~300원, Phase 1.6-f). 나머지(~8,000원+)는 파싱 실패 재시도용 버퍼로만 보류
- confidence는 모델 자기보고 점수가 아니라 **동일 청크 2~3회 재생성 후 self-consistency(일치도) 기반**으로 정의 (보정 문제 회피, 로컬 모델이라 반복 호출이 API 비용 없이 가능)

> **개정 (methodology 리뷰 반영, 순환논증 방지)**: 학생 모델의 학습 신호(교사 출력)와 서브3의 평가 정답(교사 출력)이 원래 동일했던 문제를 보완하기 위해, 이 서브프로젝트에서 **인간 검수 골드셋(30~50 triple)**을 별도로 만든다 (Phase 1.6). 이 골드셋이 있어야 `graphrag_03_graph_construction.md` Phase 3.7에서 "교사-학생 일치율"과 분리된 "실제 정확도"를 산출할 수 있다. 또한 고정 평가 서브셋을 50~100청크 → **150~200청크**(도메인당 약 8~11개)로 확대해 도메인별 분석의 표본 크기 문제를 완화했다.

---

## Phase 0.0 (선행) — 로컬 교사 모델 서빙
`graphrag_00_overview.md` §5 참고. 이 서브프로젝트의 모든 Phase는 Phase 0.0의 vLLM 엔드포인트를 전제한다.

## Phase 1.0 — 환경 & 스키마 준비
- **1.0-a** (S) UltraDomain HF `datasets` 로더 스크립트 작성, 18개 도메인 서브셋 목록 확인
- **1.0-b** (S) DocRED 로더 스크립트 작성 (train_annotated/dev/test split 구조 확인, relation id ↔ 이름 매핑 테이블 로드)
- **1.0-c** (S) Triple 스키마를 Pydantic/JSON Schema로 고정, 버전 태그(`schema_v1`) 부여
- **1.0-d** (M) 로컬 교사 모델(Phase 0.0 엔드포인트) 호출 wrapper 작성: 배치 큐잉, 실패 시 재시도, 호출별 소요시간 로깅(비용 대신 GPU-시간 추적)
- Done when: 두 데이터셋 모두 100개 샘플을 로컬에 캐시하고, 스키마 검증 유닛테스트 통과
- **산출물**: 데이터 로더 스크립트, 스키마 문서(JSON Schema)

## Phase 1.1 — 프롬프트 설계 & 파일럿
- **1.1-a** (M) few-shot 예시 3~5개 수작업 제작 (도메인 다양성 고려: 서사/기술/금융 각 1개 이상)
- **1.1-b** (S) 출력 강제 포맷(JSON mode 또는 구조화 출력) 적용, 파싱 실패시 재시도 로직(최대 2회) 구현
- **1.1-c** (M) 20~30개 청크로 파일럿 실행, 실패율/파싱 오류 유형 수기 분류
- **1.1-d** (S) 파일럿 결과 기반 프롬프트 1차 개정 (오류 패턴 반영)
- Done when: 파일럿 파싱 성공률 ≥95%, 실패 사례 로그 문서화
- **산출물**: 확정 프롬프트, 파서 코드, 파일럿 결과 노트

## Phase 1.2 — UltraDomain 대량 생성
- **1.2-a** (S) 청크 샘플링 전략 확정 (도메인별 균등 vs 랜덤) — Phase 0.0 처리량 벤치마크 기준으로 500~1,000개가 GPU 시간상 현실적인지 재확인
- **1.2-b** (M) 배치 실행 파이프라인: 체크포인트 저장(N개마다), 중단 후 재개 가능하도록 설계
- **1.2-c** (S) 청크 해시 기반 캐싱 (동일 청크 재요청 방지)
- **1.2-d** (L) 500~1,000개 청크 실제 실행 (GPU-시간 추적, API 비용 없음)
- **1.2-e** (S) 원본 청크 ↔ 생성 triple 매핑을 parquet/JSON lines로 저장
- **1.2-f** (S) 도메인별 균등 샘플 **150~200개**(도메인당 약 8~11개)를 **고정 평가 서브셋**으로 별도 태깅·저장 (`graphrag_03_graph_construction.md` Phase 3.7에서 재사용)
- Done when: 목표 청크 수 완료 + 고정 평가 서브셋 확정, 실행 로그(총 GPU-시간, 실패율) 요약 리포트 존재
- **산출물**: 합성 triple 데이터셋(JSON, 원시 버전) + 고정 평가 서브셋 목록(150~200청크)

## Phase 1.3 — DocRED 대조군 실행 & 관계어 정렬
- **1.3-a** (S) DocRED 문서를 UltraDomain 청크와 동일 길이 단위로 재분할 (공정 비교를 위해)
- **1.3-b** (M) 동일 프롬프트/파이프라인으로 DocRED 문서 실행 (표본 수는 1.2와 비례하거나 전체 test split)
- **1.3-c** (S) 엔티티 alias resolution: DocRED의 coreference 정보와 교사 모델 출력 엔티티 문자열 정렬 로직
- **1.3-d** (M) **관계어 정렬**: 교사 모델의 자유생성 관계어(예: "founded by")를 DocRED 96종 스키마(예: P112)로 매핑하는 로직 구현. 방식: (a) 수작업 매핑 테이블 우선 작성 후 (b) 커버 안 되는 관계어는 임베딩 유사도로 최근접 DocRED 관계에 매핑, 매핑 신뢰도 낮으면 "unmapped"로 분리. 이 단계 없이 넘어가면 스키마 불일치로 정확도가 인위적으로 낮게 나옴
- **1.3-e** (S) 매핑 커버리지 확인 (전체 교사 관계어 중 몇 %가 DocRED 스키마로 매핑되었는지)
- Done when: DocRED 문서 예측 결과 JSON 저장 + 관계어 정렬 매핑 테이블 + 매핑 커버리지 리포트 완성
- **산출물**: DocRED 위 교사 모델 예측 결과(JSON) + 관계어 정렬 매핑 테이블

## Phase 1.4 — 정확도 상한선 정량화
- **1.4-a** (S) 매칭 규칙 정의: exact match vs relation-type-only match vs partial(엔티티만 일치) 3단계로 분리 채점 (정렬된 관계어 기준)
- **1.4-b** (M) Precision/Recall/F1 계산 스크립트, 도메인·relation type별 breakdown
- **1.4-c** (S) 오류 유형 수동 샘플링(50건) 분류: hallucination / schema 불일치 / relation 방향 오류 / 매핑 실패 등
- **1.4-d** (S) 이 수치를 "닫힌 스키마 RE 벤치마크 참고 성능"으로 명확히 라벨링 — UltraDomain 자체 품질은 `graphrag_03_graph_construction.md` Phase 3.7의 골드셋 정확도로 별도 검증됨을 명시
- Done when: 정량 리포트 + 오류 유형 분포 표 완성
- **산출물**: 교사 모델 상한선 리포트 (수치 + 오류 유형 분석)

## Phase 1.5 — 필터링 규칙 설계
- **1.5-a** (S) Rule 기반 필터 구현: 빈 triple, self-loop, 타입 불일치 제거
- **1.5-b** (M) confidence 산출: 동일 청크 2~3회 재생성 후 self-consistency 일치도로 계산
- **1.5-c** (M) 통계 기반 필터: relation 빈도 이상치, confidence 임계값 스윕(0.3/0.5/0.7) 별 영향 측정
- **1.5-d** (S) 필터링 모듈 유닛테스트 (edge case: 빈 문자열, 중복 대소문자 등)
- Done when: 필터 모듈 코드 + confidence 산출 로직 문서 + 임계값별 영향 로그
- **산출물**: 필터링 모듈 코드, confidence 산출 로직 문서

## Phase 1.6 — 필터 전후 비교 & 최종 데이터셋 확정
- **1.6-a** (S) 필터 적용 전/후 데이터 규모, 노이즈 비율 비교표
- **1.6-b** (S) train/val split (도메인 균형 고려한 stratified split, 고정 평가 서브셋은 train에서 제외)
- **1.6-c** (S) 데이터셋 버전 태깅 및 카드(datasheet) 작성: 생성 모델(로컬 Qwen2.5 버전), 날짜, 규모, 알려진 한계(로컬 교사가 proprietary 대비 약할 수 있음 명시)
- **1.6-d** (M, 순환논증 방지) 고정 평가 서브셋(1.2-f) 중 30~50 triple을 무작위 층화 추출해 저자 본인이 원문 `source_span` 대조로 직접 gold 라벨링(정답 엔티티/관계 여부 수기 판정). 이 골드셋이 있어야 `graphrag_03_graph_construction.md` Phase 3.7에서 "교사-학생 일치율"과 분리된 "실제 정확도"를 산출 가능
- **1.6-e** (S, 신규 — API 예산 배분 ①) GPT-4o로 고정 평가 서브셋(150~200청크) **전체**에 대해 1.1에서 확정한 동일 프롬프트/스키마로 triple 재추출 실행, 로컬 교사(Qwen2.5-32B-AWQ) 출력과 recall/precision 비교 → 로컬 교사 품질 상한을 실측치로 검증. 예산: ~1,300원
- **1.6-f** (S, 신규 — API 예산 배분 ②) GPT-4o를 2차 라벨러로 투입: 1.6-d의 골드셋 30~50 triple 각각에 대해 GPT-4o가 독립적으로 correct/incorrect 판정(저자 라벨은 노출하지 않음), 두 라벨 간 Cohen's κ 계산 → 단일 annotator 한계 부분 보강(완전한 2인 human 교차검증 대체는 아님, datasheet에 명시). 예산: ~300원
- Done when: 최종 JSON 데이터셋 파일 + datasheet 문서 + 인간 검수 골드셋(30~50 triple, 라벨 포함) + GPT-4o sanity-check 비교 리포트(1.6-e) + 골드셋 inter-annotator agreement(κ, 1.6-f) 모두 존재, `graphrag_02_distillation.md`에 바로 투입 가능
- **산출물**: 최종 합성 데이터셋(JSON) + 품질 리포트(정밀도/재현율 vs DocRED, 필터 효과) + 인간 검수 골드셋(30~50 triple, 라벨 포함) + GPT-4o sanity-check 리포트 + 골드셋 inter-annotator agreement(κ)

---

## Ablation 설계 (GPU-시간 기준으로 스코프, API 비용 없음)
- 합성 데이터 양에 따른 성능 곡선 (100 / 500 / 1,000 pairs — 로컬 GPU 생성 속도 고려해 5K는 시간 여유 시 선택적 확장)
- 교사 모델 크기 변화 (로컬 14B vs 32B 양자화 모델이 만든 데이터 품질 비교 — 3090 한 장으로 실행 가능한 범위로 조정, 원래의 70B/8B는 단일 3090으로는 비현실적)
- 필터링 유무 비교

---

## 코드 매핑

```
src/data_gen/
├── teacher_prompt.py      # few-shot 프롬프트 템플릿 (Phase 1.1)
├── teacher_serve.py       # 로컬 강모델 vLLM 서빙 기동 스크립트, OpenAI 호환 엔드포인트 (Phase 0.0)
├── generate.py            # 로컬 교사 모델 호출 배치 스크립트 (Phase 1.2)
├── docred_align.py        # 교사 자유 관계어 ↔ DocRED 96종 스키마 정렬(매핑) (Phase 1.3-d)
├── filter.py              # 중복/노이즈 필터링, self-consistency 기반 confidence 산출 (Phase 1.5)
└── gpt4o_verify.py        # GPT-4o sanity-check(1.6-e) + 골드셋 2차 라벨러(1.6-f), 예산 캡 ~10,000원 하드코딩
```

- 입력: 텍스트 청크 → 로컬 강모델 vLLM 서빙(`teacher_serve.py`) 호출 — API 비용 없음, GPU 순차 처리 시간만 소요
- 출력 스키마 고정(JSON), 파싱 실패 시 재시도 로직, 배치 처리 + 캐싱(같은 청크 재생성 방지)
- (확정, ~10,000원 예산) GPT-4o로 2차 검증(Phase 1.6-e/f) — 단, "완전 LLM-free"라는 논문 주장과 충돌하지 않도록 **오프라인 학습 데이터 생성/검증 단계**에만 API를 쓰고, **인덱싱(추론) 단계**에는 절대 유료 API·강모델을 쓰지 않는다는 경계를 코드/논문 모두에서 명확히 구분

### 기술 스택
| 영역 | 도구 |
|---|---|
| 로컬 교사 모델 서빙 | `vllm` (OpenAI 호환 엔드포인트, 4bit AWQ/GPTQ 양자화) |
| 합성데이터 생성 | 로컬 vLLM 서빙 모델 (API 비용 없음) |
| 검증(sanity-check + 골드셋 2차 라벨러) | GPT-4o API, 예산 캡 ~10,000원(Phase 1.6-e/f 전용, `gpt4o_verify.py`) |

---

## 이 파트의 한계
- 합성 데이터의 교사 모델 편향이 학생 모델(`graphrag_02_distillation.md`)에 전이될 위험
- **로컬 오픈소스 교사(32B급)는 GPT-4o보다 약할 수 있어, 합성 데이터 품질 상한 자체가 낮을 가능성** (예산 제약에 따른 의도적 트레이드오프) — Phase 1.6-e에서 격차를 실측하지만, 격차가 확인되어도 생성 규모(500~1,000청크)를 유료 API로 대체하지는 않음
- **인간 검수 골드셋(30~50 triple)은 저자 본인(1차) + GPT-4o(2차, Phase 1.6-f)로 라벨링** — Cohen's κ로 일치도는 확보하지만, 완전한 2인 human 교차검증(전문가 2인 독립 라벨링) 대비로는 여전히 약한 검증

## 산출물 총괄
합성 학습 데이터셋(JSON) + 데이터 품질 리포트(정밀도/재현율 vs DocRED, 관계어 정렬 매핑 테이블) + 고정 평가 서브셋(150~200청크) + 인간 검수 골드셋(30~50 triple, 라벨 포함) + GPT-4o sanity-check 비교 리포트 + 골드셋 inter-annotator agreement(κ)

## 다음으로 도와드릴 수 있는 것
- `data_gen/teacher_prompt.py`용 실제 few-shot 프롬프트 초안 작성
- 서브프로젝트 1의 실제 데이터 다운로드 + 전처리 코드
- Phase 1.3-d(DocRED 관계어 정렬) 로직 상세 설계
