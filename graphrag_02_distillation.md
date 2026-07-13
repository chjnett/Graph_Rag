# 서브프로젝트 2 — 경량 추출 모델 SFT (증류)

> 리소스 전제는 `graphrag_00_overview.md` 참고.

## 목표
서브프로젝트 1의 합성 데이터로 경량 모델을 학습시켜, LLM 없이 triple을 추출하는 모델을 확보한다.

## 데이터셋
- `graphrag_01_data_pipeline.md` 산출물(주 학습 데이터)
- SciERC, WebNLG, FewRel — 일반화를 위한 보조 학습 데이터 (비율 실험 필요)
- DocRED test split — 최종 평가용, 라벨 공개되어 있어 외부 재현 가능

## 설계 개요
- 베이스 모델 후보: DeBERTa-v3 (span-based NER/RE), 또는 1B 이하 소형 생성 모델
- 학습 방식: QLoRA/LoRA 파인튜닝, label-to-span 방식(MetaIE 참고) 검토
- 하이퍼파라미터: lr, batch size, epoch 등 실험 로그화

---

## Phase 2.0 — 데이터 통합 설계
- **2.0-a** (S) SciERC/WebNLG/FewRel을 공통 triple 스키마(entity_type, source_span 포함)로 변환하는 어댑터 작성
- **2.0-b** (S) 라이선스 확인 (각 데이터셋 재배포/연구용 사용 조건 체크 — 논문에 명시)
- **2.0-c** (S) 혼합 비율 실험 설계표 작성 (0%/10%/30%/50%, 각 조합의 총 샘플 수 계산)
- Done when: 통합 로더 하나로 4개 데이터소스 모두 동일 포맷 산출 확인
- **산출물**: 통합 데이터 로더, 비율 실험 계획표

## Phase 2.1 — 모델/학습 인프라
- **2.1-a** (M) DeBERTa-v3 span-based 헤드 vs 1B급 seq2seq 두 후보 최소 구현체 프로토타입 (더미 데이터로 forward pass만 확인, 둘 다 3090 24GB에 여유 있게 적재됨)
- **2.1-b** (S) 선택 기준 명문화: 추론 속도, 정확도, 파인튜닝 난이도 3축 비교 기준표
- **2.1-c** (M) LoRA/QLoRA 설정(rank, alpha, target modules) 및 학습 스크립트 골격, config 기반 재현 가능하게
- **2.1-d** (S) wandb(무료 tier) 로깅 연동, 재현용 seed 고정
- Done when: 학습 스크립트가 100샘플 더미로 1 epoch 정상 종료
- **산출물**: 학습 파이프라인 코드(재현 가능한 config 포함)

## Phase 2.2 — 1차 학습 (주 데이터만)
- **2.2-a** (M) 서브프로젝트 1 데이터만으로 전체 학습 실행 (단일 GPU 순차 실행 시간 기록)
- **2.2-b** (S) Loss curve 검토, overfitting 징후 체크 (train/val gap)
- **2.2-c** (S) 조기 정성 검증: 임의 5개 문장에 대해 직접 추론 결과 육안 확인
- Done when: 베이스라인 체크포인트 저장 + 학습 로그 존재
- **산출물**: 베이스라인 체크포인트

## Phase 2.3 — 보조 데이터 비율 실험
- **2.3-a** (M) 각 비율(0/10/30/50%)별 반복 학습 실행 (동일 시드, 동일 epoch 수로 통제, 단일 GPU라 순차 실행 — 총 소요시간 사전 추정해 일정에 반영)
- **2.3-b** (S) 비율별 검증 성능 비교표 + 최적 비율 선정 근거
- Done when: 비율별 체크포인트 4개 + 비교 로그
- **산출물**: 비율별 체크포인트 + 비교 로그

## Phase 2.4 — DocRED 최종 평가
- **2.4-a** (S) 최적 비율 모델을 DocRED test split에 F1 평가 (`graphrag_01_data_pipeline.md` Phase 1.3-d와 동일한 관계어 정렬 로직 재사용)
- **2.4-b** (S) Phase 1.4 교사 모델 상한선과 나란히 비교하는 표 생성 → "교사 대비 학생 성능 유지율(%)" 확정 (핵심 결과물)
- Done when: 유지율 수치 확정
- **산출물**: DocRED 벤치마크 성능표

## Phase 2.5 — FewRel 일반화 테스트
- **2.5-a** (M) 미학습 relation 타입에 대한 zero-shot/few-shot 평가 실행
- **2.5-b** (S) 실패 사례(관계 유형별) 분석, 어떤 관계 카테고리가 특히 약한지 정리
- Done when: 일반화 리포트 + 약점 relation 유형 목록
- **산출물**: FewRel 일반화 리포트

## Phase 2.6 — 최종 모델 확정 & 문서화
- **2.6-a** (S) 최종 체크포인트 선정 근거 정리 (비율, 성능, 속도 종합)
- **2.6-b** (S) 모델 카드 작성 (학습 데이터, 하이퍼파라미터, 한계, 라이선스, 교사 모델이 로컬 오픈소스임을 명시)
- Done when: 공개 가능한 체크포인트 + 모델 카드 + 최종 벤치마크 표 확정 → `graphrag_03_graph_construction.md` 착수 가능
- **산출물**: 학생 모델 체크포인트(공개용) + DocRED/FewRel 벤치마크 성능표(최종)

---

## 코드 매핑

```
src/training/
├── sft_train.py      # QLoRA/LoRA SFT 학습 루프 (Phase 2.1~2.3)
├── model_def.py       # 학생 모델 아키텍처 (span-based or seq2seq) (Phase 2.1)
└── domain_adapt.py    # 2차 도메인 파인튜닝 (`graphrag_05_domain_adaptation.md`에서 사용)
```

### 기술 스택
| 영역 | 도구 |
|---|---|
| 학생 모델 학습 | HuggingFace `transformers`, `peft`, `bitsandbytes` (QLoRA, 단일 RTX 3090 기준) |
| 평가/실험관리 | `wandb` 또는 `mlflow` (무료 tier) |

## 산출물 총괄
학생 모델 체크포인트(공개용) + DocRED/FewRel 벤치마크 성능표 + 비율별 ablation 로그

## 다음으로 도와드릴 수 있는 것
- 학생 모델 학습 스크립트(`sft_train.py`) 실제 코드 작성
