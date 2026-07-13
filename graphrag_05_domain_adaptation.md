# 서브프로젝트 5 — 도메인 적응 실험

> 리소스 전제는 `graphrag_00_overview.md` 참고.

## 목표
범용 학생 모델 vs 도메인 특화 파인튜닝 모델을 비교한다 (JD의 "도메인 특화 파인튜닝" 요구사항 직접 대응).

## 데이터셋
UltraDomain 내 **Fin(금융)**, **Leg(법률)** 서브셋 — 이미 공개되어 있고 도메인이 명확히 분리되어 있어 별도 도메인 데이터 수집 불필요

## 설계 개요
- 범용 코퍼스로 1차 학습된 모델(`graphrag_02_distillation.md` 산출물) → 특정 도메인(Fin/Leg) 소량 데이터로 2차 파인튜닝
- Zero-shot(범용 모델) vs Domain-adapted 성능 비교가 핵심 실험
- 성능 격차는 triple 레벨뿐 아니라 다운스트림 QA 레벨까지 확인해야 함 — triple 지표만으로는 "도메인 적응이 실제 검색 품질에 도움되는가"에 답할 수 없기 때문

---

## Phase 5.0 — Fin/Leg 서브셋 준비
- **5.0-a** (S) UltraDomain Fin/Leg 서브셋 분리, train/eval split
- **5.0-b** (S) 스키마 정합성 확인 (서브프로젝트 1~2와 동일 포맷인지)
- Done when: Fin/Leg 학습·평가 데이터셋 파일 확정
- **산출물**: Fin/Leg 학습·평가 데이터셋

## Phase 5.1 — Zero-shot 베이스라인
- **5.1-a** (M) 범용 모델(`graphrag_02_distillation.md` Phase 2.6 산출물)을 Fin/Leg에 그대로 추론
- **5.1-b** (S) 라벨 없는 경우 대비 정성 평가 기준(수기 샘플 채점) 마련
- Done when: zero-shot 성능 결과 확정
- **산출물**: Zero-shot 성능 결과

## Phase 5.2 — 도메인 특화 파인튜닝
- **5.2-a** (M) Fin, Leg 각각 별도 어댑터로 추가 SFT (LoRA 계속 학습 or 신규 어댑터, Freeze 전략 실험: 전체 vs 마지막 레이어만)
- **5.2-b** (S) 소규모 하이퍼파라미터 탐색 (learning rate 2~3개 값)
- Done when: Fin/Leg 전용 체크포인트 각 1개 확정
- **산출물**: 도메인 적응 체크포인트(Fin, Leg 각각)

## Phase 5.3 — 재평가 & 격차 정량화 (triple 레벨)
- **5.3-a** (S) 동일 평가셋에서 zero-shot vs adapted 비교
- **5.3-b** (S) 통계적 유의성 검정 (bootstrap 또는 t-test), 표본 수가 작을 경우 신뢰구간으로 대체
- Done when: triple 레벨 ablation 표 + 유의성 검정 결과 확정
- **산출물**: 도메인 적응 전/후 triple 성능 비교표

## Phase 5.4 — QA 레벨 도메인 적응 검증
- **5.4-a** (M) Fin/Leg 도메인 특화 질문셋 구성 (GraphRAG-Bench 내 해당 도메인 질문 재사용 우선, 부족하면 소규모 수작업 제작)
- **5.4-b** (M) `graphrag_04_evaluation.md` Phase 4.1~4.2의 QA 평가 하네스를 재사용해 zero-shot vs adapted 모델의 QA 정확도 비교
- **5.4-c** (S) triple 레벨 격차(5.3)와 QA 레벨 격차(5.4)를 나란히 비교해 "triple 성능 향상이 실제 QA 향상으로 이어지는가" 해석
- Done when: 최종 ablation 표(triple + QA 레벨 모두 포함) 확정 — 논문 기여 C2의 핵심 근거
- **산출물**: 도메인 적응 전/후 QA 정확도 비교표

---

## 코드 매핑

```
src/training/domain_adapt.py   # 1차 학습된 체크포인트 로드 → 도메인 소량 데이터로 추가 파인튜닝, Freeze 전략 실험
```

### 기술 스택
`graphrag_02_distillation.md`와 동일 (HuggingFace `transformers`/`peft`/`bitsandbytes`)

---

## 이 파트의 한계
- 초정밀 도메인(의료/법률)에서는 여전히 (로컬이든 API든) LLM 대비 격차 존재 가능성 — Fin/Leg 실험에서도 유사한 격차가 관찰될 수 있음

## 산출물 총괄
Fin/Leg 도메인 적응 체크포인트 + triple 레벨 성능 비교표 + QA 레벨 성능 비교표 (논문의 핵심 ablation 중 하나, 기여 C2 근거)

## 다음으로 도와드릴 수 있는 것
- 각 서브프로젝트를 개별 논문/기술 리포트로 쪼갤지, 하나의 큰 논문으로 합칠지에 대한 전략 논의
