# SPEC — 서브프로젝트 4: Baseline 재현 & 다운스트림 QA 평가

> 개발용 기술 명세. 연구 설계는 `graphrag_04_evaluation.md`가 원본이며 이 문서는 그것의 실행 가능한 인터페이스 버전이다.
> 리소스/API 예산 전제는 `graphrag_00_overview.md` §리소스 제약 참고.

## 0. 목표 / 비목표

**목표**: 우리 방법(경량 증류 모델 기반 GraphRAG)과 5종 baseline을 동일 조건(로컬 LLM 인덱싱)에서 비용·정확도·환각률로 비교하고, 원 논문 수치를 삼각비교 앵커로 병기한 최종 Table을 만든다.

**비목표**:
- 데이터 생성(서브1), 모델 증류(서브2), 그래프 구축 자체(서브3)는 이 스펙 범위 밖 — 이 모듈은 그 산출물을 **소비**만 한다.
- baseline 논문 저자 코드를 개선/최적화하지 않는다. 로컬 엔드포인트로 config만 바꿔 재현.

## 1. 업스트림 의존성 (계약)

이 모듈이 실제로 동작하려면 아래 산출물이 먼저 존재해야 한다. 개발은 목(mock)으로 먼저 시작 가능.

| 산출물 | 출처 | 파일 형식 | 스키마 |
|---|---|---|---|
| Triple 스키마 | `graphrag_01` Phase 1.0-c | JSON | `(entity1, entity1_type, relation, entity2, entity2_type, source_span, confidence)` |
| 고정 평가 서브셋 (150~200청크, 교사 참조 triple) | `graphrag_01` Phase 1.2-f | JSONL | 위 triple 스키마 + `chunk_id`, `domain` |
| 인간 검수 골드셋 (30~50 triple) | `graphrag_01` Phase 1.6-d | JSONL | 위 + `human_label: correct\|incorrect` |
| GPT-4o sanity-check 리포트 | `graphrag_01` Phase 1.6-e | JSON | teacher vs GPT-4o recall/precision |
| 골드셋 inter-annotator agreement(κ) | `graphrag_01` Phase 1.6-f | JSON | `{cohen_kappa: float, n: int}` |
| 학생 모델 체크포인트 | `graphrag_02` Phase 2.6 | HF checkpoint dir | — |
| 그래프 DB 인스턴스 | `graphrag_03` Phase 3.3 | NetworkX pickle / Neo4j | 노드: `{id, type, aliases[]}`, 엣지: `{relation, source_span, confidence}` |
| 검색 인터페이스 | `graphrag_03` Phase 3.35 | Python 함수 | §4 `query()` 시그니처와 동일해야 함 |
| 교사-학생 일치율 / 골드셋 실제정확도 리포트 | `graphrag_03` Phase 3.7 | JSON | `{agreement: {...}, gold_accuracy: {...}}` |
| 도메인 적응 체크포인트 (선택, 5.4용) | `graphrag_05` Phase 5.2 | HF checkpoint dir | Fin/Leg 각 1개 |

**개발 순서 원칙**: Phase 0.5(baseline wrapper)는 위 표와 무관하게 즉시 착수 가능 — baseline은 자체 코퍼스로 독립 인덱싱한다. Phase 4.1 이후는 위 산출물이 실제로 있어야 실행 가능(아래 §6 참고).

## 2. 환경 / 인프라 전제

- **교사/baseline 인덱싱 엔드포인트**: `graphrag_00` Phase 0.0의 vLLM OpenAI 호환 서버. 기본값 `http://localhost:8000/v1`, 모델 `Qwen2.5-32B-Instruct-AWQ`
- **VRAM 안전장치 (Phase 0.0-e, 필수 준수)**: vLLM `--gpu-memory-utilization 0.90`, 청크 크기 500~1,000자 상한. 이 값을 넘기면 baseline 인덱싱 중 OOM 위험 — config에서 강제
- **API 예산**: GPT-4o 관련 항목(§1의 sanity-check/κ)은 서브1 소관. 서브4는 이 예산을 쓰지 않는다(baseline 인덱싱은 전부 로컬 LLM)
- **단일 GPU 제약**: 5개 baseline + 우리 방법을 **순차 실행**해야 함. 동시 실행 시 VRAM 경합으로 둘 다 실패할 수 있으므로 `benchmark.py`는 기본적으로 method 하나씩 직렬 처리

## 3. 코퍼스 범위 — 미확정 변수

`graphrag_04` Phase 0.5-a0의 처리량 사전 체크 결과에 따라 결정되는 **런타임 스위치**다. 코드는 아래 두 값을 모두 지원하도록 설계한다(하드코딩 금지).

```yaml
corpus_scope: full     # UltraDomain 428개 전체
# 또는
corpus_scope: subset   # GraphRAG-Bench/HotpotQA/MultiHop-RAG 질문이 실제로 걸쳐 있는 문서만
```

`corpus_scope`는 Phase 0.5-a0 완료 시점에 `configs/eval.yaml`에 값이 채워진다. 그 전까지 코드는 `subset`을 기본값(안전한 쪽)으로 가정하고 개발한다.

## 4. 공통 QA 인터페이스 (Phase 0.5-f)

5개 baseline wrapper + 우리 방법이 전부 이 시그니처를 구현한다. `benchmark.py`는 이 인터페이스에만 의존하고 구현 세부는 모른다.

```python
from dataclasses import dataclass

@dataclass
class Evidence:
    source_span: str
    doc_id: str
    chunk_id: str | None = None

@dataclass
class QAResult:
    answer: str
    evidence: list[Evidence]
    latency_ms: float
    indexing_llm_calls: int   # 인덱싱 시점 LLM 호출 수. 우리 방법/LiteSemRAG/NoLLMRAG는 항상 0

class GraphRAGMethod:
    """모든 wrapper(baselines/*.py, src/graph_construction 기반 우리 방법)가 구현하는 공통 인터페이스."""

    def index(self, corpus_path: str, scope: str) -> "IndexStats":
        """코퍼스를 인덱싱. scope: 'full' | 'subset'. 반환값에 GPU-시간/LLM 호출 수 포함."""
        ...

    def query(self, question: str, top_k: int = 5) -> QAResult:
        """단일 질의에 대해 검색+응답 생성까지 end-to-end 수행."""
        ...

@dataclass
class IndexStats:
    wall_clock_sec: float
    gpu_hours: float
    llm_calls: int
    node_count: int | None = None
    edge_count: int | None = None
```

`graphrag_03` Phase 3.35의 검색 인터페이스는 `GraphRAGMethod.query()`를 감싸는 형태로 노출되어야 한다 — 별도 함수명으로 노출하지 말 것(3.35-a/b/c 로직은 내부 구현, 외부 계약은 이 클래스 하나).

## 5. 모듈 구조

```
baselines/
├── ms_graphrag_wrapper.py     # GraphRAGMethod 구현 — Phase 0.5-a
├── lightrag_wrapper.py        # GraphRAGMethod 구현 — Phase 0.5-b
├── litesemrag_wrapper.py      # GraphRAGMethod 구현 — Phase 0.5-c (LLM-free, indexing_llm_calls=0 고정)
└── deps_parsing_wrapper.py    # GraphRAGMethod 구현 — Phase 0.5-d (arXiv:2507.03226)
    # NoLLMRAG는 0.5-e, 별도 wrapper 파일 없으면 litesemrag_wrapper.py에 인접 배치 가능
src/eval/
├── benchmark.py            # 통합 러너 — GraphRAGMethod 리스트를 받아 순차 실행, 결과 집계
├── metrics_cost.py         # IndexStats → 비용 비교 테이블
├── metrics_coverage.py     # 교사-학생 일치율 (graphrag_03 Phase 3.7 리포트 소비, 재계산 아님)
├── metrics_gold_accuracy.py# 골드셋 실제 정확도 (동일)
├── metrics_qa.py           # QAResult 리스트 → EM/F1/LLM-as-judge, 원 논문 앵커 병기
└── metrics_hallucination.py# Evidence.source_span 대조 기반 환각률
configs/
└── eval.yaml                # §7 스키마
```

## 6. 실행 순서와 블로킹 관계

```
Phase 0.5-a0 (처리량 체크) ── 독립적으로 즉시 착수 가능
Phase 0.5-a~f (wrapper 5종 + 공통 인터페이스) ── 독립적으로 즉시 착수 가능
        │
        ▼
Phase 4.1~4.3 (QA 벤치마크 실행) ── graphrag_03 Phase 3.35(검색 인터페이스) + Phase 3.7(일치율/실정확도 리포트) 필요
        │
        ▼
Phase 4.4 (비용 집계) ── Phase 0.5 IndexStats + graphrag_01~03 GPU-시간 로그 필요
        │
        ▼
Phase 4.5 (종합 Table) ── graphrag_03 Phase 3.6-c(원 논문 앵커) 필요
```

**즉, 지금 바로 코딩 가능한 것**: `GraphRAGMethod` 인터페이스 정의, 5개 baseline wrapper의 `index()`/`query()` 뼈대(로컬 엔드포인트 연동까지), `metrics_*.py`의 함수 시그니처와 목 데이터 기반 유닛테스트. `benchmark.py`의 실제 end-to-end 실행은 서브1/서브3 산출물이 나온 뒤.

## 7. `configs/eval.yaml` 스키마

```yaml
teacher_endpoint: "http://localhost:8000/v1"
teacher_model: "Qwen2.5-32B-Instruct-AWQ"
gpu_memory_utilization: 0.90       # Phase 0.0-e
chunk_size_chars: [500, 1000]      # Phase 0.0-e

methods: [ours, ms_graphrag, lightrag, litesemrag, deps_parsing, nollmrag]
datasets: [graphrag_bench, hotpotqa, multihop_rag]

corpus_scope: subset               # full | subset — Phase 0.5-a0 의사결정 규칙으로 확정
corpus_scope_decided_at: null      # 0.5-a0 완료 시 ISO 날짜 기입

fixed_eval_subset_path: data/synthetic/fixed_eval_subset.jsonl   # graphrag_01 Phase 1.2-f
gold_set_path: data/synthetic/gold_set.jsonl                     # graphrag_01 Phase 1.6-d/f
agreement_report_path: reports/sub3_phase3_7.json                # graphrag_03 Phase 3.7
original_paper_anchor_path: reports/sub3_phase3_6c_anchor.json   # graphrag_03 Phase 3.6-c

seeds: [13, 27, 42]                # Phase 4.1-c 다중 시드
bootstrap_n: 1000                  # 신뢰구간용, seed 대안/보조

throughput_pilot:                  # Phase 0.5-a0 전용
  stage1_doc_count: 3
  stage2_doc_count: 20
  extrapolation_threshold_days: 14 # 이 값 초과 시 corpus_scope=subset 강제
```

## 8. `benchmark.py` 출력 스키마

```json
{
  "method": "ours",
  "dataset": "graphrag_bench",
  "difficulty": "multi_hop",
  "em": 0.61,
  "f1": 0.68,
  "ci95": [0.58, 0.64],
  "teacher_student_agreement": 0.74,
  "gold_accuracy": {"precision": 0.71, "recall": 0.66},
  "hallucination_rate": 0.09,
  "gpu_hours": 12.4,
  "indexing_llm_calls": 0,
  "original_paper_anchor": null,
  "corpus_scope": "subset"
}
```

- `gold_accuracy`, `original_paper_anchor`는 baseline row에서는 각각 `null`/원 논문 수치, 우리 방법 row에서만 `gold_accuracy`가 채워짐 (baseline은 별도 골드셋 채점 대상이 아님 — 순환논증 방지 지표는 우리 방법 전용)
- `indexing_llm_calls`는 검증용 컬럼: 우리 방법·LiteSemRAG·NoLLMRAG는 항상 `0`이어야 하고, 0이 아니면 버그

## 9. 미확정 사항 (개발 중 명시적으로 TODO로 남길 것)

1. `corpus_scope` 최종값 — Phase 0.5-a0 완료 전까지 `subset` 가정, 실측 후 갱신
2. `deps_parsing_wrapper.py`의 대상 코드가 공개돼 있는지 미확인 (Phase 0.5-d) — 비공개 시 논문 기술 기반 최소 재현으로 스코프 축소
3. LLM-as-judge 채점기 사용 여부 — 채점기 자체가 LLM 호출이라 별도 GPU-시간/비용으로 집계할지 결정 필요 (Phase 4.1-b)
