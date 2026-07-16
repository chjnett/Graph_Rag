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
├── metrics_hallucination.py# Evidence.source_span 대조 기반 환각률
├── difficulty_split.py     # GraphRAG-Bench 난이도별(4단계) 질문 분리 — Phase 4.1-a
└── report_table.py         # 논문용 Table 초안 + 난이도별 성능 분해 그룹화 — Phase 4.5-a/b
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
2. `deps_parsing_wrapper.py`의 대상 코드가 공개돼 있는지 미확인 (Phase 0.5-d) — 비공개 시 논문 기술 기반 최소 재현으로 스코프 축소. 원문 확인 결과(§9-2 참고), 공개 저장소 링크는 여전히 확인 안 됨 → 최소 재현 스코프 유지
3. LLM-as-judge 채점기 사용 여부 — 채점기 자체가 LLM 호출이라 별도 GPU-시간/비용으로 집계할지 결정 필요 (Phase 4.1-b)

## 9-2. `deps_parsing_wrapper.py` 대상 논문 아키텍처 (arXiv:2507.03226)

> "Towards Practical GraphRAG: Efficient Knowledge Graph Construction and Hybrid Retrieval at Scale" (Min et al.). 공개 코드 저장소는 arXiv 페이지 기준 확인 안 됨(2026-07 기준) — 아래는 논문 본문 기술 내용을 코드 재현 목적으로 정리한 것.

**핵심 주장**: 의존구문분석 기반 추출이 LLM(GPT-4o) 기반 추출 대비 **~94% 성능을 유지하면서 비용을 대폭 절감**. 이것이 서브4의 "로컬 강모델도 무거워서 못 쓴다 → 경량/LLM-free 대안" 논지와 같은 방향이라 삼각비교 앵커로 유용함.

**파이프라인 (논문 원문 기준)**:
1. **전처리**: Docling으로 문서 파싱(PDF/HTML 등 포맷 무관) → 계층적 청킹(섹션 경계 존중, 2048자 초과 시 재귀 분할) → spaCy 문장 분할 → 동사구 없는 문장 필터링
2. **추출**: spaCy 의존구문분석 트리에서 주어(`nsubj`/`nsubjpass`)-동사-목적어(`dobj`/`pobj`/`attr`) 구조로 triple 추출 (예: "SAP launched Joule for Consultants" → `(SAP, launched, Joule)`, `(Joule, for, Consultants)`)
3. **후처리**: 수동태 정규화, 다중토큰 엔티티 병합(예: "Supplier management"), 지시어 해석(coreference resolution), 짧은 엔티티(2자 미만)·불용어 제거
4. **엔티티 타입**: 도메인별 세분화 없이 전부 `type="Concept"`로 단일 처리

**원 논문 보고 수치 (삼각비교 앵커 후보, `reports/sub3_phase3_6c_anchor.json` 반영 예정)**:

| 지표 | 의존구문분석 | GPT-4o | 달성률 |
|---|---|---|---|
| Context Precision | 61.07% | 63.82% | 94% |
| Semantic Alignment | 61.87% | 65.83% | 94% |
| Full Coverage Rate | 51.08% | 58.99% | 86.6% |

**현재 재현 상태 vs 논문 원문 — 격차 (2026-07-16 기준)**: `baselines/deps_parsing_wrapper.py`는 위 2단계(SVO 추출)만 최소 재현했고, 1단계 전처리(문장 필터링)와 3단계 후처리(수동태 정규화/엔티티 병합/coreference/불용어 제거) 및 4단계 `Concept` 타입 부여는 아직 없음. 이 격차는 `TODO_mac.md`에서 관리.

## 9-3. `original_paper_anchor` 앵커 방법론 결정 (2026-07-16)

MS GraphRAG(arXiv:2404.16130)·LightRAG(arXiv:2410.05779) 원 논문을 §8 `original_paper_anchor` 컬럼(EM/F1 병기 전제)의 1차 소스로 쓰려 했으나, 조사 결과 두 논문 다 **EM/F1을 보고하지 않는다** — GPT-4(o)를 심사자로 쓴 LLM-as-judge 승률(comprehensiveness/diversity/empowerment)이고, 평가 코퍼스도 우리와 다르다(MS GraphRAG: 팟캐스트/뉴스 기사, LightRAG: UltraDomain 일부만 공유). 같은 컬럼에 숫자로 병기하면 서로 다른 지표를 동일선상에 놓는 오해를 부른다.

**결정**: GraphRAG-Bench 자체 논문(arXiv:2506.05690, "When to use Graphs in RAG")의 Table 2 — 이 논문이 GraphRAG-Bench 벤치마크로 MS GraphRAG/LightRAG를 직접 accuracy(%)로 재평가한 수치 — 를 1차 앵커로 채택. 우리 프로젝트가 쓰는 GraphRAG-Bench 데이터셋과 동일 조건이라 직접 비교 가능. 원 논문의 win-rate 수치는 `reports/sub3_phase3_6c_anchor.json`의 `qualitative_notes`에 참고용으로만 병기(§8 `original_paper_anchor` 컬럼에는 넣지 않음).

**미해결**: HotpotQA/MultiHop-RAG용 MS GraphRAG·LightRAG EM/F1 앵커는 아직 못 찾음 — 두 원 논문 모두 이 데이터셋으로 평가하지 않음. 제3자 재평가 논문이 있는지 추가 조사 필요.

**검증 필요**: `reports/sub3_phase3_6c_anchor.json`의 GraphRAG-Bench 수치는 arXiv HTML 파싱 기반으로 수집됨 — 논문에 실제 인용하기 전 PDF 원문 Table 2와 대조 재확인 권장.
