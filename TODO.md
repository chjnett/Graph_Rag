# TODO — 서브프로젝트 4 개발 착수용 (spec.md 기준)

> 체크박스 = 실제 코딩 단위. (S)/(M)/(L) = 원문서(`graphrag_04_evaluation.md`) 사이즈 태그 그대로.
> "🔓 즉시 가능" = 다른 서브프로젝트 산출물 없이 지금 시작 가능. "🔒 블록" = 명시된 산출물이 있어야 실행(코딩은 미리 가능, 실행/검증은 블록).

## Milestone 0 — 인프라 확인 (🔓, 최우선)

- [ ] Phase 0.0 vLLM 엔드포인트가 이미 떠 있는지 확인 (`graphrag_00` 담당, 서브4는 소비만) — **진행 중, 접근법 전환**: GPU는 RTX 3090 24GB 확인됨(driver 591.86, CUDA 13.1). WSL2 Ubuntu 26.04 + conda(Python 3.11) 네이티브 방식으로 4차례 시도했으나 순차적으로 (1) UVA 비활성화 (2) KV 캐시 예산 부족 (3) nvcc 미설치 (4) gcc 15 비호환 + curand.h 누락(flashinfer 커널 JIT 컴파일 실패)까지 해결했지만 계속 새 트러블슈팅 필요 → **Docker(vLLM 공식 이미지 `vllm/vllm-openai:v0.25.1`)로 전환 결정**(사용자 지시). `docker/vllm.Dockerfile` + `docker/docker-compose.yml` + `docker/setup_docker_wsl.sh` 작성 완료. 모델 가중치(~19GB)는 WSL `~/.cache/huggingface`에 이미 받아둔 것을 컨테이너에 볼륨 마운트해 재사용. 교사 모델 **Qwen2.5-32B-Instruct-AWQ**, `--gpu-memory-utilization 0.90`, `--max-model-len 4096`(실측 기반 안전값)으로 compose에 고정. **남은 것(사용자가 sudo로 직접 실행 필요)**: `docker/SETUP_GUIDE.md`(WSL 초기화 옵션 + Docker Engine + NVIDIA Container Toolkit 설치 + vLLM 컨테이너 기동 + 트러블슈팅표 전부 포함, 자기 완결형 단일 문서)를 따라 진행 → 문서 끝 완료 체크리스트 전부 체크되면 이 항목도 완료 처리. **진단 이력**: 1차 시도 시 `get.docker.com` 원클릭 스크립트가 Docker Engine 설치는 실패하고 NVIDIA Container Toolkit만 설치된 채 조용히 끝남(원인 불명, 로그 없음) — apt 공식 문서 방식의 단계별 수동 설치로 교체, Ubuntu 26.04(resolute) codename의 Docker 저장소 실존 여부까지 사전 검증 확인함
- [x] `configs/eval.yaml` 스킵톤 작성 (spec.md §7 스키마 그대로) — `corpus_scope: subset` 기본값으로 시작
- [x] `GraphRAGMethod` / `QAResult` / `Evidence` / `IndexStats` 데이터클래스 구현 (spec.md §4) — `src/eval/interface.py`, 이후 모든 wrapper·metrics가 이 계약에 의존

## Milestone 1 — Phase 0.5-a0: 처리량 사전 체크 (🔓, 가장 리스크 큰 항목)

> `scripts/throughput_pilot.py` + `tests/test_throughput_pilot.py`(12개, 전부 pass) 작성 완료.
> 순수 로직(샘플링/리포트 저장/scope 의사결정)은 검증됨. 아래 실행 항목 자체는 로컬 vLLM+GPU+UltraDomain 코퍼스가 있어야 하는 🔒 상태라 미완.

- [ ] (S) 1단계: UltraDomain 무작위 3개 문서로 MS GraphRAG 표준 인덱싱을 로컬 엔드포인트로 실행, OOM/설정 문제 확인
- [ ] (S) 1단계 통과 시 2단계: 10~20개 문서로 확장, 문서당 소요시간·LLM 호출 수 로깅
- [ ] (S) LightRAG도 동일 절차로 별도 측정
- [ ] (S) 428개 전체 외삽 → 예상 소요일수 산출, `throughput_pilot` 결과를 `configs/eval.yaml`에 기록
- [ ] **의사결정**: baseline당 외삽치가 2주 초과 → `corpus_scope: full` → `subset`으로 변경, 그렇지 않으면 `full` 유지 — 로직/테스트는 `decide_corpus_scope()`로 준비됨, 실행만 남음
- [ ] Done when: 두 baseline의 처리량 벤치마크 리포트 + `corpus_scope` 확정값이 `eval.yaml`에 반영됨

## Milestone 2 — Phase 0.5: Baseline wrapper 5종 + 공통 인터페이스 (🔓)

- [ ] (L) `baselines/ms_graphrag_wrapper.py` — `GraphRAGMethod` 골격/설정 전달은 구현됨, `graphrag` 패키지 설치 후 실제 인덱싱 LLM 호출 지점 연동은 TODO로 남겨둠(추측 API로 채우지 않음)
- [ ] (M) `baselines/lightrag_wrapper.py` — 동일 상태(`lightrag-hku` 설치 후 실 연동 필요)
- [x] (M) `baselines/litesemrag_wrapper.py` — LLM-free 계열, `indexing_llm_calls=0` 하드코딩 + 유닛테스트로 고정(TF-IDF 코사인 유사도 기반 최소 재현, 공식 코드 미확인 명시)
- [x] (M) `baselines/deps_parsing_wrapper.py` — arXiv:2507.03226 공개 코드 미확인(spec.md §9-2) → spaCy 의존구문분석 SVO 추출로 스코프 축소, 파일 상단에 그 사실 기록
- [x] (M) NoLLMRAG 재현 — `litesemrag_wrapper.py`에 인접 배치(순수 co-occurrence 빈도 기반)
- [x] (S) 5개 wrapper 전부 `GraphRAGMethod.query()` 반환 타입이 `QAResult`인지 확인하는 계약 테스트 작성 — `tests/test_baseline_contracts.py`(10개, 9 pass + spaCy 미설치로 1 skip)
- [ ] Done when: 5개 baseline 전부 샘플 질의 1건에 정상 응답 — litesemrag/nollmrag/deps_parsing 3종은 충족, ms_graphrag/lightrag 2종은 미충족(패키지 미설치)

## Milestone 3 — `src/eval/metrics_*.py` 구현 (🔓, 목 데이터로 선행 가능)

- [x] (S) `metrics_cost.py` — `IndexStats` 리스트 → GPU-시간 비교 테이블, (선택) API 환산 비용 병기
- [x] (M) `metrics_coverage.py` — `graphrag_03` Phase 3.7 일치율 리포트를 읽어 그대로 노출 (재계산 금지, 순환논증 지표는 서브3 소관)
- [x] (M) `metrics_gold_accuracy.py` — 동일하게 서브3 Phase 3.7 골드셋 정확도 리포트 소비
- [x] (M) `metrics_qa.py` — `QAResult` 리스트 + 정답 → EM/F1, LLM-as-judge 옵션, 원 논문 앵커 컬럼 병합
- [x] (M) `metrics_hallucination.py` — `Evidence.source_span`을 원문과 대조해 환각률 계산
- [x] 각 스크립트에 목(mock) `QAResult`/`IndexStats` 기반 유닛테스트 작성 — `tests/test_metrics.py`(13개, 전부 pass), 서브1/서브3 실제 데이터 없이도 통과 확인됨

## Milestone 4 — `benchmark.py` 통합 러너 (🔓 골격 / 🔒 실제 실행)

- [x] (M) `configs/eval.yaml` 파싱 → `methods` 순차 실행 루프 (동시 실행 금지, VRAM 경합 방지) — `load_config`/`sequential_run`, 직렬 실행 순서를 테스트로 증명
- [x] (S) 결과를 spec.md §8 JSON 스키마로 직렬화 — `serialize_result_row`
- [x] (S) `gold_accuracy`/`original_paper_anchor`가 우리 방법에만 채워지고 baseline에는 null인지 검증하는 테스트
- [x] (S) `indexing_llm_calls != 0`인데 LLM-free 계열(litesemrag/nollmrag/ours)로 표시된 경우 assert로 실패시키는 가드 추가 — `tests/test_benchmark.py`(7개, 전부 pass)
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
