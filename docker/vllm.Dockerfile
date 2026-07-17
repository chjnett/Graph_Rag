# 로컬 교사 LLM 서빙용 vLLM 컨테이너 — graphrag_00_overview.md §5 (Phase 0.0) 기준.
#
# WSL2 네이티브 conda 환경에서 vLLM을 직접 띄우려다 다음 문제들을 겪었다:
#   1) UVA is not available (WSL2 pinned memory 기본 비활성화)
#   2) KV cache 메모리 부족 (max-model-len이 VRAM 예산보다 큼)
#   3) Could not find nvcc (CUDA 툴킷 미설치)
#   4) gcc 15 미지원 + curand.h 누락 (flashinfer 커널 JIT 컴파일 실패)
# vLLM 공식 이미지는 이미 호환되는 CUDA devel 툴체인을 갖추고 있어 3), 4)를 원천 차단한다.
# 1)은 컨테이너도 WSL2 커널을 공유하므로 여전히 필요 (VLLM_WSL2_ENABLE_PIN_MEMORY=1, compose에서 설정).

FROM vllm/vllm-openai:v0.25.1

# spec.md §2 "VRAM 안전장치" 원칙: 여기서 강제하지 않고 compose/entrypoint의 커맨드 인자로 넘긴다
# (corpus_scope처럼 런타임에 바뀔 수 있는 값을 이미지에 하드코딩하지 않기 위함).
EXPOSE 8000
