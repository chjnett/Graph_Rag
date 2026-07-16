"""Phase 0.5-a0 — Baseline 처리량 사전 체크 (TODO.md Milestone 1).

MS GraphRAG / LightRAG를 로컬 vLLM 엔드포인트로 표준 인덱싱하며
1단계(문서 3개) → 2단계(문서 10~20개)로 확장, 문서당 소요시간/LLM 호출 수를 로깅하고
UltraDomain 전체(428개) 외삽 소요일수를 산출해 corpus_scope(full|subset) 결정을 돕는다.

실행 전제 (이 저장소에는 아직 없음 — 실제 실행은 로컬 GPU 환경에서):
  - Phase 0.0 vLLM 서버가 configs/eval.yaml의 teacher_endpoint에 떠 있을 것
  - MS GraphRAG(`pip install graphrag`) / LightRAG(`pip install lightrag-hku`) 설치
  - data/raw_corpus/ultradomain/ 아래 UltraDomain 문서(.txt/.md 등)가 받아져 있을 것

사용 예:
  python scripts/throughput_pilot.py --method ms_graphrag --stage 1
  python scripts/throughput_pilot.py --method lightrag --stage 2
  python scripts/throughput_pilot.py --decide   # 두 baseline의 stage2 리포트를 읽어 corpus_scope 결정
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

# Windows 콘솔 기본 인코딩(cp949)에서 한글 출력이 UnicodeEncodeError로 죽는 것을 방지.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_CONFIG_PATH = REPO_ROOT / "configs" / "eval.yaml"
REPORTS_DIR = REPO_ROOT / "reports"
DEFAULT_CORPUS_DIR = REPO_ROOT / "data" / "raw_corpus" / "ultradomain"
ULTRADOMAIN_TOTAL_DOCS = 428


@dataclass
class DocTiming:
    doc_id: str
    wall_clock_sec: float
    llm_calls: int


@dataclass
class ThroughputReport:
    method: str
    stage: int
    doc_count: int
    doc_timings: list[DocTiming]
    avg_sec_per_doc: float
    avg_llm_calls_per_doc: float
    extrapolated_full_corpus_days: float


def load_eval_config() -> dict:
    with open(EVAL_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_endpoint(teacher_endpoint: str) -> bool:
    """Phase 0.0 vLLM 엔드포인트 응답 확인. 서브4는 이 서버를 소비만 하고 기동시키지 않는다."""
    import requests

    try:
        resp = requests.get(f"{teacher_endpoint}/models", timeout=5)
        return resp.ok
    except requests.RequestException as exc:
        print(f"[throughput_pilot] 엔드포인트 응답 없음: {teacher_endpoint} ({exc})")
        return False


def sample_docs(corpus_dir: Path, n: int, seed: int = 13) -> list[Path]:
    docs = sorted(corpus_dir.glob("*"))
    if len(docs) < n:
        raise FileNotFoundError(
            f"{corpus_dir}에 문서가 {len(docs)}개뿐입니다 (요청: {n}개). "
            "UltraDomain 코퍼스를 data/raw_corpus/ultradomain/에 먼저 받아두세요."
        )
    rng = random.Random(seed)
    return rng.sample(docs, n)


def index_one_doc(method: str, doc_path: Path, teacher_endpoint: str, teacher_model: str) -> DocTiming:
    """단일 문서를 method(ms_graphrag|lightrag)로 표준 인덱싱하고 소요시간/LLM 호출 수를 반환.

    실제 라이브러리 연동은 각 baseline이 설치된 환경에서 채워 넣는다.
    여기서는 스켈레톤만 제공 — 설치돼 있지 않으면 명확한 에러로 실패한다.
    """
    start = time.monotonic()
    if method == "ms_graphrag":
        try:
            import graphrag  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "graphrag 미설치. `pip install graphrag` 후 재시도하세요."
            ) from exc
        raise NotImplementedError(
            "MS GraphRAG 인덱싱 연동은 baselines/ms_graphrag_wrapper.py 완성 후 이 함수에서 호출한다."
        )
    elif method == "lightrag":
        try:
            import lightrag  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "lightrag 미설치. `pip install lightrag-hku` 후 재시도하세요."
            ) from exc
        raise NotImplementedError(
            "LightRAG 인덱싱 연동은 baselines/lightrag_wrapper.py 완성 후 이 함수에서 호출한다."
        )
    else:
        raise ValueError(f"알 수 없는 method: {method}")

    wall_clock_sec = time.monotonic() - start
    return DocTiming(doc_id=doc_path.name, wall_clock_sec=wall_clock_sec, llm_calls=0)


def run_stage(
    method: str, stage: int, corpus_dir: Path, config: dict, reports_dir: Path = REPORTS_DIR
) -> ThroughputReport:
    stage_doc_counts = config["throughput_pilot"]
    n = stage_doc_counts["stage1_doc_count"] if stage == 1 else stage_doc_counts["stage2_doc_count"]

    if not check_endpoint(config["teacher_endpoint"]):
        raise SystemExit(
            "vLLM 엔드포인트가 응답하지 않습니다. Phase 0.0을 먼저 완료하세요 (graphrag_00 담당)."
        )

    docs = sample_docs(corpus_dir, n)
    timings: list[DocTiming] = []
    for doc_path in docs:
        print(f"[throughput_pilot] {method} stage{stage}: indexing {doc_path.name} ...")
        timing = index_one_doc(method, doc_path, config["teacher_endpoint"], config["teacher_model"])
        timings.append(timing)
        print(f"  -> {timing.wall_clock_sec:.1f}s, llm_calls={timing.llm_calls}")

    avg_sec = sum(t.wall_clock_sec for t in timings) / len(timings)
    avg_calls = sum(t.llm_calls for t in timings) / len(timings)
    extrapolated_days = (avg_sec * ULTRADOMAIN_TOTAL_DOCS) / 86400

    report = ThroughputReport(
        method=method,
        stage=stage,
        doc_count=n,
        doc_timings=timings,
        avg_sec_per_doc=avg_sec,
        avg_llm_calls_per_doc=avg_calls,
        extrapolated_full_corpus_days=extrapolated_days,
    )
    _write_report(report, reports_dir)
    return report


def _write_report(report: ThroughputReport, reports_dir: Path = REPORTS_DIR) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / f"throughput_pilot_{report.method}_stage{report.stage}.json"
    payload = asdict(report)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[throughput_pilot] 리포트 저장: {out_path}")
    return out_path


def decide_corpus_scope(
    config: dict, reports_dir: Path = REPORTS_DIR, eval_config_path: Path = EVAL_CONFIG_PATH
) -> str:
    """두 baseline(ms_graphrag, lightrag)의 stage2 리포트를 읽어 corpus_scope를 결정하고
    eval_config_path에 기록한다 (TODO.md Milestone 1 '의사결정' 항목)."""
    threshold = config["throughput_pilot"]["extrapolation_threshold_days"]
    methods = ["ms_graphrag", "lightrag"]
    max_days = 0.0
    for method in methods:
        report_path = reports_dir / f"throughput_pilot_{method}_stage2.json"
        if not report_path.exists():
            raise SystemExit(
                f"{report_path} 없음 — 먼저 `--method {method} --stage 2`를 실행하세요."
            )
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
        max_days = max(max_days, report["extrapolated_full_corpus_days"])

    scope = "subset" if max_days > threshold else "full"
    config["corpus_scope"] = scope
    config["corpus_scope_decided_at"] = time.strftime("%Y-%m-%d")
    with open(eval_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

    print(
        f"[throughput_pilot] 결정: corpus_scope={scope} "
        f"(외삽 최대 {max_days:.1f}일 vs 임계값 {threshold}일)"
    )
    return scope


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["ms_graphrag", "lightrag"])
    parser.add_argument("--stage", type=int, choices=[1, 2])
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--decide", action="store_true", help="stage2 리포트 기반 corpus_scope 확정")
    args = parser.parse_args()

    config = load_eval_config()

    if args.decide:
        decide_corpus_scope(config)
        return

    if not args.method or not args.stage:
        parser.error("--method와 --stage를 지정하거나 --decide를 사용하세요.")

    run_stage(args.method, args.stage, args.corpus_dir, config)


if __name__ == "__main__":
    main()
