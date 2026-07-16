"""로컬 더미 OpenAI 호환 HTTP 서버 — TODO_mac.md #4.

실제 vLLM 없이 ms_graphrag_wrapper.py/lightrag_wrapper.py의 배선(HTTP 요청이
teacher_endpoint까지 실제로 도달하는지)만 검증하기 위한 최소 서버. 완성도 있는
추출 결과를 만들 필요는 없다 — graphrag/lightrag 양쪽 다 파싱 실패 시 빈
결과로 넘어가도록 방어적으로 짜여 있어(예: graphrag의 GraphExtractor), 응답
내용은 "형식만 맞으면" 충분하다.
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_EMBEDDING_DIM = 8

# graphrag의 GRAPH_EXTRACTION_PROMPT는 "-Goal-"로 시작하고 entity_type 플레이스홀더를
# 채워 넣는다 — 이 마커로 "지금 요청이 엔티티 추출 1차 호출인지"를 구분한다.
_ENTITY_EXTRACTION_RESPONSE = (
    '("entity"<|>APPLE<|>ORGANIZATION<|>Apple is a technology company)##'
    '("entity"<|>STEVE JOBS<|>PERSON<|>Steve Jobs founded Apple)##'
    '("relationship"<|>STEVE JOBS<|>APPLE<|>Steve Jobs founded Apple<|>9)##'
    "<|COMPLETE|>"
)
_GENERIC_CONTENT = "dummy response for wiring verification only."

# graphrag의 community_report 프롬프트는 litellm의 구조화 출력 스키마 검증을
# 거치므로(JSONSchemaValidationError) 실제 스키마(CommunityReportResponse)에
# 맞는 JSON 문자열을 돌려줘야 파이프라인이 죽지 않는다.
_COMMUNITY_REPORT_JSON = (
    '{"title": "Dummy Community", '
    '"summary": "A small dummy community for wiring verification only.", '
    '"findings": [{"summary": "Dummy finding", "explanation": "Dummy explanation."}], '
    '"rating": 5.0, "rating_explanation": "Dummy rating explanation."}'
)


def _chat_content_for(messages: list[dict]) -> str:
    """요청 프롬프트를 보고 graphrag/lightrag 파이프라인이 죽지 않을 만한 응답을 고른다.

    실제 추출 품질은 검증 대상이 아니다 — 형식만 맞춰 파싱 단계가 예외 없이
    끝까지 도는지(배선 검증)가 목적.
    """
    text = " ".join(m.get("content", "") for m in messages if isinstance(m, dict))

    if "Answer Y if" in text or "MANY entities and relationships were missed" in text:
        return "N"  # gleaning loop을 즉시 종료
    if "-Goal-" in text and "entity_type" in text:
        return _ENTITY_EXTRACTION_RESPONSE
    if "Write a comprehensive report of a community" in text:
        return _COMMUNITY_REPORT_JSON
    return _GENERIC_CONTENT


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002 - BaseHTTPRequestHandler 시그니처
        pass  # 테스트 출력 조용히 유지

    def _send_json(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler 시그니처
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            payload = {}

        server: DummyLLMServer = self.server  # type: ignore[assignment]
        server.request_count += 1

        if self.path.endswith("/embeddings"):
            inputs = payload.get("input", [""])
            if isinstance(inputs, str):
                inputs = [inputs]
            self._send_json(
                {
                    "object": "list",
                    "model": payload.get("model", "dummy-embedding"),
                    "data": [
                        {"object": "embedding", "index": i, "embedding": [0.01] * _EMBEDDING_DIM}
                        for i in range(len(inputs))
                    ],
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                }
            )
            return

        # chat/completions (그리고 그 외 알 수 없는 POST 경로도 동일 포맷으로 응답)
        content = _chat_content_for(payload.get("messages", []))
        self._send_json(
            {
                "id": "dummy-cmpl",
                "object": "chat.completion",
                "created": 0,
                "model": payload.get("model", "dummy-chat"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )

    def do_GET(self):  # noqa: N802
        if self.path.endswith("/models"):
            self._send_json({"object": "list", "data": [{"id": "dummy-model", "object": "model"}]})
            return
        self._send_json({"status": "ok"})


class DummyLLMServer(ThreadingHTTPServer):
    request_count: int = 0


@contextmanager
def run_dummy_server():
    """빈 포트에 더미 서버를 띄우고 `(base_url, server)`를 yield. base_url은 `/v1` 포함."""
    server = DummyLLMServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield f"http://{host}:{port}/v1", server
    finally:
        server.shutdown()
        thread.join(timeout=5)
