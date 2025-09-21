import json
import os
import uuid
from urllib.parse import urlparse

import azure.functions as func

from lambda_handler import lambda_handler


class AzureLambdaContext:
    """Adapta o contexto do Azure Functions para o handler do Lambda."""

    def __init__(self, context: func.Context, timeout_ms: int = 300000):
        self.function_name = getattr(context, "function_name", os.getenv("WEBSITE_SITE_NAME", "agent-core-azure"))
        self.aws_request_id = getattr(context, "invocation_id", str(uuid.uuid4()))
        self._timeout_ms = timeout_ms

    def get_remaining_time_in_millis(self) -> int:
        return self._timeout_ms


def _normalize_path(raw_path: str) -> str:
    if not raw_path:
        return "/"

    path = raw_path
    if path.startswith("/api"):
        path = path[4:] or "/"
    if not path.startswith("/"):
        path = f"/{path}"
    return path or "/"


def _build_lambda_event(req: func.HttpRequest) -> dict:
    parsed = urlparse(req.url)
    body_bytes = req.get_body()
    body_text = body_bytes.decode("utf-8") if body_bytes else ""

    return {
        "body": body_text,
        "path": _normalize_path(parsed.path),
        "httpMethod": req.method,
        "headers": dict(req.headers) if req.headers else {},
        "queryStringParameters": dict(req.params) if req.params else {},
    }


def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    """HTTP trigger que delega o processamento ao Lambda handler existente."""
    lambda_event = _build_lambda_event(req)
    adapted_context = AzureLambdaContext(context)

    lambda_response = lambda_handler(lambda_event, adapted_context)

    status_code = lambda_response.get("statusCode", 500)
    headers = lambda_response.get("headers") or {}
    body = lambda_response.get("body", "{}")

    if not isinstance(body, (str, bytes)):
        body = json.dumps(body, ensure_ascii=False)

    return func.HttpResponse(body=body, status_code=status_code, headers=headers)
