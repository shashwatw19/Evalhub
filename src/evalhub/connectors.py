"""Invoke configured agent endpoints and normalize their responses."""

from __future__ import annotations

import importlib
import inspect
import http.client
import json
import os
import socket
import time
from typing import Any, Callable, Optional, Union
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from pydantic import BaseModel

from evalhub.models import HTTPAgentConfig, PythonAgentConfig, ToolCallRecord


class AgentResponse(BaseModel):
    actual_output: Any
    tool_calls: Optional[list[ToolCallRecord]] = None
    retrieval_context: Optional[list[str]] = None


class AgentExecutionError(RuntimeError):
    """A safe-to-display agent invocation failure."""


class AgentTimeoutError(AgentExecutionError):
    """An agent endpoint exceeded its configured timeout."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, _req, _fp, _code, _msg, _headers, _newurl):
        return None


def _extract(value: Any, path: Optional[str]) -> Any:
    if path is None or path == "":
        return value
    current = value
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            raise AgentExecutionError(f"Agent response is missing field path '{path}'.")
        current = current[key]
    return current


def _response(value: Any, config: Union[HTTPAgentConfig, PythonAgentConfig]) -> AgentResponse:
    output = _extract(value, config.response_path)
    if output is None:
        raise AgentExecutionError("Agent returned a null output.")
    try:
        result = AgentResponse(
            actual_output=output,
            tool_calls=_extract(value, config.tool_calls_path) if config.tool_calls_path else None,
            retrieval_context=(
                _extract(value, config.retrieval_context_path)
                if config.retrieval_context_path
                else None
            ),
        )
        json.dumps(result.model_dump(mode="python"), allow_nan=False)
        return result
    except (TypeError, ValueError) as exc:
        raise AgentExecutionError("Agent returned a non-JSON output or invalid trace/context.") from exc


def _load_callable(path: str) -> Callable[[Any], Any]:
    module_name, separator, attribute = path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("Python callable must use the format 'module:function'.")
    try:
        value = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Unable to load Python callable '{path}'.") from exc
    if not callable(value):
        raise ValueError(f"Python agent '{path}' is not callable.")
    if inspect.iscoroutinefunction(value):
        raise ValueError("Python agent callables must be synchronous.")
    return value


def create_agent(config: Union[HTTPAgentConfig, PythonAgentConfig]) -> Callable[[Any], AgentResponse]:
    """Create an agent callable. Python modules must come from trusted code."""
    if isinstance(config, PythonAgentConfig):
        agent = _load_callable(config.callable)

        def invoke_python(input_value: Any) -> AgentResponse:
            try:
                value = agent(input_value)
            except Exception as exc:
                raise AgentExecutionError(
                    f"Python agent failed ({type(exc).__name__})."
                ) from exc
            return _response(value, config)

        return invoke_python

    url = urlsplit(config.url)
    if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password:
        raise ValueError("Agent URL must be an HTTP(S) URL without embedded credentials.")
    token = None
    if config.auth_env:
        token = os.environ.get(config.auth_env)
        if not token:
            raise ValueError(f"Agent authentication environment variable '{config.auth_env}' is unset.")
    opener = build_opener(_NoRedirect())

    def invoke_http(input_value: Any) -> AgentResponse:
        body = json.dumps({config.input_field: input_value}, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if token is not None:
            headers[config.auth_header] = config.auth_prefix + token
        request = Request(config.url, data=body, headers=headers, method="POST")
        for attempt in range(config.retries + 1):
            try:
                with opener.open(request, timeout=config.timeout_seconds) as response:
                    value = json.load(response)
                return _response(value, config)
            except HTTPError as exc:
                if attempt < config.retries and (exc.code == 429 or 500 <= exc.code <= 599):
                    time.sleep(min(2 ** attempt, 4))
                    continue
                raise AgentExecutionError(f"Agent HTTP request failed (status {exc.code}).") from exc
            except (URLError, OSError, http.client.HTTPException) as exc:
                if attempt < config.retries:
                    time.sleep(min(2 ** attempt, 4))
                    continue
                reason = exc.reason if isinstance(exc, URLError) else exc
                if isinstance(reason, (TimeoutError, socket.timeout)):
                    raise AgentTimeoutError("Agent HTTP request timed out.") from exc
                raise AgentExecutionError("Agent HTTP request failed or timed out.") from exc
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise AgentExecutionError("Agent HTTP response was not valid JSON.") from exc
        raise AssertionError("HTTP retry loop exited unexpectedly.")

    return invoke_http
