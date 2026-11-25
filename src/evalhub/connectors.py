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
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from pydantic import BaseModel

from evalhub.models import HTTPAgentConfig, PythonAgentConfig, RegisteredAgent, ToolCallRecord


class AgentResponse(BaseModel):
    actual_output: Any = None
    tool_calls: Optional[list[ToolCallRecord]] = None
    retrieval_context: Optional[list[str]] = None
    original_input: Any = None
    source_run_id: Optional[str] = None
    source_status: Optional[str] = None
    ready: bool = True
    explanation: Any = None
    evidence: Any = None
    raw_response: Any = None


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
    extra = {}
    if isinstance(config, HTTPAgentConfig) and config.method == "GET":
        if not isinstance(value, dict):
            raise AgentExecutionError("Results API must return one run object, not a list or scalar.")
        extra["raw_response"] = value
        if config.status_path:
            status = _extract(value, config.status_path)
            if not isinstance(status, str) or not status.strip():
                raise AgentExecutionError("Source run status must be a non-empty string.")
            extra["source_status"] = status
            if status.strip().casefold() not in {
                item.strip().casefold() for item in config.completed_statuses
            }:
                return AgentResponse(**extra, ready=False)
        for field, path in (
            ("original_input", config.input_path), ("source_run_id", config.run_id_path),
            ("explanation", config.explanation_path), ("evidence", config.evidence_path),
        ):
            if path:
                extra[field] = _extract(value, path)
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
            **extra,
        )
        json.dumps(result.model_dump(mode="python"), allow_nan=False)
        return result
    except (TypeError, ValueError) as exc:
        raise AgentExecutionError("Agent returned a non-JSON output or invalid trace/context.") from exc


def build_get_url(config: HTTPAgentConfig, input_value: Any) -> str:
    """Encode lookup values without allowing them to change the URL structure."""
    if config.query_params or config.path_params:
        if not isinstance(input_value, dict):
            raise ValueError("Mapped GET requests require a lookup object.")
        lookup = input_value
        query_mapping = config.query_params
    else:
        lookup = {config.input_field: input_value}
        query_mapping = {config.input_field: config.input_field}
    encoded_path = {}
    query = []
    for mapping, destination in ((config.path_params, encoded_path), (query_mapping, query)):
        for parameter, field in mapping.items():
            if field not in lookup:
                raise ValueError(f"GET lookup is missing field '{field}'.")
            value = lookup[field]
            if not isinstance(value, (str, int, float, bool)) or value == "":
                raise ValueError(f"GET lookup field '{field}' must be a non-empty scalar.")
            json.dumps(value, allow_nan=False)
            text = str(value).lower() if isinstance(value, bool) else str(value)
            if isinstance(destination, dict):
                if text in (".", ".."):
                    raise ValueError("Path lookup values must not be dot segments.")
                destination[parameter] = quote(text, safe="")
            else:
                destination.append((parameter, text))
    url = urlsplit(config.url.format(**encoded_path))
    params = parse_qsl(url.query, keep_blank_values=True) + query
    return urlunsplit((url.scheme, url.netloc, url.path, urlencode(params), ""))


def retrieve_registered_run(registration: RegisteredAgent, lookup: dict[str, Any]) -> AgentResponse:
    if not registration.enabled:
        raise ValueError(f"Agent '{registration.agent_id}' is disabled.")
    registration.validate_lookup(lookup)
    response = create_agent(registration.connector)(lookup)
    if response.source_run_id is not None and "run_id" in lookup:
        if response.source_run_id != str(lookup["run_id"]):
            raise AgentExecutionError("Results API returned a different source run ID.")
    return response


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
        body = None
        request_url = config.url
        headers = {"Accept": "application/json"}
        if config.method == "GET":
            request_url = build_get_url(config, input_value)
        else:
            body = json.dumps({config.input_field: input_value}, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if token is not None:
            headers[config.auth_header] = config.auth_prefix + token
        request = Request(request_url, data=body, headers=headers, method=config.method)
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
            except (TypeError, ValueError) as exc:
                raise AgentExecutionError("Agent response contains invalid mapped fields.") from exc
        raise AssertionError("HTTP retry loop exited unexpectedly.")

    return invoke_http
