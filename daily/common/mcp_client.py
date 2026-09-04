"""Minimal MCP streamable-HTTP client (stdlib only).

The daily runners must not depend on the `mcp` SDK just to read a couple of
numbers from the central `daily_results` server, so this implements the small
slice of the protocol that a one-shot tool call needs: initialize, the
`notifications/initialized` notification, then `tools/call`.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

log = logging.getLogger(__name__)

PROTOCOL_VERSION = "2025-06-18"
_CLIENT_INFO = {"name": "daily-report", "version": "1"}


class McpError(RuntimeError):
    """Raised when the server is unreachable or returns a JSON-RPC error."""


class McpHttpClient:
    """One-shot client for a streamable-HTTP MCP endpoint."""

    def __init__(self, url: str, *, timeout: float = 15.0) -> None:
        self.url = url
        self.timeout = timeout
        self._session_id: str | None = None
        self._next_id = 0

    def __enter__(self) -> "McpHttpClient":
        self._initialize()
        return self

    def __exit__(self, *exc_info) -> None:
        return None

    def call_tool(self, name: str, arguments: dict) -> str:
        """Call *name* and return the concatenated text content of the result."""
        result = self._request("tools/call", {"name": name, "arguments": arguments})
        if result.get("isError"):
            raise McpError(f"tool {name} reported an error: {result}")
        texts = [
            block.get("text", "")
            for block in result.get("content", [])
            if block.get("type") == "text"
        ]
        return "".join(texts)

    # -- protocol -----------------------------------------------------------

    def _initialize(self) -> None:
        self._request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": _CLIENT_INFO,
            },
        )
        self._notify("notifications/initialized")

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": PROTOCOL_VERSION,
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return headers

    def _post(self, payload: dict) -> tuple[str, dict]:
        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                session_id = resp.headers.get("mcp-session-id")
                if session_id:
                    self._session_id = session_id
                return resp.read().decode("utf-8", "replace"), dict(resp.headers)
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise McpError(f"{self.url}: {exc}") from exc

    def _notify(self, method: str) -> None:
        self._post({"jsonrpc": "2.0", "method": method})

    def _request(self, method: str, params: dict) -> dict:
        self._next_id += 1
        body, _ = self._post(
            {"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params}
        )
        message = _parse_response(body)
        if "error" in message:
            raise McpError(f"{method} failed: {message['error']}")
        return message.get("result", {})


def _parse_response(body: str) -> dict:
    """Decode a JSON-RPC response sent either as JSON or as an SSE stream."""
    text = body.strip()
    if not text:
        raise McpError("empty response from server")
    if text.startswith("{"):
        return json.loads(text)
    for line in text.splitlines():
        if line.startswith("data:"):
            chunk = line[5:].strip()
            if chunk:
                message = json.loads(chunk)
                if "id" in message:
                    return message
    raise McpError(f"no JSON-RPC message in response: {text[:200]}")


def run_sql(url: str, sql: str, *, timeout: float = 15.0) -> list[dict]:
    """Run a read-only query through the `daily_results_run_sql` tool."""
    with McpHttpClient(url, timeout=timeout) as client:
        payload = client.call_tool("daily_results_run_sql", {"sql": sql})
    try:
        rows = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise McpError(f"unparsable tool output: {payload[:200]}") from exc
    if isinstance(rows, dict) and "error" in rows:
        raise McpError(f"query rejected: {rows['error']}")
    if not isinstance(rows, list):
        raise McpError(f"unexpected tool output: {payload[:200]}")
    return rows
