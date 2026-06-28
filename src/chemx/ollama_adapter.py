from __future__ import annotations

import argparse
import http.client
import json
import os
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

ALLOWED_TOOLS = frozenset({"exec_command", "write_stdin", "view_image"})
HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)


def _tool_name(tool: dict[str, Any]) -> str:
    if tool.get("type") != "function":
        return ""
    return str(tool.get("name") or tool.get("function", {}).get("name") or "")


def rewrite_responses_request(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rewritten = dict(payload)
    incoming_tools = payload.get("tools")
    if not isinstance(incoming_tools, list) or not incoming_tools:
        return rewritten, []

    tools = [tool for tool in incoming_tools if _tool_name(tool) in ALLOWED_TOOLS]
    rewritten["tools"] = tools
    text = rewritten.get("text")
    if isinstance(text, dict):
        text_format = text.get("format")
        if isinstance(text_format, dict) and text_format.get("type") == "json_schema":
            rewritten["text"] = {"format": {"type": "text"}}
    rewritten["temperature"] = 0.15
    rewritten["top_p"] = 0.7
    return rewritten, tools


def infer_function_name(arguments: str | dict[str, Any], tools: list[dict[str, Any]]) -> str:
    if isinstance(arguments, str):
        try:
            values = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return ""
    else:
        values = arguments
    if not isinstance(values, dict):
        return ""

    keys = set(values)
    explicit = {
        "cmd": "exec_command",
        "session_id": "write_stdin",
        "path": "view_image",
    }
    for key, name in explicit.items():
        if key in keys and any(_tool_name(tool) == name for tool in tools):
            return name

    matches: list[tuple[int, str]] = []
    for tool in tools:
        name = _tool_name(tool)
        schema = tool.get("parameters") or tool.get("function", {}).get("parameters") or {}
        properties = set(schema.get("properties", {}))
        required = set(schema.get("required", []))
        if not required.issubset(keys):
            continue
        if schema.get("additionalProperties") is False and not keys.issubset(properties):
            continue
        matches.append((len(keys & properties) + len(required), name))
    if not matches:
        return _tool_name(tools[0]) if len(tools) == 1 else ""
    matches.sort(reverse=True)
    if len(matches) > 1 and matches[0][0] == matches[1][0]:
        return ""
    return matches[0][1]


def _patch_call(
    item: dict[str, Any],
    tools: list[dict[str, Any]],
    arguments_by_id: dict[str, str],
) -> None:
    if item.get("type") != "function_call" or item.get("name"):
        return
    arguments = str(item.get("arguments") or arguments_by_id.get(str(item.get("id")), ""))
    name = infer_function_name(arguments, tools)
    if name:
        item["name"] = name


def patch_responses_json(data: bytes, tools: list[dict[str, Any]]) -> bytes:
    payload = json.loads(data)
    for item in payload.get("output", []):
        if isinstance(item, dict):
            _patch_call(item, tools, {})
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()


def patch_responses_sse(data: bytes, tools: list[dict[str, Any]]) -> bytes:
    text = data.decode("utf-8")
    parsed: list[dict[str, Any] | None] = []
    arguments_by_id: dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            parsed.append(None)
            continue
        try:
            event = json.loads(line[6:])
        except json.JSONDecodeError:
            parsed.append(None)
            continue
        parsed.append(event)
        item = event.get("item")
        if isinstance(item, dict) and item.get("type") == "function_call":
            arguments_by_id[str(item.get("id"))] = str(item.get("arguments") or "")
        item_id = event.get("item_id")
        if item_id and "arguments" in event:
            arguments_by_id[str(item_id)] = str(event["arguments"])

    output: list[str] = []
    event_index = 0
    for line in text.splitlines():
        event = parsed[event_index]
        event_index += 1
        if event is None:
            output.append(line)
            continue
        item = event.get("item")
        if isinstance(item, dict):
            _patch_call(item, tools, arguments_by_id)
        response = event.get("response")
        if isinstance(response, dict):
            for response_item in response.get("output", []):
                if isinstance(response_item, dict):
                    _patch_call(response_item, tools, arguments_by_id)
        output.append("data: " + json.dumps(event, ensure_ascii=False, separators=(",", ":")))
    suffix = "\n" if text.endswith("\n") else ""
    return ("\n".join(output) + suffix).encode()


def merge_models_response(openai_data: bytes) -> bytes:
    openai_payload = json.loads(openai_data)
    models = openai_payload.setdefault("models", [])
    if not isinstance(openai_payload.get("data"), list):
        openai_payload["data"] = [
            {"id": model_id, "object": "model"}
            for model in models
            if isinstance(model, dict)
            for model_id in [model.get("name") or model.get("model") or model.get("id")]
            if isinstance(model_id, str)
        ]
    openai_payload.setdefault("object", "list")
    return json.dumps(openai_payload, ensure_ascii=False, separators=(",", ":")).encode()


def split_address(value: str) -> tuple[str, int]:
    parsed = urlsplit(value if "://" in value else f"http://{value}")
    if not parsed.hostname or not parsed.port:
        raise ValueError(f"invalid host:port: {value}")
    return parsed.hostname, parsed.port


def upstream_ready(address: tuple[str, int]) -> bool:
    try:
        connection = http.client.HTTPConnection(*address, timeout=1)
        connection.request("HEAD", "/")
        response = connection.getresponse()
        response.read()
        connection.close()
        return response.status < 500
    except OSError:
        return False


class AdapterServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, listen: tuple[str, int], upstream: tuple[str, int]):
        super().__init__(listen, AdapterHandler)
        self.upstream = upstream


class AdapterHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: AdapterServer

    def do_HEAD(self) -> None:
        self._proxy()

    def do_GET(self) -> None:
        self._proxy()

    def do_POST(self) -> None:
        self._proxy()

    def _proxy(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(length) if length else b""
        tools: list[dict[str, Any]] = []
        is_responses = self.path.split("?", 1)[0] == "/v1/responses"
        if is_responses and body:
            try:
                payload, tools = rewrite_responses_request(json.loads(body))
                body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
            except (json.JSONDecodeError, TypeError):
                pass

        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in HOP_BY_HOP_HEADERS | {"content-length", "host"}
        }
        headers["accept-encoding"] = "identity"
        headers["content-length"] = str(len(body))
        try:
            connection = http.client.HTTPConnection(*self.server.upstream, timeout=3600)
            connection.request(self.command, self.path, body=body or None, headers=headers)
            response = connection.getresponse()
            response_body = response.read()
            response_headers = list(response.getheaders())
            connection.close()
            if self.command == "GET" and self.path.split("?", 1)[0] == "/v1/models":
                response_body = merge_models_response(response_body)
            if is_responses and response.status < 400 and response_body:
                content_type = response.getheader("content-type", "")
                if "text/event-stream" in content_type:
                    response_body = patch_responses_sse(response_body, tools)
                elif "application/json" in content_type:
                    response_body = patch_responses_json(response_body, tools)
            self.send_response(response.status, response.reason)
            for key, value in response_headers:
                if key.lower() not in HOP_BY_HOP_HEADERS | {"content-length"}:
                    self.send_header(key, value)
            self.send_header("content-length", str(len(response_body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(response_body)
        except OSError as exc:
            message = json.dumps({"error": f"Ollama upstream unavailable: {exc}"}).encode()
            self.send_response(502)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(message)))
            self.end_headers()
            self.wfile.write(message)

    def log_message(self, format: str, *args: object) -> None:
        sys.stderr.write(f"ollama-adapter: {self.address_string()} {format % args}\n")


def start_upstream(
    address: tuple[str, int], ollama_bin: str, wait_seconds: float = 30
) -> subprocess.Popen[str] | None:
    if upstream_ready(address):
        return None
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"{address[0]}:{address[1]}"
    process = subprocess.Popen([ollama_bin, "serve"], env=env, text=True)
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Ollama exited with code {process.returncode}")
        if upstream_ready(address):
            return process
        time.sleep(0.2)
    process.terminate()
    raise TimeoutError("Ollama did not become ready")


def main() -> None:
    parser = argparse.ArgumentParser(description="ChemX Codex-to-Ollama compatibility adapter")
    parser.add_argument("--listen", default="127.0.0.1:11434")
    parser.add_argument("--upstream", default="127.0.0.1:11435")
    parser.add_argument("--ollama-bin", default="ollama")
    args = parser.parse_args()
    listen = split_address(args.listen)
    upstream = split_address(args.upstream)
    process = start_upstream(upstream, args.ollama_bin)
    server = AdapterServer(listen, upstream)
    print(f"ChemX Ollama adapter listening on {listen[0]}:{listen[1]}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if process is not None:
            process.terminate()
            process.wait(timeout=10)


if __name__ == "__main__":
    main()
