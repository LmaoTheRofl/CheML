import json

from chemx.ollama_adapter import (
    infer_function_name,
    merge_models_response,
    patch_responses_json,
    patch_responses_sse,
    rewrite_responses_request,
)

TOOLS = [
    {
        "type": "function",
        "name": "exec_command",
        "parameters": {
            "type": "object",
            "properties": {"cmd": {"type": "string"}},
            "required": ["cmd"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "write_stdin",
        "parameters": {
            "type": "object",
            "properties": {"session_id": {"type": "integer"}, "chars": {"type": "string"}},
            "required": ["session_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "view_image",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
]


def test_rewrite_filters_tools_and_disables_schema_grammar() -> None:
    payload = {
        "tools": [*TOOLS, {"type": "function", "name": "update_plan"}],
        "text": {"format": {"type": "json_schema", "schema": {"type": "object"}}},
    }
    rewritten, tools = rewrite_responses_request(payload)
    assert [_tool["name"] for _tool in tools] == [
        "exec_command",
        "write_stdin",
        "view_image",
    ]
    assert rewritten["text"] == {"format": {"type": "text"}}
    assert rewritten["temperature"] == 0.15
    assert rewritten["top_p"] == 0.7


def test_rewrite_without_tools_preserves_schema() -> None:
    payload = {"text": {"format": {"type": "json_schema"}}}
    rewritten, tools = rewrite_responses_request(payload)
    assert rewritten == payload
    assert tools == []


def test_infer_function_name_from_distinct_argument_shapes() -> None:
    assert infer_function_name('{"cmd":"pwd"}', TOOLS) == "exec_command"
    assert infer_function_name('{"session_id":12,"chars":""}', TOOLS) == "write_stdin"
    assert infer_function_name('{"path":"page.png"}', TOOLS) == "view_image"


def test_infer_function_name_rejects_unknown_arguments() -> None:
    assert infer_function_name('{"unknown":true}', TOOLS) == ""
    assert infer_function_name("not-json", TOOLS) == ""


def test_patch_streaming_response_restores_blank_call_names() -> None:
    def data(payload: dict[str, object]) -> str:
        return "data: " + json.dumps(payload, separators=(",", ":"))

    events = [
        "event: response.output_item.added",
        data(
            {
                "type": "response.output_item.added",
                "item": {
                    "id": "fc_1",
                    "type": "function_call",
                    "name": "",
                    "arguments": "",
                },
            }
        ),
        "",
        "event: response.function_call_arguments.done",
        data(
            {
                "type": "response.function_call_arguments.done",
                "item_id": "fc_1",
                "arguments": '{"cmd":"pwd"}',
            }
        ),
        "",
        "event: response.output_item.done",
        data(
            {
                "type": "response.output_item.done",
                "item": {
                    "id": "fc_1",
                    "type": "function_call",
                    "name": "",
                    "arguments": '{"cmd":"pwd"}',
                },
            }
        ),
        "",
    ]
    patched = patch_responses_sse("\n".join(events).encode(), TOOLS).decode()
    assert patched.count('"name":"exec_command"') == 2


def test_patch_json_preserves_existing_function_name() -> None:
    body = json.dumps(
        {
            "output": [
                {
                    "type": "function_call",
                    "name": "view_image",
                    "arguments": '{"path":"page.png"}',
                }
            ]
        }
    ).encode()
    patched = json.loads(patch_responses_json(body, TOOLS))
    assert patched["output"][0]["name"] == "view_image"


def test_merge_models_response_supports_both_codex_decoders() -> None:
    merged = json.loads(merge_models_response(b'{"object":"list","data":[{"id":"gemma"}]}'))
    assert merged["data"] == [{"id": "gemma"}]
    assert merged["models"] == []

    native = json.loads(merge_models_response(b'{"models":[{"name":"gemma:latest"}]}'))
    assert native["object"] == "list"
    assert native["data"] == [{"id": "gemma:latest", "object": "model"}]
