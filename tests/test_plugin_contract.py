# -*- coding: utf-8 -*-
"""Anthropic plugin-contract tests (LLM Provider Plugin plan §6.2 / A.2).

Single-turn tool branches, provider_state round-trip (§5.3/§5.5), pooled
client injection (§5.2.1/D7) and usage extraction (Q6) — verified against
fake response objects with no network and no model calls.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest
from llm_provider_base import LlmProviderHandler

from llm_provider_anthropic import AnthropicProvider


def _make_agent(**overrides):
    agent = {
        "instructions": "You are a test agent.",
        "configuration": {"model": "claude-test-1", "max_tokens": 100},
        "llm": {"llm_name": "anthropic"},
        "tool_call_role": "tool",
    }
    agent.update(overrides)
    return agent


def _ctx(**overrides):
    defaults = {
        "http_client": httpx.Client(),
        "credentials": {"api_key": "test-key"},
        "endpoint": "https://api.anthropic.com",
        "model_identifier": "claude-test-1",
        "options": {},
        "trace_id": "tid-1",
        "emit": None,
        "is_cancelled": None,
        "pool_name": "httpx_llm_anthropic",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _handler(ctx=None, agent=None):
    return AnthropicProvider(
        __import__("logging").getLogger("kit"),
        agent or _make_agent(),
        ctx=ctx if ctx is not None else _ctx(),
    )


def _blocks_response(stop_reason, blocks, usage=None, response_id="resp_1"):
    container = None
    if stop_reason == "tool_use":
        container = SimpleNamespace(id="container_1")
    return SimpleNamespace(
        id=response_id,
        role="assistant",
        stop_reason=stop_reason,
        content=blocks,
        usage=usage or SimpleNamespace(input_tokens=3, output_tokens=4),
        container=container,
    )


def _thinking(text, signature=None):
    return SimpleNamespace(type="thinking", thinking=text, signature=signature)


def _text(t):
    return SimpleNamespace(type="text", text=t)


def _tool_use(id, name, tool_input):
    return SimpleNamespace(type="tool_use", id=id, name=name, input=tool_input)


# ---------------------------------------------------------------- metadata


def test_handler_is_plugin_base_subclass():
    assert issubclass(AnthropicProvider, LlmProviderHandler)


def test_plugin_metadata():
    assert AnthropicProvider.api_protocol == "anthropic"
    assert AnthropicProvider.contract_version == "1"
    assert AnthropicProvider.fallback_protocol is None
    assert AnthropicProvider.default_pool_name == "httpx_llm_anthropic"
    caps = AnthropicProvider.capabilities
    assert caps["tools"] is True
    assert caps["reasoning"] is True
    assert caps["streaming"] is True
    md = AnthropicProvider.plugin_metadata()
    assert md["api_protocol"] == "anthropic"
    assert md["display_name"] == "Anthropic"
    assert isinstance(md["config_schema"], dict)
    # Q8/CREDS: api_key is a credential now — absent from config_schema
    assert "api_key" not in md["config_schema"].get("properties", {})
    assert "api_key" not in md["config_schema"].get("required", [])


# ---------------------------------------------------------------- __init__


def test_init_requires_pooled_client():
    with pytest.raises(Exception, match="ctx.http_client"):
        _handler(ctx=SimpleNamespace(http_client=None, credentials={}))


def test_init_builds_sdk_on_injected_client():
    handler = _handler()
    assert type(handler.client).__name__ == "Anthropic"


def test_init_uses_ctx_credentials_not_configuration():
    agent = _make_agent(configuration={"model": "claude-test-1", "api_key": "leaky"})
    handler = _handler(agent=agent)
    # api_key never read from configuration; model_setting excludes it
    assert "api_key" not in handler.model_setting
    assert "base_url" not in handler.model_setting


def test_init_base_url_from_configuration():
    handler = _handler(
        agent=_make_agent(
            configuration={
                "model": "claude-test-1",
                "base_url": "https://custom.endpoint/v1",
            }
        )
    )
    assert str(handler.client.base_url).rstrip("/").endswith("custom.endpoint/v1")


def test_init_converts_neutral_tools():
    handler = _handler(
        agent=_make_agent(
            configuration={
                "model": "claude-test-1",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "find_order",
                            "description": "find",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            }
        )
    )
    tools = handler.model_setting["tools"]
    assert tools == [
        {
            "name": "find_order",
            "description": "find",
            "input_schema": {"type": "object", "properties": {}},
        }
    ]


# ------------------------------------------------- neutral → native (§5.5)


def test_to_native_drops_system_role():
    handler = _handler()
    native = handler.to_native_messages(
        [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
        ]
    )
    assert all(m["role"] != "system" for m in native)
    assert native == [{"role": "user", "content": "hi"}]


def test_to_native_tool_shapes():
    handler = _handler()
    native = handler.to_native_messages(
        [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "find_order", "arguments": '{"q": 1}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        ]
    )
    assert native[1] == {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "call_1",
                "name": "find_order",
                "input": {"q": 1},
            }
        ],
    }
    assert native[2]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "call_1",
        "content": "result",
    }


def test_to_native_provider_state_replay():
    handler = _handler()
    state = {
        "api_protocol": "anthropic",
        "model": "claude-test-1",
        "container_id": "container_7",
        "blocks": [
            {"type": "thinking", "thinking": "t", "signature": "sig1"},
            {
                "type": "tool_use",
                "id": "tu_1",
                "name": "find_order",
                "input": '{"q": 1}',
            },
        ],
    }
    native = handler.to_native_messages(
        [
            {"role": "assistant", "content": "", "provider_state": state},
            {"role": "user", "content": "continue"},
        ]
    )
    assert native[0]["content"] == [
        {"type": "thinking", "thinking": "t", "signature": "sig1"},
        {"type": "tool_use", "id": "tu_1", "name": "find_order", "input": {"q": 1}},
    ]
    # skills container continuity (§5.3 / A.2)
    assert handler._container_id == "container_7"


def test_to_native_stale_state_dropped_message_kept():
    handler = _handler()
    stale_state = {
        "api_protocol": "anthropic",
        "model": "some-other-model",
        "blocks": [{"type": "thinking", "thinking": "old", "signature": "old"}],
    }
    native = handler.to_native_messages(
        [{"role": "assistant", "content": "prior text", "provider_state": stale_state}]
    )
    # §5.5: stale state dropped, the message itself is kept (plain text)
    assert native[0]["content"] == [{"type": "text", "text": "prior text"}]


# --------------------------------------------- single-turn handle_response

def test_handle_response_tool_branch_returns_calls_and_state():
    handler = _handler()
    response = _blocks_response(
        "tool_use",
        [
            _thinking("deep thought", "sig123"),
            _text("let me check."),
            _tool_use("tu_1", "find_order", {"q": "x"}),
        ],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )
    handler.handle_response(response, [])

    calls = handler.final_output["tool_calls"]
    assert len(calls) == 1
    assert calls[0]["id"] == "tu_1"
    assert calls[0]["type"] == "function"
    assert calls[0]["function"]["name"] == "find_order"
    assert json.loads(calls[0]["function"]["arguments"]) == {"q": "x"}

    state = handler.final_output["provider_state"]
    assert state["api_protocol"] == "anthropic"
    assert state["model"] == "claude-test-1"
    assert state["container_id"] == "container_1"
    assert [b["type"] for b in state["blocks"]] == ["thinking", "text", "tool_use"]
    assert state["blocks"][0]["signature"] == "sig123"

    assert handler.final_output["finish_reason"] == "tool_calls"
    assert handler.final_output["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }
    assert handler.final_output["content"] == "let me check."


def test_handle_response_end_turn_usage_and_signed_state():
    handler = _handler()
    response = _blocks_response(
        "end_turn", [_thinking("musing", "sig9"), _text("answer")]
    )
    handler.handle_response(response, [])
    assert handler.final_output["finish_reason"] == "end_turn"
    assert handler.final_output["usage"]["total_tokens"] == 7
    assert handler.final_output["provider_state"]["blocks"] == [
        {"type": "thinking", "thinking": "musing", "signature": "sig9"}
    ]
    assert handler.final_output["content"] == "answer"


def test_handle_response_plain_answer_has_no_state():
    handler = _handler()
    response = _blocks_response(
        "end_turn", [_text("just answer")], usage=SimpleNamespace(
            input_tokens=1, output_tokens=1
        )
    )
    handler.handle_response(response, [])
    assert handler.final_output["provider_state"] is None
    assert handler.final_output["usage"]["total_tokens"] == 2


# ----------------------------------------------- single-turn handle_stream

def _chunk(type, **attrs):
    return SimpleNamespace(type=type, **attrs)


def test_handle_stream_tool_branch_single_turn():
    emitted = []
    finished = __import__("threading").Event()
    ctx = _ctx(emit=lambda c: emitted.append(dict(c)))
    handler = _handler(ctx=ctx)
    handler.run = {"run_uuid": "run-1", "thread_uuid": "thread-1"}

    chunks = [
        _chunk(
            "message_start",
            message=SimpleNamespace(
                id="msg_1",
                usage=SimpleNamespace(
                    input_tokens=10, cache_read_input_tokens=4
                ),
                container=SimpleNamespace(id="container_9"),
            ),
        ),
        _chunk(
            "content_block_start",
            content_block=SimpleNamespace(type="tool_use", id="tu_2", name="lookup"),
        ),
        _chunk(
            "content_block_delta",
            delta=SimpleNamespace(type="input_json_delta", partial_json='{"q": 2}'),
        ),
        _chunk("content_block_stop"),
        _chunk(
            "message_delta",
            delta=SimpleNamespace(
                stop_reason="tool_use",
                usage=SimpleNamespace(output_tokens=6),
            ),
        ),
        _chunk("message_stop"),
    ]
    handler.handle_stream(iter(chunks), input_messages=[], stream_event=finished)

    assert finished.is_set()
    calls = handler.final_output["tool_calls"]
    assert calls[0]["function"]["name"] == "lookup"
    assert json.loads(calls[0]["function"]["arguments"]) == {"q": 2}
    state = handler.final_output["provider_state"]
    assert state["api_protocol"] == "anthropic"
    assert state["container_id"] == "container_9"
    assert [b["type"] for b in state["blocks"]] == ["tool_use"]
    assert handler.final_output["finish_reason"] == "tool_calls"
    assert handler.final_output["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 6,
        "total_tokens": 16,
        "cache_read_tokens": 4,
    }
    # single-turn end marker emitted for the tool-call turn (§7.2 shape)
    end_markers = [c for c in emitted if c.get("is_message_end")]
    assert end_markers, f"no message-end marker in {emitted!r}"
    for chunk in emitted:
        for key in ("message_group_id", "event_type", "index", "data_format",
                    "chunk_delta", "is_message_end"):
            assert key in chunk, f"payload missing {key!r}: {chunk!r}"


def test_handle_stream_end_turn_usage_and_signed_state():
    ctx = _ctx(emit=lambda c: None)
    handler = _handler(ctx=ctx)
    handler.run = {"run_uuid": "run-1", "thread_uuid": "thread-1"}

    chunks = [
        _chunk(
            "message_start",
            message=SimpleNamespace(id="msg_2", usage=SimpleNamespace(input_tokens=2)),
        ),
        _chunk(
            "content_block_start",
            content_block=SimpleNamespace(
                type="thinking",
                thinking="m",
                signature="s",
                index=0,
            ),
        ),
        # reasoning delta path
        _chunk(
            "content_block_delta",
            delta=SimpleNamespace(
                type="thinking_delta", thinking="ore", signature=None
            ),
        ),
        _chunk(
            "content_block_delta",
            delta=SimpleNamespace(type="signature_delta", signature="sig_full"),
        ),
        # thinking close arm matches the block's index (0)
        _chunk("content_block_stop", index=0),
        _chunk(
            "content_block_start",
            content_block=SimpleNamespace(type="text", index=1),
        ),
        _chunk(
            "content_block_delta",
            delta=SimpleNamespace(type="text_delta", text="final"),
        ),
        _chunk("content_block_stop"),
        _chunk(
            "message_delta",
            delta=SimpleNamespace(
                stop_reason="end_turn", usage=SimpleNamespace(output_tokens=3)
            ),
        ),
        _chunk("message_stop"),
    ]
    handler.handle_stream(iter(chunks), input_messages=[], stream_event=None)

    assert handler.final_output["finish_reason"] == "end_turn"
    assert handler.final_output["usage"]["total_tokens"] == 5
    assert handler.final_output["content"] == "final"
    state = handler.final_output["provider_state"]
    assert state is not None
    assert state["blocks"][0]["signature"] == "sig_full"


# ------------------------------------------------------------------- usage


def test_map_usage_dict_and_object():
    handler = _handler()
    assert handler._map_usage(None) == {}
    mapped = handler._map_usage(
        {"input_tokens": "5", "output_tokens": None, "cache_read_input_tokens": 2}
    )
    assert mapped == {
        "prompt_tokens": 5,
        "completion_tokens": 0,
        "total_tokens": 5,
        "cache_read_tokens": 2,
    }
    mapped = handler._map_usage(SimpleNamespace(input_tokens=1, output_tokens=2))
    assert mapped["total_tokens"] == 3


def test_insert_file_routes_through_pooled_client():
    """insert_file downloads via ctx.http_client (plan D7 / HTTP)."""
    from unittest import mock

    captured = {}

    def _handler_transport(request: "httpx.Request") -> "httpx.Response":
        captured["url"] = str(request.url)
        return httpx.Response(200, content=b"raw-bytes")

    # Real pooled httpx.Client with a mock transport (no network); the SDK
    # call itself is stubbed.
    pooled = httpx.Client(transport=httpx.MockTransport(_handler_transport))
    handler = _handler(ctx=_ctx(http_client=pooled))
    with mock.patch.object(
        handler.client.beta.files,
        "upload",
        return_value=SimpleNamespace(id="file_1"),
    ):
        result = handler.insert_file(
            file_uri="https://example.com/f.pdf",
            filename="f.pdf",
            mime_type="application/pdf",
        )
    assert captured["url"] == "https://example.com/f.pdf"
    assert result.id == "file_1"
