# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import http.client
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence

import pytest
import torch

from rl_engine.executors.openai_server import (
    MAX_REQUEST_BODY_BYTES,
    GeneratedText,
    OpenAICompletionService,
    OpenAIServingError,
    RolloutCompletionBackend,
    StreamChunk,
    TransformersCompletionBackend,
    create_server,
    fallback_chat_prompt,
)


class RecordingBackend:
    model_id = "recording-model"

    def __init__(self):
        self.calls = []

    def format_chat_prompt(self, messages: Sequence[Mapping[str, object]]) -> str:
        return fallback_chat_prompt(messages)

    def generate(
        self,
        prompts: Sequence[str],
        *,
        n: int,
        max_tokens: int,
        temperature: float | None,
        top_p: float | None,
        stop: str | Sequence[str] | None,
        extra: Mapping[str, object] | None = None,
    ) -> list[list[GeneratedText]]:
        self.calls.append(
            {
                "prompts": list(prompts),
                "n": n,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stop": stop,
                "extra": dict(extra or {}),
            }
        )
        return [
            [
                GeneratedText(
                    text=f"{prompt} completion {index}",
                    finish_reason="length",
                    prompt_tokens=len(prompt.split()),
                    completion_tokens=2,
                )
                for index in range(n)
            ]
            for prompt in prompts
        ]

    def stream_generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float | None,
        top_p: float | None,
        stop: str | Sequence[str] | None,
        extra: Mapping[str, object] | None = None,
    ):
        del max_tokens, temperature, top_p, stop, extra
        yield f"{prompt} "
        yield "stream"
        yield StreamChunk(finish_reason="length")


class FakeRolloutCandidate:
    def __init__(self):
        self.text = "rollout completion"
        self.finish_reason = "stop"
        self.prompt_token_ids = (1, 2)
        self.token_ids = (3, 4, 5)


class RecordingRolloutExecutor:
    def __init__(self):
        self.calls = []
        self.config = {"model": "rollout-model"}

    def generate_candidates(self, prompts, *, num_generations, sampling_params=None):
        self.calls.append(
            {
                "prompts": list(prompts),
                "num_generations": num_generations,
                "sampling_params": dict(sampling_params or {}),
            }
        )
        return {"normalized_outputs": [[FakeRolloutCandidate()]]}


class FailingGenerateModel:
    def generate(self, **kwargs):
        del kwargs
        raise RuntimeError("stream failure")


class CancellableStreamingModel:
    def __init__(self):
        self.cancel_seen = False
        self.finished = threading.Event()

    def generate(self, **kwargs):
        streamer = kwargs["streamer"]
        stopping_criteria = kwargs["stopping_criteria"]
        streamer.on_finalized_text("alpha STOP omega", stream_end=False)
        try:
            deadline = time.time() + 2
            while time.time() < deadline:
                should_stop = stopping_criteria(torch.tensor([[1]]), None).item()
                if should_stop:
                    self.cancel_seen = True
                    break
                time.sleep(0.01)
            return torch.tensor([[1, 2, 3]])
        finally:
            streamer.on_finalized_text("", stream_end=True)
            self.finished.set()


class StreamingErrorTokenizer:
    pad_token_id = 0
    eos_token_id = None

    def __call__(self, text, *, return_tensors, truncation):
        del text, return_tensors, truncation
        return {"input_ids": torch.tensor([[1]])}

    def decode(self, token_ids, **kwargs):
        del token_ids, kwargs
        return ""


def test_completion_response_shape_with_multiple_prompts_and_choices():
    backend = RecordingBackend()
    service = OpenAICompletionService(backend)

    payload = service.create_completion(
        {
            "model": "client-model",
            "prompt": ["alpha", "beta"],
            "n": 2,
            "max_tokens": 5,
            "temperature": 0,
            "top_p": 1,
        }
    )

    assert payload["object"] == "text_completion"
    assert payload["model"] == "client-model"
    assert [choice["index"] for choice in payload["choices"]] == [0, 1, 2, 3]
    assert payload["choices"][0]["text"] == "alpha completion 0"
    assert payload["usage"] == {"prompt_tokens": 4, "completion_tokens": 8, "total_tokens": 12}
    assert backend.calls[0]["prompts"] == ["alpha", "beta"]
    assert backend.calls[0]["temperature"] == 0.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("echo", True),
        ("suffix", "tail"),
        ("logprobs", 1),
        ("best_of", 2),
        ("stream_options", {"include_usage": True}),
    ],
)
def test_completion_rejects_unsupported_fields_before_generation(field, value):
    backend = RecordingBackend()
    service = OpenAICompletionService(backend)

    with pytest.raises(OpenAIServingError) as exc_info:
        service.create_completion(
            {
                "prompt": "hello",
                "max_tokens": 2,
                field: value,
            }
        )

    assert exc_info.value.param == field
    assert backend.calls == []


def test_stream_completion_rejects_unsupported_fields_before_first_event():
    backend = RecordingBackend()
    service = OpenAICompletionService(backend)

    events = service.stream_completion(
        {
            "prompt": "hello",
            "max_tokens": 2,
            "stream_options": {"include_usage": True},
        }
    )
    with pytest.raises(OpenAIServingError) as exc_info:
        next(events)

    assert exc_info.value.param == "stream_options"
    assert backend.calls == []


def test_generation_params_rejects_invalid_stop_before_generation():
    backend = RecordingBackend()
    service = OpenAICompletionService(backend)

    with pytest.raises(OpenAIServingError, match="stop"):
        service.create_completion({"prompt": "hello", "max_tokens": 2, "stop": 42})

    assert backend.calls == []


@pytest.mark.parametrize("field", ["n", "max_tokens"])
def test_generation_params_rejects_fractional_integer_fields(field):
    backend = RecordingBackend()
    service = OpenAICompletionService(backend)

    payload = {"prompt": "hello", "max_tokens": 2, field: 1.9}
    with pytest.raises(OpenAIServingError) as exc_info:
        service.create_completion(payload)

    assert exc_info.value.param == field
    assert backend.calls == []


def test_chat_completion_uses_assistant_message_shape():
    service = OpenAICompletionService(RecordingBackend())

    payload = service.create_chat_completion(
        {
            "messages": [
                {"role": "system", "content": "be direct"},
                {"role": "user", "content": "hello"},
            ],
            "n": 1,
            "max_tokens": 3,
        }
    )

    assert payload["object"] == "chat.completion"
    choice = payload["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert "system: be direct" in choice["message"]["content"]
    assert "user: hello" in choice["message"]["content"]


def test_chat_completion_rejects_unsupported_content_parts():
    service = OpenAICompletionService(RecordingBackend())

    with pytest.raises(OpenAIServingError, match="image_url"):
        service.create_chat_completion(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.invalid/image.png"},
                            }
                        ],
                    }
                ],
            }
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tools", [{"type": "function", "function": {"name": "lookup"}}]),
        ("tool_choice", "auto"),
        ("functions", [{"name": "lookup"}]),
        ("function_call", {"name": "lookup"}),
        ("response_format", {"type": "json_object"}),
        ("modalities", ["text", "audio"]),
        ("audio", {"voice": "alloy", "format": "mp3"}),
        ("logprobs", True),
        ("top_logprobs", 1),
    ],
)
def test_chat_completion_rejects_unsupported_fields(field, value):
    backend = RecordingBackend()
    service = OpenAICompletionService(backend)

    with pytest.raises(OpenAIServingError) as exc_info:
        service.create_chat_completion(
            {
                "messages": [{"role": "user", "content": "hello"}],
                field: value,
            }
        )

    assert exc_info.value.param == field
    assert backend.calls == []


def test_completion_validates_model_before_generation():
    backend = RecordingBackend()
    service = OpenAICompletionService(backend)

    with pytest.raises(OpenAIServingError, match="model"):
        service.create_completion({"model": [], "prompt": "hello", "max_tokens": 2})

    assert backend.calls == []


def test_chat_completion_validates_model_before_generation():
    backend = RecordingBackend()
    service = OpenAICompletionService(backend)

    with pytest.raises(OpenAIServingError, match="model"):
        service.create_chat_completion(
            {
                "model": [],
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 2,
            }
        )

    assert backend.calls == []


def test_http_endpoints_return_json():
    backend = RecordingBackend()
    service = OpenAICompletionService(backend)
    server = create_server("127.0.0.1", 0, service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        models = _get_json(f"{base_url}/v1/models")
        completion = _post_json(
            f"{base_url}/v1/completions",
            {"prompt": "hello", "max_tokens": 2, "temperature": 0},
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert models["data"][0]["id"] == "recording-model"
    assert completion["choices"][0]["text"] == "hello completion 0"


def test_http_rejects_invalid_and_oversized_request_bodies():
    backend = RecordingBackend()
    service = OpenAICompletionService(backend)
    server = create_server("127.0.0.1", 0, service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        invalid_length_status, invalid_length = _post_with_headers(
            f"{base_url}/v1/responses",
            {
                "Content-Type": "application/json",
                "Content-Length": "not-an-int",
            },
        )
        oversized_status, oversized = _post_with_headers(
            f"{base_url}/v1/responses",
            {
                "Content-Type": "application/json",
                "Content-Length": str(MAX_REQUEST_BODY_BYTES + 1),
            },
        )
        invalid_utf_status, invalid_utf = _post_with_headers(
            f"{base_url}/v1/responses",
            {
                "Content-Type": "application/json",
                "Content-Length": "1",
            },
            body=b"\xff",
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert invalid_length_status == 400
    assert invalid_length["error"]["message"] == "Content-Length must be an integer."
    assert oversized_status == 413
    assert oversized["error"]["message"] == "Request body is too large."
    assert invalid_utf_status == 400
    assert invalid_utf["error"]["message"] == "Request body is not valid JSON."


def test_responses_endpoint_returns_output_text_shape():
    service = OpenAICompletionService(RecordingBackend())

    payload = service.create_response(
        {
            "model": "client-model",
            "instructions": "be brief",
            "input": "hello",
            "max_output_tokens": 3,
            "temperature": 0,
        }
    )

    assert payload["object"] == "response"
    assert payload["status"] == "completed"
    assert payload["model"] == "client-model"
    assert payload["output"][0]["type"] == "message"
    assert payload["output"][0]["content"][0]["type"] == "output_text"
    assert payload["output_text"] == "system: be brief\nhello completion 0"
    assert payload["text"] == {"format": {"type": "text"}}
    assert payload["reasoning"] == {"effort": None, "summary": None}
    assert payload["usage"] == {
        "input_tokens": 4,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 2,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 6,
    }


def test_responses_stream_uses_typed_sse_events():
    service = OpenAICompletionService(RecordingBackend())

    events = list(
        service.stream_response(
            {
                "input": [{"role": "user", "content": "hello"}],
                "max_output_tokens": 3,
                "temperature": 0,
                "stream": True,
            }
        )
    )

    event_names = [name for name, _ in events]
    assert event_names[0] == "response.created"
    assert event_names[1] == "response.in_progress"
    assert "response.output_text.delta" in event_names
    assert event_names[-1] == "response.completed"
    deltas = [payload["delta"] for name, payload in events if name == "response.output_text.delta"]
    assert "".join(deltas) == "user: hello stream"


def test_response_store_retrieve_input_items_delete_and_previous_response():
    service = OpenAICompletionService(RecordingBackend())

    first = service.create_response({"input": "first", "max_output_tokens": 2})
    retrieved = service.retrieve_response(first["id"])
    items = service.list_response_input_items(first["id"], order="asc")
    second = service.create_response(
        {
            "input": "second",
            "previous_response_id": first["id"],
            "max_output_tokens": 2,
        }
    )
    deleted = service.delete_response(first["id"])

    assert retrieved["id"] == first["id"]
    assert items["object"] == "list"
    assert items["data"][0]["type"] == "message"
    assert "assistant:" in second["output_text"]
    assert deleted == {"id": first["id"], "object": "response.deleted", "deleted": True}
    with pytest.raises(OpenAIServingError, match="not found") as exc_info:
        service.retrieve_response(first["id"])
    assert exc_info.value.status_code == 404


def test_response_store_evicts_oldest_when_bounded():
    service = OpenAICompletionService(
        RecordingBackend(),
        response_store_max=2,
        response_store_ttl=None,
    )

    first = service.create_response({"input": "first", "max_output_tokens": 2})
    second = service.create_response({"input": "second", "max_output_tokens": 2})
    third = service.create_response({"input": "third", "max_output_tokens": 2})

    with pytest.raises(OpenAIServingError, match="not found"):
        service.retrieve_response(first["id"])
    assert service.retrieve_response(second["id"])["id"] == second["id"]
    assert service.retrieve_response(third["id"])["id"] == third["id"]


def test_response_store_evicts_expired_entries():
    service = OpenAICompletionService(
        RecordingBackend(),
        response_store_max=10,
        response_store_ttl=0.001,
    )

    response = service.create_response({"input": "hello", "max_output_tokens": 2})
    time.sleep(0.01)

    with pytest.raises(OpenAIServingError, match="not found"):
        service.retrieve_response(response["id"])


def test_response_output_message_can_be_reused_as_input_item():
    service = OpenAICompletionService(RecordingBackend())

    first = service.create_response({"input": "first", "max_output_tokens": 2})
    second = service.create_response(
        {
            "input": [
                first["output"][0],
                {"role": "user", "content": "next"},
            ],
            "max_output_tokens": 2,
        }
    )
    input_items = service.list_response_input_items(second["id"], order="asc")

    assert second["output_text"] == "assistant: first completion 0\nuser: next completion 0"
    assert input_items["data"][0]["role"] == "assistant"
    assert input_items["data"][0]["content"][0]["type"] == "output_text"


@pytest.mark.parametrize(
    "content",
    [
        [{"type": "input_text"}],
        [{"type": "output_text", "text": 42}],
        [{"type": "text", "input_text": 42}],
    ],
)
def test_responses_rejects_malformed_text_parts(content):
    service = OpenAICompletionService(RecordingBackend())

    with pytest.raises(OpenAIServingError, match="string text"):
        service.create_response({"input": [{"role": "user", "content": content}]})


def test_responses_rejects_unknown_input_item_types():
    backend = RecordingBackend()
    service = OpenAICompletionService(backend)

    with pytest.raises(OpenAIServingError, match="input_image") as exc_info:
        service.create_response({"input": [{"type": "input_image", "image_url": "file-id"}]})

    assert exc_info.value.param == "input"
    assert backend.calls == []


def test_rollout_backend_filters_request_fields_from_sampling_params():
    executor = RecordingRolloutExecutor()
    backend = RolloutCompletionBackend(executor)

    result = backend.generate(
        ["prompt"],
        n=1,
        max_tokens=7,
        temperature=0.2,
        top_p=0.8,
        stop=None,
        extra={
            "model": "client-model",
            "prompt": "prompt",
            "stream": True,
            "n": 1,
            "top_k": 20,
            "repetition_penalty": 1.1,
        },
    )

    assert result[0][0].text == "rollout completion"
    assert executor.calls[0]["sampling_params"] == {
        "max_tokens": 7,
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.1,
    }


def test_transformers_streaming_propagates_generation_errors():
    backend = TransformersCompletionBackend.__new__(TransformersCompletionBackend)
    backend.model_id = "failing-model"
    backend.device = "cpu"
    backend.model = FailingGenerateModel()
    backend.tokenizer = StreamingErrorTokenizer()

    with pytest.raises(OpenAIServingError, match="streaming"):
        list(
            backend.stream_generate(
                "hello",
                max_tokens=3,
                temperature=0,
                top_p=1,
                stop=None,
            )
        )


def test_transformers_streaming_cancels_generation_after_text_stop():
    model = CancellableStreamingModel()
    backend = TransformersCompletionBackend.__new__(TransformersCompletionBackend)
    backend.model_id = "cancellable-model"
    backend.device = "cpu"
    backend.model = model
    backend.tokenizer = StreamingErrorTokenizer()

    chunks = list(
        backend.stream_generate(
            "hello",
            max_tokens=10,
            temperature=0,
            top_p=1,
            stop="STOP",
        )
    )
    deadline = time.time() + 2
    while not model.finished.is_set() and time.time() < deadline:
        time.sleep(0.01)

    assert "".join(chunk.text for chunk in chunks if chunk.text) == "alpha "
    assert chunks[-1].finish_reason == "stop"
    assert model.cancel_seen is True


def test_responses_rejects_unsupported_multimodal_inputs():
    service = OpenAICompletionService(RecordingBackend())

    with pytest.raises(OpenAIServingError, match="input_image"):
        service.create_response(
            {
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"}
                        ],
                    }
                ]
            }
        )


def test_responses_rejects_unsupported_structured_output():
    service = OpenAICompletionService(RecordingBackend())

    with pytest.raises(OpenAIServingError, match="Structured output"):
        service.create_response(
            {
                "input": "hello",
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "answer",
                        "schema": {"type": "object", "properties": {}},
                    }
                },
            }
        )


def test_responses_rejects_bad_metadata_before_generation():
    backend = RecordingBackend()
    service = OpenAICompletionService(backend)

    with pytest.raises(OpenAIServingError) as exc_info:
        service.create_response({"input": "hello", "metadata": "bad"})

    assert exc_info.value.param == "metadata"
    assert backend.calls == []


def test_forced_function_tool_call_response_shape():
    service = OpenAICompletionService(RecordingBackend())

    response = service.create_response(
        {
            "input": "use the tool",
            "tools": [
                {
                    "type": "function",
                    "name": "lookup_metric",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
            "tool_choice": {
                "type": "function",
                "name": "lookup_metric",
            },
        }
    )

    function_call = response["output"][0]
    assert function_call["type"] == "function_call"
    assert function_call["name"] == "lookup_metric"
    assert function_call["arguments"] == "{}"
    assert response["output_text"] == ""


def test_required_function_tool_does_not_fake_required_arguments():
    service = OpenAICompletionService(RecordingBackend())

    with pytest.raises(OpenAIServingError, match="cannot infer required function arguments"):
        service.create_response(
            {
                "input": "use the tool",
                "tools": [
                    {
                        "type": "function",
                        "name": "lookup_metric",
                        "parameters": {
                            "type": "object",
                            "properties": {"metric": {"type": "string"}},
                            "required": ["metric"],
                        },
                    }
                ],
                "tool_choice": "required",
            }
        )


def test_required_function_tool_rejects_ambiguous_tool_choice():
    service = OpenAICompletionService(RecordingBackend())

    with pytest.raises(OpenAIServingError, match="exactly one"):
        service.create_response(
            {
                "input": "use the tool",
                "tools": [
                    {"type": "function", "name": "lookup_metric"},
                    {"type": "function", "name": "lookup_owner"},
                ],
                "tool_choice": "required",
            }
        )


def test_function_call_output_is_used_as_follow_up_input():
    service = OpenAICompletionService(RecordingBackend())

    response = service.create_response(
        {
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_lookup_metric",
                    "output": '{"value": 42}',
                }
            ]
        }
    )

    assert response["output_text"] == 'tool: {"value": 42} completion 0'


def test_reasoning_request_returns_reasoning_item_without_raw_thoughts():
    service = OpenAICompletionService(RecordingBackend())

    response = service.create_response(
        {
            "input": "hello",
            "reasoning": {"effort": "low", "summary": "auto"},
            "include": ["reasoning.encrypted_content"],
        }
    )

    reasoning = response["output"][0]
    assert reasoning["type"] == "reasoning"
    assert reasoning["summary"] == []
    assert reasoning["encrypted_content"] is None
    assert "thought" not in json.dumps(reasoning).lower()


def test_function_tool_call_stream_uses_function_argument_events():
    service = OpenAICompletionService(RecordingBackend())

    events = list(
        service.stream_response(
            {
                "input": "use the tool",
                "tools": [{"type": "function", "name": "lookup_metric"}],
                "tool_choice": "required",
                "stream": True,
            }
        )
    )

    names = [name for name, _ in events]
    assert "response.function_call_arguments.delta" in names
    assert "response.function_call_arguments.done" in names
    assert names[-1] == "response.completed"


def test_completion_stream_yields_initial_chunk_before_backend_streaming():
    class BlockingStreamBackend(RecordingBackend):
        def __init__(self):
            super().__init__()
            self.stream_started = False

        def stream_generate(self, *args, **kwargs):
            del args, kwargs
            self.stream_started = True
            raise AssertionError("backend stream should not start before the first SSE chunk")
            yield ""

    backend = BlockingStreamBackend()
    service = OpenAICompletionService(backend)

    events = service.stream_completion(
        {
            "prompt": "hello",
            "max_tokens": 3,
            "temperature": 0,
            "stream": True,
        }
    )
    first = next(events)
    events.close()

    assert backend.stream_started is False
    assert first[0] is None
    assert first[1]["choices"][0]["text"] == ""
    assert first[1]["choices"][0]["finish_reason"] is None


def test_completion_stream_preserves_finish_reason():
    service = OpenAICompletionService(RecordingBackend())

    events = list(
        service.stream_completion(
            {
                "prompt": "hello",
                "max_tokens": 3,
                "temperature": 0,
                "stream": True,
            }
        )
    )

    assert events[-1] == (None, "[DONE]")
    final_chunk = events[-2][1]
    assert final_chunk["choices"][0]["finish_reason"] == "length"


def test_chat_stream_uses_data_only_sse_chunks():
    service = OpenAICompletionService(RecordingBackend())

    events = list(
        service.stream_chat_completion(
            {
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 3,
                "temperature": 0,
                "stream": True,
            }
        )
    )

    assert all(name is None for name, _ in events)
    assert events[-1] == (None, "[DONE]")
    chunks = [payload for _, payload in events if isinstance(payload, dict)]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert any(chunk["choices"][0]["delta"].get("content") for chunk in chunks)
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"


def test_http_responses_stream_returns_event_stream():
    backend = RecordingBackend()
    service = OpenAICompletionService(backend)
    server = create_server("127.0.0.1", 0, service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        body = _post_raw(
            f"{base_url}/v1/responses",
            {"input": "hello", "max_output_tokens": 3, "stream": True},
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert "event: response.created" in body
    assert "event: response.in_progress" in body
    assert "event: response.output_text.delta" in body
    assert "event: response.completed" in body


def test_http_response_lifecycle_endpoints():
    backend = RecordingBackend()
    service = OpenAICompletionService(backend)
    server = create_server("127.0.0.1", 0, service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        response = _post_json(
            f"{base_url}/v1/responses",
            {"input": "hello", "max_output_tokens": 3},
        )
        retrieved = _get_json(f"{base_url}/v1/responses/{response['id']}")
        input_items = _get_json(f"{base_url}/v1/responses/{response['id']}/input_items")
        deleted = _delete_json(f"{base_url}/v1/responses/{response['id']}")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert retrieved["id"] == response["id"]
    assert input_items["object"] == "list"
    assert input_items["data"][0]["type"] == "message"
    assert deleted["deleted"] is True


@pytest.mark.skipif(
    os.environ.get("RL_KERNEL_RUN_LOCAL_SERVE_SMOKE") != "1",
    reason="Set RL_KERNEL_RUN_LOCAL_SERVE_SMOKE=1 to run the real local GPU server smoke.",
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for local GPU smoke.")
def test_transformers_backend_cuda_http_smoke():
    model_id = os.environ.get("RL_KERNEL_SERVE_SMOKE_MODEL", "sshleifer/tiny-gpt2")
    backend = TransformersCompletionBackend(model_id, device="cuda", dtype="float16")
    assert next(backend.model.parameters()).device.type == "cuda"

    server = create_server("127.0.0.1", 0, OpenAICompletionService(backend))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        chat = _post_json(
            f"{base_url}/v1/chat/completions",
            {
                "model": model_id,
                "messages": [{"role": "user", "content": "Say hello"}],
                "max_tokens": 4,
                "temperature": 0,
            },
        )
        completion = _post_json(
            f"{base_url}/v1/completions",
            {
                "model": model_id,
                "prompt": "RL-Kernel local GPU",
                "max_tokens": 4,
                "temperature": 0,
            },
        )
        response = _post_json(
            f"{base_url}/v1/responses",
            {
                "model": model_id,
                "input": "RL-Kernel responses API",
                "max_output_tokens": 4,
                "temperature": 0,
            },
        )
        response_stream = _post_raw(
            f"{base_url}/v1/responses",
            {
                "model": model_id,
                "input": "RL-Kernel streaming",
                "max_output_tokens": 4,
                "temperature": 0,
                "stream": True,
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert chat["object"] == "chat.completion"
    assert chat["choices"][0]["message"]["role"] == "assistant"
    assert chat["usage"]["completion_tokens"] > 0
    assert completion["object"] == "text_completion"
    assert completion["choices"][0]["text"]
    assert response["object"] == "response"
    assert response["output_text"]
    assert "event: response.output_text.delta" in response_stream
    assert "event: response.completed" in response_stream


def _get_json(url: str) -> dict[str, object]:
    with urllib.request.urlopen(_require_http_url(url), timeout=15) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _post_json(url: str, payload: Mapping[str, object]) -> dict[str, object]:
    request = urllib.request.Request(  # noqa: S310
        _require_http_url(url),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        raise AssertionError(f"HTTP {exc.code}: {body}") from exc


def _post_raw(url: str, payload: Mapping[str, object]) -> str:
    request = urllib.request.Request(  # noqa: S310
        _require_http_url(url),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        raise AssertionError(f"HTTP {exc.code}: {body}") from exc


def _delete_json(url: str) -> dict[str, object]:
    request = urllib.request.Request(_require_http_url(url), method="DELETE")  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=15) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        raise AssertionError(f"HTTP {exc.code}: {body}") from exc


def _post_with_headers(
    url: str,
    headers: Mapping[str, str],
    *,
    body: bytes = b"",
) -> tuple[int, dict[str, object]]:
    parsed = urllib.parse.urlparse(_require_http_url(url))
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=15)
    try:
        connection.putrequest("POST", path)
        for key, value in headers.items():
            connection.putheader(key, value)
        connection.endheaders()
        if body:
            connection.send(body)
        response = connection.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        return response.status, payload
    finally:
        connection.close()


def _require_http_url(url: str) -> str:
    scheme = urllib.parse.urlparse(url).scheme
    if scheme not in {"http", "https"}:
        raise AssertionError(f"Unsupported test URL scheme: {scheme!r}")
    return url
