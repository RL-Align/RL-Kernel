# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping, NoReturn, Optional, Protocol, Sequence
from urllib.parse import parse_qs, urlparse

import torch

from rl_engine.executors.rollout import RolloutExecutor
from rl_engine.utils.logger import logger

MAX_REQUEST_BODY_BYTES = 10 * 1024 * 1024


@dataclass(frozen=True)
class GeneratedText:
    text: str
    finish_reason: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass(frozen=True)
class StreamChunk:
    text: str = ""
    finish_reason: Optional[str] = None


@dataclass(frozen=True)
class StoredResponse:
    payload: Mapping[str, Any]
    input_items: list[Mapping[str, Any]]
    stored_at: float


class CompletionBackend(Protocol):
    model_id: str

    def format_chat_prompt(self, messages: Sequence[Mapping[str, Any]]) -> str: ...

    def generate(
        self,
        prompts: Sequence[str],
        *,
        n: int,
        max_tokens: int,
        temperature: Optional[float],
        top_p: Optional[float],
        stop: Optional[str | Sequence[str]],
        extra: Optional[Mapping[str, Any]] = None,
    ) -> list[list[GeneratedText]]: ...

    def stream_generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: Optional[float],
        top_p: Optional[float],
        stop: Optional[str | Sequence[str]],
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Iterator[StreamChunk | str]: ...

    def count_tokens(self, text: str) -> int: ...


class OpenAIServingError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int = HTTPStatus.BAD_REQUEST,
        error_type: str = "invalid_request_error",
        param: Optional[str] = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.param = param

    def to_payload(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": None,
            }
        }


class _CancelGenerationCriteria:
    def __init__(self, stop_event: threading.Event):
        self.stop_event = stop_event

    def __call__(self, input_ids: torch.Tensor, scores: Any, **kwargs: Any) -> torch.Tensor:
        del scores, kwargs
        return torch.full(
            (input_ids.shape[0],),
            self.stop_event.is_set(),
            device=input_ids.device,
            dtype=torch.bool,
        )


class TransformersCompletionBackend:
    """Local causal-LM backend for OpenAI-compatible development serving."""

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = False,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.device = _resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        dtype_value = _resolve_dtype(dtype, self.device)
        model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if dtype_value is not None:
            model_kwargs["dtype"] = dtype_value
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        except TypeError:
            if dtype_value is not None:
                model_kwargs.pop("dtype", None)
                model_kwargs["torch_dtype"] = dtype_value
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Loaded %s for OpenAI-compatible serving on %s", model_id, self.device)

    def format_chat_prompt(self, messages: Sequence[Mapping[str, Any]]) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=True,
            )
        return fallback_chat_prompt(messages)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    @torch.inference_mode()
    def generate(
        self,
        prompts: Sequence[str],
        *,
        n: int,
        max_tokens: int,
        temperature: Optional[float],
        top_p: Optional[float],
        stop: Optional[str | Sequence[str]],
        extra: Optional[Mapping[str, Any]] = None,
    ) -> list[list[GeneratedText]]:
        expanded_prompts = [prompt for prompt in prompts for _ in range(n)]
        encoded = self.tokenizer(
            expanded_prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        generation_kwargs = self._generation_kwargs(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            extra=extra,
        )
        output_ids = self.model.generate(**encoded, **generation_kwargs)
        prompt_width = int(encoded["input_ids"].shape[1])
        prompt_token_counts = [int(count) for count in encoded["attention_mask"].sum(dim=1)]

        flat: list[GeneratedText] = []
        for row_index, sequence_ids in enumerate(output_ids):
            completion_ids = sequence_ids[prompt_width:].detach().cpu().tolist()
            text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
            text, stop_hit = _apply_stop(text, stop)
            finish_reason = _finish_reason(
                completion_ids,
                max_tokens=max_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                stop_hit=stop_hit,
            )
            flat.append(
                GeneratedText(
                    text=text,
                    finish_reason=finish_reason,
                    prompt_tokens=prompt_token_counts[row_index],
                    completion_tokens=len(completion_ids),
                )
            )
        return _group_generated(flat, batch_size=len(prompts), n=n)

    def _generation_kwargs(
        self,
        *,
        max_tokens: int,
        temperature: Optional[float],
        top_p: Optional[float],
        extra: Optional[Mapping[str, Any]],
    ) -> dict[str, Any]:
        temperature_value = float(temperature) if temperature is not None else None
        do_sample = temperature_value is not None and temperature_value > 0
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.tokenizer.eos_token_id is not None:
            kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        if do_sample:
            kwargs["temperature"] = temperature_value
        if top_p is not None and float(top_p) < 1.0:
            kwargs["top_p"] = float(top_p)
        if extra:
            allowed = {"top_k", "repetition_penalty", "no_repeat_ngram_size"}
            kwargs.update({key: extra[key] for key in allowed if key in extra})
        return kwargs

    def stream_generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: Optional[float],
        top_p: Optional[float],
        stop: Optional[str | Sequence[str]],
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Iterator[StreamChunk]:
        from transformers import StoppingCriteriaList, TextIteratorStreamer

        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=False)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        prompt_width = int(encoded["input_ids"].shape[1])
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generation_kwargs = self._generation_kwargs(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            extra=extra,
        )
        stop_event = threading.Event()
        generation_kwargs["stopping_criteria"] = StoppingCriteriaList(
            [_CancelGenerationCriteria(stop_event)]
        )
        generation_kwargs["streamer"] = streamer

        generation_errors: list[Exception] = []
        generation_outputs: list[Any] = []

        def _run_generate() -> None:
            try:
                generation_outputs.append(self.model.generate(**encoded, **generation_kwargs))
            except Exception as exc:
                generation_errors.append(exc)
                streamer.end()

        thread = threading.Thread(
            target=_run_generate,
            daemon=True,
        )
        thread.start()
        finish_reason: Optional[str] = None
        try:
            for chunk in _stream_with_stop(streamer, stop):
                if generation_errors:
                    raise OpenAIServingError(
                        "Transformers generation failed during streaming.",
                        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                        error_type="server_error",
                    ) from generation_errors[0]
                if chunk.finish_reason:
                    finish_reason = chunk.finish_reason
                    stop_event.set()
                    continue
                yield chunk
            if finish_reason is None:
                thread.join(timeout=5)
            if generation_errors:
                raise OpenAIServingError(
                    "Transformers generation failed during streaming.",
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                    error_type="server_error",
                ) from generation_errors[0]
            if finish_reason is None:
                finish_reason = _stream_finish_reason_from_output(
                    generation_outputs[0] if generation_outputs else None,
                    prompt_width=prompt_width,
                    max_tokens=max_tokens,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            yield StreamChunk(finish_reason=finish_reason)
        finally:
            if thread.is_alive():
                stop_event.set()
            else:
                thread.join(timeout=5)


class RolloutCompletionBackend:
    """Adapter from OpenAI-compatible requests to RL-Kernel rollout generation."""

    def __init__(self, executor: RolloutExecutor):
        self.executor = executor
        self.model_id = str(executor.config.get("model") or "rl-kernel-rollout")

    def format_chat_prompt(self, messages: Sequence[Mapping[str, Any]]) -> str:
        return fallback_chat_prompt(messages)

    def count_tokens(self, text: str) -> int:
        tokenizer = getattr(getattr(self.executor, "sampler", None), "tokenizer", None)
        if tokenizer is None:
            return 0
        encode = getattr(tokenizer, "encode", None)
        if not callable(encode):
            return 0
        return len(encode(text))

    def generate(
        self,
        prompts: Sequence[str],
        *,
        n: int,
        max_tokens: int,
        temperature: Optional[float],
        top_p: Optional[float],
        stop: Optional[str | Sequence[str]],
        extra: Optional[Mapping[str, Any]] = None,
    ) -> list[list[GeneratedText]]:
        sampling_params = _sampling_param_overrides(extra)
        sampling_params["max_tokens"] = max_tokens
        if temperature is not None:
            sampling_params["temperature"] = temperature
        if top_p is not None:
            sampling_params["top_p"] = top_p

        payload = self.executor.generate_candidates(
            list(prompts),
            num_generations=n,
            sampling_params=sampling_params,
        )
        grouped = payload.get("normalized_outputs")
        if not isinstance(grouped, Sequence):
            raise OpenAIServingError(
                "Rollout backend returned no normalized outputs.",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                error_type="server_error",
            )

        results: list[list[GeneratedText]] = []
        for group in grouped:
            generated_group: list[GeneratedText] = []
            for candidate in group:
                text, stop_hit = _apply_stop(str(candidate.text), stop)
                finish_reason = "stop" if stop_hit else (candidate.finish_reason or "stop")
                prompt_tokens = len(candidate.prompt_token_ids or [])
                generated_group.append(
                    GeneratedText(
                        text=text,
                        finish_reason=finish_reason,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=len(candidate.token_ids),
                    )
                )
            results.append(generated_group)
        return results

    def stream_generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: Optional[float],
        top_p: Optional[float],
        stop: Optional[str | Sequence[str]],
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Iterator[StreamChunk]:
        generated = self.generate(
            [prompt],
            n=1,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            extra=extra,
        )[0][0]
        for chunk in _chunk_text(generated.text):
            yield StreamChunk(chunk)
        yield StreamChunk(finish_reason=generated.finish_reason)


class OpenAICompletionService:
    def __init__(
        self,
        backend: CompletionBackend,
        *,
        response_store_max: Optional[int] = None,
        response_store_ttl: Optional[float] = None,
    ):
        self.backend = backend
        self._responses: dict[str, StoredResponse] = {}
        self._responses_lock = threading.Lock()
        self._response_store_max = (
            _env_non_negative_int("RL_KERNEL_OPENAI_RESPONSE_STORE_MAX", 1000)
            if response_store_max is None
            else max(0, response_store_max)
        )
        response_store_ttl = (
            _env_non_negative_float("RL_KERNEL_OPENAI_RESPONSE_STORE_TTL_SECONDS", 3600.0)
            if response_store_ttl is None
            else response_store_ttl
        )
        self._response_store_ttl = (
            response_store_ttl
            if response_store_ttl is not None and response_store_ttl > 0
            else None
        )

    def models(self) -> dict[str, Any]:
        created = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": self.backend.model_id,
                    "object": "model",
                    "created": created,
                    "owned_by": "rl-kernel",
                }
            ],
        }

    def retrieve_response(self, response_id: str) -> Mapping[str, Any]:
        record = self._response_record(response_id)
        return record.payload

    def delete_response(self, response_id: str) -> dict[str, Any]:
        with self._responses_lock:
            self._evict_stored_responses_locked()
            record = self._responses.pop(response_id, None)
        if record is None:
            raise OpenAIServingError(
                f"Response {response_id!r} was not found.",
                status_code=HTTPStatus.NOT_FOUND,
                error_type="not_found_error",
            )
        return {"id": response_id, "object": "response.deleted", "deleted": True}

    def list_response_input_items(
        self,
        response_id: str,
        *,
        limit: int = 20,
        order: str = "desc",
        after: Optional[str] = None,
    ) -> dict[str, Any]:
        record = self._response_record(response_id)
        items = list(record.input_items)
        if order not in {"asc", "desc"}:
            raise OpenAIServingError("order must be 'asc' or 'desc'.", param="order")
        if order == "desc":
            items.reverse()
        if after is not None:
            after_index = next(
                (index for index, item in enumerate(items) if item.get("id") == after),
                None,
            )
            if after_index is not None:
                items = items[after_index + 1 :]
        limit = max(1, min(int(limit), 100))
        page = items[:limit]
        return {
            "object": "list",
            "data": page,
            "first_id": page[0]["id"] if page else None,
            "last_id": page[-1]["id"] if page else None,
            "has_more": len(items) > len(page),
        }

    def create_completion(self, body: Mapping[str, Any]) -> dict[str, Any]:
        _reject_unsupported_completion_fields(body)
        prompts = _completion_prompts(body)
        params = _generation_params(body)
        response_model = _response_model(body, self.backend.model_id)
        groups = self.backend.generate(prompts, **params)
        created = int(time.time())
        choices = []
        usage = _empty_usage()

        flat_index = 0
        for group in groups:
            for generated in group:
                _accumulate_usage(usage, generated)
                choices.append(
                    {
                        "text": generated.text,
                        "index": flat_index,
                        "logprobs": None,
                        "finish_reason": generated.finish_reason,
                    }
                )
                flat_index += 1

        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        return {
            "id": f"cmpl-{uuid.uuid4().hex}",
            "object": "text_completion",
            "created": created,
            "model": response_model,
            "choices": choices,
            "usage": usage,
        }

    def stream_completion(
        self,
        body: Mapping[str, Any],
    ) -> Iterator[tuple[None, Mapping[str, Any] | str]]:
        _reject_unsupported_completion_fields(body)
        prompts = _completion_prompts(body)
        if len(prompts) != 1:
            raise OpenAIServingError("Streaming completions support one prompt per request.")
        params = _generation_params(body)
        _reject_stream_n(params["n"])
        response_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        response_model = _response_model(body, self.backend.model_id)
        yield None, _completion_stream_chunk(response_id, created, response_model, "")
        finish_reason = "stop"
        for chunk in _stream_backend_chunks(self.backend, prompts[0], params):
            if chunk.finish_reason is not None:
                finish_reason = chunk.finish_reason
            if chunk.text:
                yield None, _completion_stream_chunk(
                    response_id,
                    created,
                    response_model,
                    chunk.text,
                )
        yield None, {
            "id": response_id,
            "object": "text_completion",
            "created": created,
            "model": response_model,
            "choices": [
                {
                    "text": "",
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
        }
        yield None, "[DONE]"

    def create_chat_completion(self, body: Mapping[str, Any]) -> dict[str, Any]:
        _reject_unsupported_chat_fields(body)
        messages = _chat_messages(body)
        prompt = self.backend.format_chat_prompt(messages)
        params = _generation_params(body)
        response_model = _response_model(body, self.backend.model_id)
        groups = self.backend.generate([prompt], **params)
        created = int(time.time())
        choices = []
        usage = _empty_usage()

        for index, generated in enumerate(groups[0]):
            _accumulate_usage(usage, generated)
            choices.append(
                {
                    "index": index,
                    "message": {
                        "role": "assistant",
                        "content": generated.text,
                    },
                    "finish_reason": generated.finish_reason,
                }
            )

        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": created,
            "model": response_model,
            "choices": choices,
            "usage": usage,
        }

    def stream_chat_completion(
        self,
        body: Mapping[str, Any],
    ) -> Iterator[tuple[None, Mapping[str, Any] | str]]:
        _reject_unsupported_chat_fields(body)
        messages = _chat_messages(body)
        prompt = self.backend.format_chat_prompt(messages)
        params = _generation_params(body)
        _reject_stream_n(params["n"])
        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        response_model = _response_model(body, self.backend.model_id)

        yield None, {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": response_model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            ],
        }
        finish_reason = "stop"
        for chunk in _stream_backend_chunks(self.backend, prompt, params):
            if chunk.finish_reason is not None:
                finish_reason = chunk.finish_reason
            if chunk.text:
                yield None, {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": response_model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk.text},
                            "finish_reason": None,
                        }
                    ],
                }
        yield None, {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": response_model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason,
                }
            ],
        }
        yield None, "[DONE]"

    def create_response(self, body: Mapping[str, Any]) -> dict[str, Any]:
        _reject_unsupported_response_fields(body)
        input_items = _responses_input_items(body.get("input", ""))
        prompt = self._responses_prompt_with_context(body)
        params = _generation_params(body, max_tokens_field="max_output_tokens")
        _reject_response_n(params["n"])
        response_model = _response_model(body, self.backend.model_id)
        output_items: list[dict[str, Any]] = []
        reasoning = _reasoning_item_if_requested(body)
        if reasoning is not None:
            output_items.append(reasoning)

        function_call = _forced_function_call_item(body)
        generated: Optional[GeneratedText] = None
        if function_call is not None:
            output_items.append(function_call)
        else:
            generated = self.backend.generate([prompt], **params)[0][0]
            output_items.append(
                _response_message_item(
                    f"msg_{uuid.uuid4().hex}",
                    generated.text,
                    status="completed",
                )
            )
        response = _response_object(
            response_id=f"resp_{uuid.uuid4().hex}",
            model=response_model,
            generated=generated,
            status="completed",
            instructions=body.get("instructions"),
            created_at=None,
            output_items=output_items,
            request_body=body,
        )
        self._store_response_if_requested(response, input_items, body)
        return response

    def stream_response(
        self,
        body: Mapping[str, Any],
    ) -> Iterator[tuple[str, Mapping[str, Any]]]:
        _reject_unsupported_response_fields(body)
        input_items = _responses_input_items(body.get("input", ""))
        prompt = self._responses_prompt_with_context(body)
        params = _generation_params(body, max_tokens_field="max_output_tokens")
        _reject_response_n(params["n"])
        response_model = _response_model(body, self.backend.model_id)
        response_id = f"resp_{uuid.uuid4().hex}"
        item_id = f"msg_{uuid.uuid4().hex}"
        created = int(time.time())
        text_parts: list[str] = []
        sequence_number = 0

        reasoning = _reasoning_item_if_requested(body)
        function_call = _forced_function_call_item(body)
        started = _response_object(
            response_id=response_id,
            model=response_model,
            generated=None,
            status="in_progress",
            instructions=body.get("instructions"),
            created_at=created,
            request_body=body,
        )
        yield "response.created", {"type": "response.created", "response": started}
        yield "response.in_progress", {
            "type": "response.in_progress",
            "response": started,
        }
        output_items: list[dict[str, Any]] = []
        if reasoning is not None:
            output_items.append(reasoning)
            yield "response.output_item.added", {
                "type": "response.output_item.added",
                "response_id": response_id,
                "output_index": 0,
                "item": {**reasoning, "status": "in_progress"},
            }
            yield "response.output_item.done", {
                "type": "response.output_item.done",
                "response_id": response_id,
                "output_index": 0,
                "item": reasoning,
            }
        if function_call is not None:
            output_index = len(output_items)
            output_items.append(function_call)
            in_progress_call = {**function_call, "status": "in_progress", "arguments": ""}
            yield "response.output_item.added", {
                "type": "response.output_item.added",
                "response_id": response_id,
                "output_index": output_index,
                "item": in_progress_call,
            }
            yield "response.function_call_arguments.delta", {
                "type": "response.function_call_arguments.delta",
                "response_id": response_id,
                "item_id": function_call["id"],
                "output_index": output_index,
                "delta": function_call["arguments"],
            }
            yield "response.function_call_arguments.done", {
                "type": "response.function_call_arguments.done",
                "response_id": response_id,
                "item_id": function_call["id"],
                "output_index": output_index,
                "arguments": function_call["arguments"],
            }
            yield "response.output_item.done", {
                "type": "response.output_item.done",
                "response_id": response_id,
                "output_index": output_index,
                "item": function_call,
            }
            completed = _response_object(
                response_id=response_id,
                model=response_model,
                generated=None,
                status="completed",
                instructions=body.get("instructions"),
                created_at=created,
                output_items=output_items,
                request_body=body,
            )
            self._store_response_if_requested(completed, input_items, body)
            yield "response.completed", {"type": "response.completed", "response": completed}
            return

        output_index = len(output_items)
        yield "response.output_item.added", {
            "type": "response.output_item.added",
            "response_id": response_id,
            "output_index": output_index,
            "item": _response_message_item(item_id, "", status="in_progress"),
        }
        yield "response.content_part.added", {
            "type": "response.content_part.added",
            "response_id": response_id,
            "item_id": item_id,
            "output_index": output_index,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        }

        finish_reason = "stop"
        for chunk in _stream_backend_chunks(self.backend, prompt, params):
            if chunk.finish_reason is not None:
                finish_reason = chunk.finish_reason
            if not chunk.text:
                continue
            text_parts.append(chunk.text)
            sequence_number += 1
            yield "response.output_text.delta", {
                "type": "response.output_text.delta",
                "response_id": response_id,
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "delta": chunk.text,
                "sequence_number": sequence_number,
            }

        text = "".join(text_parts)
        generated = GeneratedText(
            text=text,
            finish_reason=finish_reason,
            prompt_tokens=_count_backend_tokens(self.backend, prompt),
            completion_tokens=_count_backend_tokens(self.backend, text) or len(text_parts),
        )
        message = _response_message_item(item_id, text, status="completed")
        yield "response.output_text.done", {
            "type": "response.output_text.done",
            "response_id": response_id,
            "item_id": item_id,
            "output_index": output_index,
            "content_index": 0,
            "text": text,
        }
        yield "response.content_part.done", {
            "type": "response.content_part.done",
            "response_id": response_id,
            "item_id": item_id,
            "output_index": output_index,
            "content_index": 0,
            "part": message["content"][0],
        }
        yield "response.output_item.done", {
            "type": "response.output_item.done",
            "response_id": response_id,
            "output_index": output_index,
            "item": message,
        }
        output_items.append(message)
        completed = _response_object(
            response_id=response_id,
            model=response_model,
            generated=generated,
            status="completed",
            instructions=body.get("instructions"),
            created_at=created,
            output_items=output_items,
            request_body=body,
        )
        self._store_response_if_requested(completed, input_items, body)
        yield "response.completed", {"type": "response.completed", "response": completed}

    def _responses_prompt_with_context(self, body: Mapping[str, Any]) -> str:
        prompt = _responses_prompt(body)
        previous_response_id = body.get("previous_response_id")
        if previous_response_id is None:
            return prompt
        if not isinstance(previous_response_id, str):
            raise OpenAIServingError(
                "previous_response_id must be a string.",
                param="previous_response_id",
            )
        previous = self.retrieve_response(previous_response_id)
        previous_text = str(previous.get("output_text") or "")
        if not previous_text:
            return prompt
        return f"assistant: {previous_text}\n{prompt}"

    def _store_response_if_requested(
        self,
        response: Mapping[str, Any],
        input_items: list[Mapping[str, Any]],
        body: Mapping[str, Any],
    ) -> None:
        if body.get("store", True) is False:
            return
        if self._response_store_max <= 0:
            return
        with self._responses_lock:
            self._evict_stored_responses_locked()
            self._responses[str(response["id"])] = StoredResponse(
                payload=response,
                input_items=input_items,
                stored_at=time.monotonic(),
            )
            self._evict_stored_responses_locked()

    def _response_record(self, response_id: str) -> StoredResponse:
        with self._responses_lock:
            self._evict_stored_responses_locked()
            record = self._responses.get(response_id)
        if record is None:
            raise OpenAIServingError(
                f"Response {response_id!r} was not found.",
                status_code=HTTPStatus.NOT_FOUND,
                error_type="not_found_error",
            )
        return record

    def _evict_stored_responses_locked(self) -> None:
        now = time.monotonic()
        if self._response_store_ttl is not None:
            expired_ids = [
                response_id
                for response_id, record in self._responses.items()
                if now - record.stored_at > self._response_store_ttl
            ]
            for response_id in expired_ids:
                self._responses.pop(response_id, None)
        excess = len(self._responses) - self._response_store_max
        if excess <= 0:
            return
        for response_id in list(self._responses)[:excess]:
            self._responses.pop(response_id, None)


def fallback_chat_prompt(messages: Sequence[Mapping[str, Any]]) -> str:
    lines = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = _content_to_text(message.get("content", ""))
        lines.append(f"{role}: {content}")
    lines.append("assistant:")
    return "\n".join(lines)


def make_handler(service: OpenAICompletionService) -> type[BaseHTTPRequestHandler]:
    class OpenAIHTTPHandler(BaseHTTPRequestHandler):
        server_version = "RLKernelOpenAIServer/0.1"

        def do_GET(self) -> None:
            try:
                parsed = urlparse(self.path)
                path = parsed.path
                if path in {"/health", "/healthz"}:
                    self._send_json(
                        HTTPStatus.OK,
                        {
                            "status": "ok",
                            "model": service.backend.model_id,
                        },
                    )
                    return
                if path == "/v1/models":
                    self._send_json(HTTPStatus.OK, service.models())
                    return
                response_id, suffix = _response_route(path)
                if response_id and suffix is None:
                    self._send_json(HTTPStatus.OK, service.retrieve_response(response_id))
                    return
                if response_id and suffix == "input_items":
                    query = parse_qs(parsed.query)
                    self._send_json(
                        HTTPStatus.OK,
                        service.list_response_input_items(
                            response_id,
                            limit=int(query.get("limit", ["20"])[0]),
                            order=query.get("order", ["desc"])[0],
                            after=query.get("after", [None])[0],
                        ),
                    )
                    return
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    _error_payload("Unknown endpoint.", error_type="not_found_error"),
                )
            except OpenAIServingError as exc:
                self._send_json(exc.status_code, exc.to_payload())
            except Exception as exc:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    _error_payload(str(exc)),
                )

        def do_DELETE(self) -> None:
            parsed = urlparse(self.path)
            response_id, suffix = _response_route(parsed.path)
            try:
                if response_id and suffix is None:
                    self._send_json(HTTPStatus.OK, service.delete_response(response_id))
                    return
                raise OpenAIServingError(
                    "Unknown endpoint.",
                    status_code=HTTPStatus.NOT_FOUND,
                    error_type="not_found_error",
                )
            except OpenAIServingError as exc:
                self._send_json(exc.status_code, exc.to_payload())

        def do_POST(self) -> None:
            try:
                body = self._read_json()
                path = urlparse(self.path).path
                if path == "/v1/completions":
                    if body.get("stream") is True:
                        self._send_sse(service.stream_completion(body))
                        return
                    self._send_json(HTTPStatus.OK, service.create_completion(body))
                    return
                if path == "/v1/chat/completions":
                    if body.get("stream") is True:
                        self._send_sse(service.stream_chat_completion(body))
                        return
                    self._send_json(HTTPStatus.OK, service.create_chat_completion(body))
                    return
                if path == "/v1/responses":
                    if body.get("stream") is True:
                        self._send_sse(service.stream_response(body))
                        return
                    self._send_json(HTTPStatus.OK, service.create_response(body))
                    return
                raise OpenAIServingError(
                    "Unknown endpoint.",
                    status_code=HTTPStatus.NOT_FOUND,
                    error_type="not_found_error",
                )
            except OpenAIServingError as exc:
                self._send_json(exc.status_code, exc.to_payload())
            except Exception as exc:
                logger.exception("OpenAI-compatible serving request failed")
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    _error_payload(str(exc), error_type="server_error"),
                )

        def log_message(self, message_format: str, *args: Any) -> None:
            logger.info("%s - %s", self.address_string(), message_format % args)

        def _read_json(self) -> Mapping[str, Any]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise OpenAIServingError("Content-Length must be an integer.") from exc
            if length <= 0:
                raise OpenAIServingError("Request body must be a JSON object.")
            if length > MAX_REQUEST_BODY_BYTES:
                raise OpenAIServingError(
                    "Request body is too large.",
                    status_code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                )
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise OpenAIServingError("Request body is not valid JSON.") from exc
            if not isinstance(payload, Mapping):
                raise OpenAIServingError("Request body must be a JSON object.")
            return payload

        def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_sse(
            self,
            events: Iterator[tuple[Optional[str], Mapping[str, Any] | str]],
        ) -> None:
            iterator = iter(events)
            try:
                first_event = next(iterator)
            except StopIteration:
                first_event = None
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            for event_name, payload in [first_event] if first_event is not None else []:
                self._write_sse_event(event_name, payload)
            try:
                for event_name, payload in iterator:
                    self._write_sse_event(event_name, payload)
            except OpenAIServingError as exc:
                self._write_sse_event("error", {"type": "error", **exc.to_payload()})
            except Exception as exc:
                logger.exception("OpenAI-compatible streaming request failed")
                self._write_sse_event(
                    "error",
                    {
                        "type": "error",
                        **_error_payload(str(exc), error_type="server_error"),
                    },
                )
            self.close_connection = True

        def _write_sse_event(
            self,
            event_name: Optional[str],
            payload: Mapping[str, Any] | str,
        ) -> None:
            if event_name:
                self.wfile.write(f"event: {event_name}\n".encode("utf-8"))
            if isinstance(payload, str):
                data = payload
            else:
                data = json.dumps(payload)
            self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
            self.wfile.flush()

    return OpenAIHTTPHandler


def create_server(
    host: str,
    port: int,
    service: OpenAICompletionService,
) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(service))


def build_backend_from_args(args: argparse.Namespace) -> CompletionBackend:
    if args.backend == "transformers":
        if not args.model:
            raise SystemExit("--model is required for the transformers backend")
        return TransformersCompletionBackend(
            args.model,
            device=args.device,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
    if args.backend == "rollout":
        if not args.model:
            raise SystemExit("--model is required for the rollout backend")
        executor = RolloutExecutor(
            {
                "model": args.model,
                "engine_kwargs": _json_arg(args.engine_kwargs, "engine-kwargs"),
                "sampling_params": _json_arg(args.sampling_params, "sampling-params"),
            }
        )
        return RolloutCompletionBackend(executor)
    raise SystemExit(f"Unsupported backend: {args.backend}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run an OpenAI-compatible RL-Kernel server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--backend", choices=("transformers", "rollout"), default="transformers")
    parser.add_argument("--model", help="Model id or local path for the selected backend.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:<index>.")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "float16", "bfloat16", "float32"),
        help="Model dtype for the transformers backend.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--engine-kwargs", default="{}", help="JSON object for vLLM LLM kwargs.")
    parser.add_argument(
        "--sampling-params",
        default="{}",
        help="JSON object with default sampling params.",
    )
    args = parser.parse_args(argv)

    service = OpenAICompletionService(build_backend_from_args(args))
    server = create_server(args.host, args.port, service)
    logger.info("Serving OpenAI-compatible API on http://%s:%s", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping OpenAI-compatible server")
    finally:
        server.server_close()
    return 0


def _completion_prompts(body: Mapping[str, Any]) -> list[str]:
    if "prompt" not in body:
        raise OpenAIServingError("Missing required field: prompt.", param="prompt")
    prompt = body["prompt"]
    if isinstance(prompt, str):
        return [prompt]
    if isinstance(prompt, Sequence) and not isinstance(prompt, (bytes, bytearray)):
        prompts = list(prompt)
        if prompts and all(isinstance(item, str) for item in prompts):
            return prompts
    raise OpenAIServingError(
        "Only string or list-of-string prompts are supported.",
        param="prompt",
    )


def _reject_unsupported_completion_fields(body: Mapping[str, Any]) -> None:
    if body.get("echo") is True:
        raise OpenAIServingError(
            "Completions echo is not supported by this local server.",
            param="echo",
        )
    if body.get("suffix") is not None:
        raise OpenAIServingError(
            "Completions suffix insertion is not supported by this local server.",
            param="suffix",
        )
    if body.get("logprobs") is not None:
        raise OpenAIServingError(
            "Completions logprobs are not supported by this local server.",
            param="logprobs",
        )
    if body.get("best_of") not in (None, 1):
        raise OpenAIServingError(
            "Completions best_of is not supported by this local server.",
            param="best_of",
        )
    if body.get("stream_options") is not None:
        raise OpenAIServingError(
            "Completions stream_options are not supported by this local server.",
            param="stream_options",
        )


def _chat_messages(body: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    messages = body.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes, bytearray)):
        raise OpenAIServingError("Missing or invalid required field: messages.", param="messages")
    normalized: list[Mapping[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise OpenAIServingError(
                f"messages[{index}] must be an object.",
                param="messages",
            )
        normalized_message = dict(message)
        normalized_message["content"] = _content_to_text(message.get("content", ""))
        normalized.append(normalized_message)
    if not normalized:
        raise OpenAIServingError("messages must not be empty.", param="messages")
    return normalized


def _reject_unsupported_chat_fields(body: Mapping[str, Any]) -> None:
    if body.get("tools"):
        raise OpenAIServingError(
            "Chat Completions tools are not supported by this local server; use /v1/responses.",
            param="tools",
        )
    if "tool_choice" in body and body.get("tool_choice") is not None:
        raise OpenAIServingError(
            "Chat Completions tool_choice is not supported by this local server.",
            param="tool_choice",
        )
    if body.get("functions"):
        raise OpenAIServingError(
            "Legacy Chat Completions functions are not supported by this local server.",
            param="functions",
        )
    if "function_call" in body and body.get("function_call") is not None:
        raise OpenAIServingError(
            "Legacy Chat Completions function_call is not supported by this local server.",
            param="function_call",
        )
    response_format = body.get("response_format")
    if response_format is not None:
        if not isinstance(response_format, Mapping) or response_format.get("type") != "text":
            raise OpenAIServingError(
                "Structured Chat Completions response formats are not supported.",
                param="response_format",
            )
    modalities = body.get("modalities")
    if modalities not in (None, ["text"], ("text",)):
        raise OpenAIServingError(
            "Only text Chat Completions are supported by this local server.",
            param="modalities",
        )
    if body.get("audio") is not None:
        raise OpenAIServingError(
            "Chat Completions audio output is not supported by this local server.",
            param="audio",
        )
    logprobs = body.get("logprobs")
    if logprobs is not None and logprobs is not False:
        raise OpenAIServingError(
            "Chat Completions logprobs are not supported by this local server.",
            param="logprobs",
        )
    if body.get("top_logprobs") not in (None, 0):
        raise OpenAIServingError(
            "Chat Completions logprobs are not supported by this local server.",
            param="top_logprobs",
        )


def _generation_params(
    body: Mapping[str, Any],
    *,
    max_tokens_field: str = "max_tokens",
) -> dict[str, Any]:
    n = _positive_int(body.get("n", 1), "n")
    max_tokens = _positive_int(
        body.get(max_tokens_field, body.get("max_tokens", 16)),
        max_tokens_field,
    )
    temperature = _optional_float(body.get("temperature", 1.0), "temperature")
    if temperature is not None and temperature < 0:
        raise OpenAIServingError("temperature must be non-negative.", param="temperature")
    top_p = _optional_float(body.get("top_p", 1.0), "top_p")
    if top_p is not None and not (0 < top_p <= 1):
        raise OpenAIServingError("top_p must be in the interval (0, 1].", param="top_p")
    return {
        "n": n,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stop": _normalize_stop(body.get("stop")),
        "extra": body,
    }


def _normalize_stop(value: Any) -> Optional[str | list[str]]:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        stop_values = list(value)
        if all(isinstance(item, str) for item in stop_values):
            return stop_values
    raise OpenAIServingError(
        "stop must be a string, a list of strings, or null.",
        param="stop",
    )


_SAMPLING_PARAM_OVERRIDE_KEYS = {
    "allowed_token_ids",
    "bad_words",
    "best_of",
    "detokenize",
    "frequency_penalty",
    "ignore_eos",
    "include_stop_str_in_output",
    "length_penalty",
    "logprobs",
    "min_p",
    "min_tokens",
    "presence_penalty",
    "prompt_logprobs",
    "repetition_penalty",
    "seed",
    "skip_special_tokens",
    "spaces_between_special_tokens",
    "stop_token_ids",
    "top_k",
    "truncate_prompt_tokens",
}


def _sampling_param_overrides(extra: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    if not extra:
        return {}
    return {key: extra[key] for key in _SAMPLING_PARAM_OVERRIDE_KEYS if key in extra}


def _reject_stream_n(n: int) -> None:
    if n != 1:
        raise OpenAIServingError("Streaming currently supports n=1 only.", param="n")


def _reject_response_n(n: int) -> None:
    if n != 1:
        raise OpenAIServingError("The Responses endpoint currently supports n=1 only.", param="n")


def _response_model(body: Mapping[str, Any], default_model: str) -> str:
    model = body.get("model")
    if model is None:
        return default_model
    if not isinstance(model, str) or not model:
        raise OpenAIServingError("model must be a non-empty string.", param="model")
    return model


def _positive_int(value: Any, param: str) -> int:
    if isinstance(value, bool):
        raise OpenAIServingError(f"{param} must be a positive integer.", param=param)
    if isinstance(value, int):
        parsed = value
    else:
        raise OpenAIServingError(f"{param} must be a positive integer.", param=param)
    if parsed < 1:
        raise OpenAIServingError(f"{param} must be a positive integer.", param=param)
    return parsed


def _env_non_negative_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        parsed = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer.") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return parsed


def _env_non_negative_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        parsed = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative number.") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative number.")
    return parsed


def _optional_float(value: Any, param: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise OpenAIServingError(f"{param} must be a number.", param=param)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise OpenAIServingError(f"{param} must be a number.", param=param) from exc


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (bytes, bytearray)):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, Mapping):
                part_type = part.get("type", "text")
                if part_type not in {"text", "input_text"}:
                    raise OpenAIServingError(
                        f"Chat content part type {part_type!r} is not supported "
                        "by this local server.",
                        param="messages",
                    )
                if not isinstance(part.get("text"), str):
                    raise OpenAIServingError("Chat text parts must include text.", param="messages")
                parts.append(part["text"])
            else:
                raise OpenAIServingError(
                    "Chat content parts must be strings or objects.",
                    param="messages",
                )
        return "".join(parts)
    raise OpenAIServingError(
        "Chat message content must be a string or text parts.",
        param="messages",
    )


def _responses_prompt(body: Mapping[str, Any]) -> str:
    if "input" not in body:
        raise OpenAIServingError("Missing required field: input.", param="input")
    prompt = _responses_input_to_text(body["input"])
    instructions = body.get("instructions")
    if instructions is not None:
        if not isinstance(instructions, str):
            raise OpenAIServingError("instructions must be a string.", param="instructions")
        prompt = f"system: {instructions}\n{prompt}"
    return prompt


def _responses_input_items(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, str):
        return [_input_message_item("user", value)]
    if isinstance(value, Mapping):
        return [_normalize_response_input_item(value)]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [_normalize_response_input_item(item) for item in value]
    raise OpenAIServingError(
        "input must be a string, an input item object, or a list of input items.",
        param="input",
    )


def _normalize_response_input_item(item: Any) -> Mapping[str, Any]:
    if isinstance(item, str):
        return _input_message_item("user", item)
    if not isinstance(item, Mapping):
        raise OpenAIServingError("Each Responses input item must be an object.", param="input")
    item_type = item.get("type", "message")
    if item_type == "message":
        role = str(item.get("role") or "user")
        return _input_message_item(role, _content_to_input_content(item.get("content", "")))
    if item_type in {
        "function_call_output",
        "custom_tool_call_output",
        "local_shell_call_output",
        "mcp_approval_response",
    }:
        normalized = dict(item)
        normalized.setdefault("id", f"item_{uuid.uuid4().hex}")
        return normalized
    _raise_unsupported_response_input_item(item_type)


def _input_message_item(role: str, content: Any) -> Mapping[str, Any]:
    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "status": "completed",
        "role": role,
        "content": _content_to_input_content(content),
    }


def _content_to_input_content(content: Any) -> list[Mapping[str, Any]]:
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    if isinstance(content, Sequence) and not isinstance(content, (bytes, bytearray)):
        normalized: list[Mapping[str, Any]] = []
        for part in content:
            if isinstance(part, str):
                normalized.append({"type": "input_text", "text": part})
            elif isinstance(part, Mapping):
                part_type = part.get("type", "input_text")
                if part_type in {"input_text", "output_text", "text"}:
                    text = _response_text_part_value(part)
                    normalized.append(
                        {
                            "type": "output_text" if part_type == "output_text" else "input_text",
                            "text": text,
                        }
                    )
                else:
                    _raise_unsupported_input_content(part_type)
            else:
                raise OpenAIServingError(
                    "Responses input content parts must be strings or objects.",
                    param="input",
                )
        return normalized
    raise OpenAIServingError(
        "Responses message content must be a string or text parts.",
        param="input",
    )


def _responses_input_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return _responses_item_to_text(value)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return "\n".join(_responses_item_to_text(item) for item in value)
    raise OpenAIServingError(
        "input must be a string, an input item object, or a list of input items.",
        param="input",
    )


def _responses_item_to_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if not isinstance(item, Mapping):
        raise OpenAIServingError("Each Responses input item must be an object.", param="input")
    item_type = item.get("type", "message")
    if item_type in {
        "function_call_output",
        "custom_tool_call_output",
        "local_shell_call_output",
    }:
        return f"tool: {item.get('output') or ''}"
    if item_type == "mcp_approval_response":
        decision = "approved" if item.get("approve") is True else "rejected"
        reason = item.get("reason")
        return f"tool approval: {decision}" + (f" ({reason})" if reason else "")
    if item_type != "message":
        _raise_unsupported_response_input_item(item_type)
    role = str(item.get("role") or "user")
    content = item.get("content", "")
    if isinstance(content, str):
        text = content
    elif isinstance(content, Sequence) and not isinstance(content, (bytes, bytearray)):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, Mapping):
                part_type = part.get("type", "input_text")
                if part_type not in {"input_text", "output_text", "text"}:
                    _raise_unsupported_input_content(part_type)
                text_parts.append(_response_text_part_value(part))
            else:
                raise OpenAIServingError(
                    "Responses input content parts must be strings or objects.",
                    param="input",
                )
        text = "".join(text_parts)
    else:
        raise OpenAIServingError(
            "Responses message content must be a string or text parts.",
            param="input",
        )
    return f"{role}: {text}" if role else text


def _raise_unsupported_response_input_item(item_type: Any) -> NoReturn:
    raise OpenAIServingError(
        f"Responses input item type {item_type!r} is not supported by this local server.",
        param="input",
    )


def _response_text_part_value(part: Mapping[str, Any]) -> str:
    text_value = part.get("text")
    if text_value is None:
        text_value = part.get("input_text")
    if not isinstance(text_value, str):
        raise OpenAIServingError(
            "Responses text parts must include string text.",
            param="input",
        )
    return text_value


def _raise_unsupported_input_content(part_type: Any) -> NoReturn:
    raise OpenAIServingError(
        f"Responses content part type {part_type!r} is not supported by this local server.",
        param="input",
    )


def _reject_unsupported_response_fields(body: Mapping[str, Any]) -> None:
    if body.get("background") is True:
        raise OpenAIServingError(
            "background mode is not supported by this local server.",
            param="background",
        )
    if "conversation" in body and body["conversation"] is not None:
        raise OpenAIServingError(
            "conversation is not supported by this local server; use previous_response_id.",
            param="conversation",
        )
    if "prompt" in body and body["prompt"] is not None:
        raise OpenAIServingError(
            "Prompt templates are not supported by this local server.",
            param="prompt",
        )
    if body.get("parallel_tool_calls") is False:
        raise OpenAIServingError(
            "parallel_tool_calls=false is not supported by this local server.",
            param="parallel_tool_calls",
        )
    if body.get("truncation") not in {None, "disabled"}:
        raise OpenAIServingError(
            "Automatic truncation is not supported by this local server.",
            param="truncation",
        )
    if "modalities" in body:
        modalities = body.get("modalities")
        if modalities not in (None, ["text"], ("text",)):
            raise OpenAIServingError(
                "Only text responses are supported by this local server.",
                param="modalities",
            )
    if body.get("audio") is not None:
        raise OpenAIServingError(
            "Audio output is not supported by this local server.",
            param="audio",
        )
    if body.get("top_logprobs") not in {None, 0}:
        raise OpenAIServingError(
            "Response logprobs are not supported by this local server.",
            param="top_logprobs",
        )
    if body.get("logprobs") not in {None, False}:
        raise OpenAIServingError(
            "Response logprobs are not supported by this local server.",
            param="logprobs",
        )
    _validate_response_include(body)
    _validate_response_text_config(body)
    _validate_response_metadata(body)
    for tool in body.get("tools", []) or []:
        if not isinstance(tool, Mapping):
            raise OpenAIServingError("Each tool must be an object.", param="tools")
        if _tool_type(tool) not in {"function"}:
            raise OpenAIServingError(
                f"Tool type {_tool_type(tool)!r} is not implemented by this local server.",
                param="tools",
            )


def _validate_response_include(body: Mapping[str, Any]) -> None:
    include = body.get("include")
    if include is None:
        return
    if not isinstance(include, Sequence) or isinstance(include, (str, bytes, bytearray)):
        raise OpenAIServingError("include must be a list of strings.", param="include")
    supported = {"reasoning.encrypted_content"}
    for item in include:
        if not isinstance(item, str):
            raise OpenAIServingError("include must be a list of strings.", param="include")
        if item not in supported:
            raise OpenAIServingError(
                f"include value {item!r} is not supported by this local server.",
                param="include",
            )


def _validate_response_text_config(body: Mapping[str, Any]) -> None:
    text = body.get("text")
    if text is None:
        return
    if not isinstance(text, Mapping):
        raise OpenAIServingError("text must be an object.", param="text")
    fmt = text.get("format")
    if fmt is None:
        return
    if not isinstance(fmt, Mapping):
        raise OpenAIServingError("text.format must be an object.", param="text.format")
    if fmt.get("type") != "text":
        raise OpenAIServingError(
            "Structured output formats are not supported by this local server.",
            param="text.format",
        )


def _validate_response_metadata(body: Mapping[str, Any]) -> None:
    metadata = body.get("metadata")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise OpenAIServingError("metadata must be an object.", param="metadata")


def _default_tool_choice(body: Mapping[str, Any]) -> str:
    return "auto" if body.get("tools") else "none"


def _forced_function_call_item(body: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    tools = list(body.get("tools", []) or [])
    tool_choice = body.get("tool_choice", _default_tool_choice(body))
    if not tools or tool_choice is None:
        return None
    if isinstance(tool_choice, str) and tool_choice in {"none", "auto"}:
        return None
    if tool_choice == "required":
        function_tools = [tool for tool in tools if _tool_type(tool) == "function"]
        if len(function_tools) != 1:
            raise OpenAIServingError(
                "tool_choice='required' needs exactly one local function tool.",
                param="tool_choice",
            )
        tool = function_tools[0]
    elif isinstance(tool_choice, Mapping):
        if _tool_type(tool_choice) != "function":
            raise OpenAIServingError(
                "Only function tool_choice objects are supported.",
                param="tool_choice",
            )
        name = _tool_choice_function_name(tool_choice)
        tool = _find_function_tool(tools, name)
    else:
        raise OpenAIServingError(
            "tool_choice must be 'none', 'auto', 'required', or a function tool choice.",
            param="tool_choice",
        )
    if tool is None:
        raise OpenAIServingError("No function tool matches tool_choice.", param="tool_choice")
    arguments = _empty_function_arguments(tool)
    return {
        "id": f"fc_{uuid.uuid4().hex}",
        "type": "function_call",
        "status": "completed",
        "call_id": f"call_{uuid.uuid4().hex}",
        "name": _function_tool_name(tool),
        "arguments": arguments,
    }


def _first_function_tool(tools: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    for tool in tools:
        if _tool_type(tool) == "function":
            return tool
    return None


def _find_function_tool(
    tools: Sequence[Mapping[str, Any]],
    name: str,
) -> Optional[Mapping[str, Any]]:
    for tool in tools:
        if _tool_type(tool) == "function" and _function_tool_name(tool) == name:
            return tool
    return None


def _tool_type(tool: Mapping[str, Any]) -> str:
    return str(tool.get("type") or "")


def _function_tool_name(tool: Mapping[str, Any]) -> str:
    if isinstance(tool.get("function"), Mapping):
        return str(tool["function"].get("name") or "")
    return str(tool.get("name") or "")


def _tool_choice_function_name(tool_choice: Mapping[str, Any]) -> str:
    if isinstance(tool_choice.get("function"), Mapping):
        return str(tool_choice["function"].get("name") or "")
    return str(tool_choice.get("name") or "")


def _empty_function_arguments(tool: Mapping[str, Any]) -> str:
    parameters = tool.get("parameters")
    if parameters is None and isinstance(tool.get("function"), Mapping):
        parameters = tool["function"].get("parameters")
    if parameters is None:
        return "{}"
    if not isinstance(parameters, Mapping):
        raise OpenAIServingError("function tool parameters must be an object.", param="tools")
    required = parameters.get("required", [])
    if required:
        raise OpenAIServingError(
            "This local server cannot infer required function arguments.",
            param="tools",
        )
    return "{}"


def _reasoning_item_if_requested(body: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    reasoning = body.get("reasoning")
    if not isinstance(reasoning, Mapping):
        return None
    effort = reasoning.get("effort")
    summary = reasoning.get("summary")
    if effort in {None, "none"} and summary in {None, "none"}:
        return None
    item: dict[str, Any] = {
        "id": f"rs_{uuid.uuid4().hex}",
        "type": "reasoning",
        "status": "completed",
        "summary": [],
    }
    if "reasoning.encrypted_content" in set(body.get("include", []) or []):
        item["encrypted_content"] = None
    return item


def _response_reasoning_config(body: Mapping[str, Any]) -> dict[str, Any]:
    reasoning = body.get("reasoning")
    if isinstance(reasoning, Mapping):
        return {
            "effort": reasoning.get("effort"),
            "summary": reasoning.get("summary") or reasoning.get("generate_summary"),
        }
    return {"effort": None, "summary": None}


def _response_text_config(body: Mapping[str, Any]) -> Mapping[str, Any]:
    text = body.get("text")
    if isinstance(text, Mapping):
        result = dict(text)
        result.setdefault("format", {"type": "text"})
        return result
    return {"format": {"type": "text"}}


def _apply_stop(text: str, stop: Optional[str | Sequence[str]]) -> tuple[str, bool]:
    if stop is None:
        return text, False
    stop_sequences = [stop] if isinstance(stop, str) else list(stop)
    earliest: Optional[int] = None
    for sequence in stop_sequences:
        if not isinstance(sequence, str) or sequence == "":
            continue
        index = text.find(sequence)
        if index >= 0 and (earliest is None or index < earliest):
            earliest = index
    if earliest is None:
        return text, False
    return text[:earliest], True


def _finish_reason(
    completion_ids: Sequence[int],
    *,
    max_tokens: int,
    eos_token_id: Optional[int],
    stop_hit: bool,
) -> str:
    if stop_hit:
        return "stop"
    if eos_token_id is not None and eos_token_id in completion_ids:
        return "stop"
    if len(completion_ids) >= max_tokens:
        return "length"
    return "stop"


def _stream_finish_reason_from_output(
    output: Any,
    *,
    prompt_width: int,
    max_tokens: int,
    eos_token_id: Optional[int],
) -> str:
    sequences = getattr(output, "sequences", output)
    try:
        first_sequence = sequences[0]
    except (TypeError, IndexError, KeyError):
        return "stop"
    if hasattr(first_sequence, "detach"):
        token_ids = first_sequence.detach().cpu().tolist()
    else:
        token_ids = list(first_sequence)
    completion_ids = token_ids[prompt_width:]
    return _finish_reason(
        completion_ids,
        max_tokens=max_tokens,
        eos_token_id=eos_token_id,
        stop_hit=False,
    )


def _completion_stream_chunk(
    response_id: str,
    created: int,
    model: str,
    text: str,
) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "text": text,
                "index": 0,
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    }


def _stream_backend_chunks(
    backend: CompletionBackend,
    prompt: str,
    params: Mapping[str, Any],
) -> Iterator[StreamChunk]:
    stream_generate = getattr(backend, "stream_generate", None)
    if callable(stream_generate):
        saw_finish_reason = False
        for chunk in stream_generate(
            prompt,
            max_tokens=int(params["max_tokens"]),
            temperature=params["temperature"],
            top_p=params["top_p"],
            stop=params["stop"],
            extra=params["extra"],
        ):
            normalized = _normalize_stream_chunk(chunk)
            saw_finish_reason = saw_finish_reason or normalized.finish_reason is not None
            yield normalized
        if not saw_finish_reason:
            yield StreamChunk(finish_reason="stop")
        return
    generated = backend.generate([prompt], **params)[0][0]
    for chunk in _chunk_text(generated.text):
        yield StreamChunk(chunk)
    yield StreamChunk(finish_reason=generated.finish_reason)


def _normalize_stream_chunk(chunk: StreamChunk | str) -> StreamChunk:
    if isinstance(chunk, StreamChunk):
        return chunk
    return StreamChunk(str(chunk))


def _stream_with_stop(
    chunks: Iterator[str],
    stop: Optional[str | Sequence[str]],
) -> Iterator[StreamChunk]:
    stop_sequences = _stop_sequences(stop)
    if not stop_sequences:
        for chunk in chunks:
            yield StreamChunk(chunk)
        return

    buffer = ""
    max_stop_len = max(len(sequence) for sequence in stop_sequences)
    for chunk in chunks:
        if not chunk:
            continue
        buffer += chunk
        stop_index = _first_stop_index(buffer, stop_sequences)
        if stop_index is not None:
            if stop_index > 0:
                yield StreamChunk(buffer[:stop_index])
            yield StreamChunk(finish_reason="stop")
            return
        emit_len = max(0, len(buffer) - max_stop_len + 1)
        if emit_len:
            yield StreamChunk(buffer[:emit_len])
            buffer = buffer[emit_len:]
    if buffer:
        yield StreamChunk(buffer)


def _chunk_text(text: str, chunk_size: int = 16) -> Iterator[str]:
    if not text:
        return
    for index in range(0, len(text), chunk_size):
        yield text[index : index + chunk_size]


def _stop_sequences(stop: Optional[str | Sequence[str]]) -> list[str]:
    if stop is None:
        return []
    raw = [stop] if isinstance(stop, str) else list(stop)
    return [sequence for sequence in raw if isinstance(sequence, str) and sequence]


def _first_stop_index(text: str, stop_sequences: Sequence[str]) -> Optional[int]:
    earliest: Optional[int] = None
    for sequence in stop_sequences:
        index = text.find(sequence)
        if index >= 0 and (earliest is None or index < earliest):
            earliest = index
    return earliest


def _response_object(
    *,
    response_id: str,
    model: str,
    generated: Optional[GeneratedText],
    status: str,
    instructions: Any,
    created_at: Optional[int] = None,
    output_items: Optional[list[dict[str, Any]]] = None,
    request_body: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    created = created_at or int(time.time())
    text = generated.text if generated is not None else ""
    items = list(output_items or [])
    if not items and generated is not None:
        items.append(
            _response_message_item(
                f"msg_{uuid.uuid4().hex}",
                text,
                status="completed",
            )
        )
    elif items:
        text = _output_text_from_items(items)
    usage = _empty_response_usage()
    if generated is not None:
        usage["input_tokens"] = generated.prompt_tokens
        usage["output_tokens"] = generated.completion_tokens
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    request_body = request_body or {}
    metadata = request_body.get("metadata")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise OpenAIServingError("metadata must be an object.", param="metadata")
    return {
        "id": response_id,
        "object": "response",
        "created_at": created,
        "status": status,
        "error": None,
        "incomplete_details": None,
        "instructions": instructions,
        "model": model,
        "output": items if status == "completed" else [],
        "output_text": text,
        "background": bool(request_body.get("background", False)),
        "max_output_tokens": request_body.get("max_output_tokens"),
        "max_tool_calls": request_body.get("max_tool_calls"),
        "metadata": dict(metadata or {}),
        "parallel_tool_calls": True,
        "prompt_cache_key": request_body.get("prompt_cache_key"),
        "prompt_cache_retention": request_body.get("prompt_cache_retention"),
        "previous_response_id": request_body.get("previous_response_id"),
        "reasoning": _response_reasoning_config(request_body),
        "safety_identifier": request_body.get("safety_identifier"),
        "service_tier": request_body.get("service_tier", "auto"),
        "store": request_body.get("store", True),
        "temperature": request_body.get("temperature", 1.0),
        "text": _response_text_config(request_body),
        "tool_choice": request_body.get("tool_choice", _default_tool_choice(request_body)),
        "tools": request_body.get("tools", []),
        "top_logprobs": request_body.get("top_logprobs", 0),
        "top_p": request_body.get("top_p", 1.0),
        "truncation": request_body.get("truncation", "disabled"),
        "user": request_body.get("user"),
        "usage": usage,
    }


def _response_message_item(item_id: str, text: str, *, status: str) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": text,
                "annotations": [],
                "logprobs": [],
            }
        ],
    }


def _output_text_from_items(items: Sequence[Mapping[str, Any]]) -> str:
    parts: list[str] = []
    for item in items:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if isinstance(content, Mapping) and content.get("type") == "output_text":
                parts.append(str(content.get("text") or ""))
    return "".join(parts)


def _response_route(path: str) -> tuple[Optional[str], Optional[str]]:
    prefix = "/v1/responses/"
    if not path.startswith(prefix):
        return None, None
    suffix = path[len(prefix) :]
    if not suffix:
        return None, None
    parts = suffix.strip("/").split("/")
    if len(parts) == 1:
        return parts[0], None
    if len(parts) == 2 and parts[1] == "input_items":
        return parts[0], "input_items"
    return None, None


def _group_generated(
    flat: Sequence[GeneratedText],
    *,
    batch_size: int,
    n: int,
) -> list[list[GeneratedText]]:
    return [list(flat[index : index + n]) for index in range(0, batch_size * n, n)]


def _empty_usage() -> dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _accumulate_usage(usage: dict[str, int], generated: GeneratedText) -> None:
    usage["prompt_tokens"] += generated.prompt_tokens
    usage["completion_tokens"] += generated.completion_tokens


def _empty_response_usage() -> dict[str, Any]:
    return {
        "input_tokens": 0,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 0,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 0,
    }


def _count_backend_tokens(backend: CompletionBackend, text: str) -> int:
    count_tokens = getattr(backend, "count_tokens", None)
    if not callable(count_tokens):
        return 0
    try:
        return int(count_tokens(text))
    except Exception:
        logger.debug("Backend token counting failed", exc_info=True)
        return 0


def _error_payload(
    message: str,
    *,
    error_type: str = "invalid_request_error",
    param: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": None,
        }
    }


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise OpenAIServingError(
            "CUDA was requested but torch.cuda.is_available() is false.",
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            error_type="server_error",
        )
    return device


def _resolve_dtype(dtype: str, device: str) -> Optional[torch.dtype]:
    if dtype == "auto":
        return torch.float16 if device.startswith("cuda") else None
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]


def _json_arg(raw: str, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--{name} must be a JSON object") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"--{name} must be a JSON object")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
