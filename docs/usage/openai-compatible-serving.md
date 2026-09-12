# OpenAI-Compatible Local Serving

RL-Kernel provides a small HTTP serving layer for local development and downstream
tooling that expects OpenAI-compatible JSON responses.

The server currently supports these OpenAI-compatible endpoints:

- `GET /v1/models`
- `POST /v1/completions`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `GET /v1/responses/{response_id}`
- `DELETE /v1/responses/{response_id}`
- `GET /v1/responses/{response_id}/input_items`

All generation endpoints support `stream: true`. Responses streaming uses typed
SSE events such as `response.created`, `response.in_progress`,
`response.output_text.delta`, and `response.completed`. Chat/completions and
legacy completions use data-only SSE chunks followed by `data: [DONE]`.

The Responses object follows the current OpenAI shape for text generation,
including `output`, `output_text`, `reasoning`, and `usage.input_tokens` /
`usage.output_tokens`. Request options such as `text.format` are validated
separately; this local server only accepts plain text output. The server does
not pretend to support hosted-only capabilities. Unsupported features such as
background mode, conversations, prompt templates, audio output, image or file
inputs, structured output formats, and built-in hosted tools return OpenAI-style
error payloads.

## Transformers Backend

Use the Transformers backend when you want to validate the HTTP layer on a local
GPU without installing vLLM:

```bash
python -m rl_engine.executors.openai_server \
  --backend transformers \
  --model sshleifer/tiny-gpt2 \
  --device cuda \
  --dtype float16 \
  --host 127.0.0.1 \
  --port 8000
```

Chat completion request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "sshleifer/tiny-gpt2",
    "messages": [{"role": "user", "content": "Say hello from RL-Kernel"}],
    "max_tokens": 16,
    "temperature": 0
  }'
```

Text completion request:

```bash
curl http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "sshleifer/tiny-gpt2",
    "prompt": "RL-Kernel local serving",
    "max_tokens": 16,
    "temperature": 0
}'
```

Responses API request:

```bash
curl http://127.0.0.1:8000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "sshleifer/tiny-gpt2",
    "input": "Say hello from the Responses API",
    "max_output_tokens": 16,
    "temperature": 0
  }'
```

Responses API streaming:

```bash
curl -N http://127.0.0.1:8000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "sshleifer/tiny-gpt2",
    "input": "Stream from RL-Kernel",
    "max_output_tokens": 16,
    "temperature": 0,
    "stream": true
  }'
```

Chat completion streaming:

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "sshleifer/tiny-gpt2",
    "messages": [{"role": "user", "content": "Stream a short reply"}],
    "max_tokens": 16,
    "temperature": 0,
    "stream": true
  }'
```

## Rollout Backend

Use the rollout backend to route requests through `RolloutExecutor`, which
constructs the vLLM sampler lazily:

```bash
python -m rl_engine.executors.openai_server \
  --backend rollout \
  --model /path/to/model \
  --engine-kwargs '{"dtype": "float16"}' \
  --sampling-params '{"top_p": 0.9}' \
  --host 127.0.0.1 \
  --port 8000
```

Install the vLLM extra before using this backend:

```bash
pip install -e ".[vllm]"
```

## Local GPU Smoke Test

The real GPU smoke test is opt-in because it downloads or loads a model:

```bash
RL_KERNEL_RUN_LOCAL_SERVE_SMOKE=1 \
RL_KERNEL_SERVE_SMOKE_MODEL=sshleifer/tiny-gpt2 \
python -m pytest tests/test_openai_server.py::test_transformers_backend_cuda_http_smoke -v
```

For the Transformers backend, streaming uses Hugging Face's
`TextIteratorStreamer`, so tokens are emitted incrementally from the local model
instead of waiting for the full completion.

## Tool Calls And Reasoning Items

The local server implements the Responses API shape for function tools:

- `tools` entries with `type: "function"`
- `tool_choice: "none" | "auto" | "required"`
- forced tool choices such as `{"type": "function", "name": "lookup_metric"}`
- output items with `type: "function_call"`
- streaming events `response.function_call_arguments.delta` and
  `response.function_call_arguments.done`

Example forced function call:

```bash
curl http://127.0.0.1:8000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "sshleifer/tiny-gpt2",
    "input": "Look up the metric",
    "tools": [{"type": "function", "name": "lookup_metric"}],
    "tool_choice": {"type": "function", "name": "lookup_metric"}
  }'
```

For `tool_choice: "auto"`, the local Transformers backend generates text. It does
not perform OpenAI-hosted tool planning. For `tool_choice: "required"`, the local
server only emits a function call when there is exactly one function tool and
that tool has no required parameters; otherwise it returns an explicit error
instead of inventing arguments. Built-in hosted tools such as web search, file
search, and computer use return an explicit unsupported error.

Reasoning requests return a standard `type: "reasoning"` output item with an empty
summary. RL-Kernel does not expose raw chain-of-thought. If
`include: ["reasoning.encrypted_content"]` is set, the field is present as `null`
because the local backend has no encrypted reasoning token stream.

## Response State

By default, generated Responses are stored in memory for the lifetime of the
server process. You can retrieve or delete them:

```bash
curl http://127.0.0.1:8000/v1/responses/resp_...
curl -X DELETE http://127.0.0.1:8000/v1/responses/resp_...
curl http://127.0.0.1:8000/v1/responses/resp_.../input_items
```

`previous_response_id` is supported against this in-memory store. Set
`"store": false` to skip local storage for a response.
