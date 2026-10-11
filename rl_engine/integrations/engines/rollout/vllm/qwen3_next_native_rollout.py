# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Actual vLLM scheduler rollout, with raw shared-provider L2 observations.

Qualification leaves backend capabilities closed; certified execution requires
the source/environment-bound profile certificate. The scheduler audits actual decisions;
configuration hints alone are never reported as executed prefill chunks.
"""

import gc
import hashlib
import os
from pathlib import Path

import torch
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

from rl_engine.integrations.common.qwen3_next_binary import (
    imported_provider_identity,
    validate_provider_identity,
)
from rl_engine.models.qwen3_next import qwen3_next_subnode as subnode
from rl_engine.validation.common.tensor_identity import assert_tensor_bitwise_equal as exact
from rl_engine.validation.models.qwen3_next_l2 import (
    embedding_digest,
    json_digest,
    layer_digest,
    load_tensor_artifact,
    save_tensor_artifact,
    validate_debug_capture,
    validate_model_artifact,
)

NATIVE_ROLLOUT_PROFILE = {
    # Caps, not schedules: ExactPrefillScheduler fixes every step's chunks. They
    # are sized so up to 64 concurrent requests fit one certified profile.
    "kv_cache_memory_bytes_per_gpu": 4 << 30,
    "max_model_len": 1040,
    "max_num_seqs": 64,
    "max_num_batched_tokens": 16384,
    "scheduler": (
        "rl_engine.integrations.engines.rollout.vllm.qwen3_next_native_rollout"
        ".ExactPrefillScheduler"
    ),
    "async_scheduling": False,
    "prefix_caching": False,
    "weights_dtype": "bfloat16",
    "recurrent_state_dtype": "float32",
    "model_runner": "v1",
}


def capabilities():
    return {
        "rollout": True,
        "debug_layers": True,
        "architecture_layers": 48,
        "tp": 4,
        "batched_cases": True,
        "native_scheduler": True,
        "scheduler_batch_invariance_verified": False,
        "full_model_l2_verified": False,
    }


def runtime_profile(config):
    scheduler = config.scheduler_config.scheduler_cls
    if isinstance(scheduler, type):
        scheduler = scheduler.__module__ + "." + scheduler.__qualname__
    actual = {
        "kv_cache_memory_bytes_per_gpu": config.cache_config.kv_cache_memory_bytes,
        "max_model_len": config.model_config.max_model_len,
        "max_num_seqs": config.scheduler_config.max_num_seqs,
        "max_num_batched_tokens": config.scheduler_config.max_num_batched_tokens,
        "scheduler": scheduler,
        "async_scheduling": config.scheduler_config.async_scheduling,
        "prefix_caching": config.cache_config.enable_prefix_caching,
        "weights_dtype": str(config.model_config.dtype).removeprefix("torch."),
        "recurrent_state_dtype": config.cache_config.mamba_ssm_cache_dtype,
        "model_runner": "v2" if config.use_v2_model_runner else "v1",
    }
    if actual != NATIVE_ROLLOUT_PROFILE:
        raise ValueError(f"Native runtime changed the explicit rollout profile: {actual}")
    return actual


def validate_request(request):
    if request.get("role") != "rollout" or request.get("schema_version") not in (1, 2):
        raise ValueError("A version 1 (L2) or 2 (batch-invariance) rollout request is required")
    # Version 2 is one traced target next to untraced companions.
    bi = request["schema_version"] == 2
    digest_input = {key: value for key, value in request.items() if key != "request_sha256"}
    if json_digest(digest_input) != request.get("request_sha256"):
        raise ValueError("Rollout request identity mismatch")
    prompts, samples = request["prompt_tokens"], request["sample_ids"]
    batch = len(prompts)
    if (
        batch not in ((1,) if bi else (1, 4))
        or len(samples) != batch
        or any(type(sample) is not int for sample in samples)
        or len(set(samples)) != batch
    ):
        raise ValueError("Rollout requires one or four uniquely identified samples")
    length = len(prompts[0])
    lengths = range(1, 1025) if bi else (8, 64, 256, 1024)
    if length not in lengths or any(len(row) != length for row in prompts):
        raise ValueError("Rollout prompts must use one official matrix length")
    responses = request["response_tokens"]
    for rows in (prompts, responses or []):
        if any(type(token) is not int or not 0 <= token < 151936 for row in rows for token in row):
            raise ValueError("Rollout tokens must be integers in the official vocabulary")
    if responses is not None and (
        len(responses) != batch or any(len(row) != 16 for row in responses)
    ):
        raise ValueError("Frozen responses must have exactly sixteen tokens per sample")
    chunks = request["prefill_chunks"]
    if (
        not chunks
        or any(type(value) is not int or value <= 0 for value in chunks)
        or sum(chunks) != length
    ):
        raise ValueError("Exact positive prefill chunks must cover the prompt")
    expected_execution = {
        "rollout_mode": "autoregressive",
        "prefill_chunks": chunks,
        "response_mode": "generate" if responses is None else "teacher_forced",
    }
    if (
        request["execution"] != expected_execution
        or request["response_length"] != 16
        or request["vocab_size"] != 151936
        or request["logits_dtype"] != "bfloat16"
        or request["logprobs_dtype"] != "float32"
        or request["prediction_positions"] != list(range(length - 1, length + 15))
        or request["consumed_response_tokens"] != 15
        or request["state_position"] != length + 14
        or request["required_layer_digests"] != list(range(48))
        or request["generation"]
        != {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "seed": 1234, "ignore_eos": True}
    ):
        raise ValueError("Rollout execution or response offsets differ from the pinned contract")
    if responses is None:
        if request["consumed_input_tokens"] is not None or (
            not bi and (batch != 4 or chunks != [length])
        ):
            raise ValueError("Only baseline B4 full prefill may define frozen responses")
    elif request["consumed_input_tokens"] != [
        prompt + response[:-1] for prompt, response in zip(prompts, responses, strict=True)
    ]:
        raise ValueError("Consumed input must end before response token sixteen")
    if any(type(layer) is not int or layer not in range(48) for layer in request["trace_layers"]):
        raise ValueError("Debug layer indices must be in [0,48)")
    if bi:
        validate_companions(request, length)
    elif "companions" in request:
        raise ValueError("Only a version 2 request may run companions")
    identity = request["identity"]
    if identity["architecture_layers"] != 48 or identity["topology"] != {
        "tp": 4,
        "pp": 1,
        "cp": 1,
        "ep": 1,
    }:
        raise ValueError("Rollout cannot change depth or topology")
    if not Path(identity["checkpoint"]).is_absolute():
        raise ValueError("Checkpoint identity must use an absolute shared path")
    if (
        identity["provider_config"].get("native_rollout_kv_cache_memory_bytes")
        != NATIVE_ROLLOUT_PROFILE["kv_cache_memory_bytes_per_gpu"]
    ):
        raise ValueError("Provider identity must pin the native rollout cache budget")
    return batch, length


def validate_companions(request, length):
    companions = request["companions"]
    tokens, arrivals = request["companion_max_tokens"], request["arrival_steps"]
    max_len = NATIVE_ROLLOUT_PROFILE["max_model_len"]
    if (
        not isinstance(companions, list)
        or len(companions) > NATIVE_ROLLOUT_PROFILE["max_num_seqs"] - 1
        or len(tokens) != len(companions)
        or len(arrivals.get("companions", [])) != len(companions)
        or type(arrivals.get("target")) is not int
        or arrivals["target"] < 0
        or any(type(step) is not int or step < 0 for step in arrivals["companions"])
        or type(request["chunk_size"]) is not int
        or request["chunk_size"] < 1
        or length + RESPONSE_LENGTH > max_len
    ):
        raise ValueError("Batch-invariance companions, arrivals or chunk size are malformed")
    for prompt, count in zip(companions, tokens, strict=True):
        if (
            not prompt
            or type(count) is not int
            or count < 1
            or len(prompt) + count > max_len
            or any(type(token) is not int or not 0 <= token < 151936 for token in prompt)
        ):
            raise ValueError("A companion does not fit the pinned rollout profile")


RESPONSE_LENGTH = 16


def next_scheduled_tokens(computed, prompt_length, chunks):
    boundary = 0
    for size in chunks:
        if computed == boundary:
            return size
        boundary += size
    if prompt_length <= computed < prompt_length + 15:
        return 1
    raise ValueError("Native scheduler is not at an exact requested chunk/decode boundary")


class ExactPrefillScheduler(Scheduler):
    """Use native cache allocation/scheduling while enforcing one frozen case at a time."""

    def schedule(self, throttle_prefills=False):
        active = [request for request in self.requests.values() if not request.is_finished()]
        if not active:
            return super().schedule(throttle_prefills)
        if any(
            (request.sampling_params.extra_args or {}).get("shared_bi_companion") is not None
            or ((request.sampling_params.extra_args or {}).get("shared_l2") or {}).get("bi")
            for request in active
        ):
            return self._schedule_batch_invariance(active, throttle_prefills)
        descriptors = [
            (request.sampling_params.extra_args or {}).get("shared_l2") for request in active
        ]
        if any(item is None for item in descriptors):
            raise ValueError("Exact Qwen scheduler accepts only identified L2 requests")
        first = descriptors[0]
        contract = (
            first["request_sha256"],
            first["batch_size"],
            first["prompt_length"],
            first["prefill_chunks"],
        )
        if any(
            (
                item["request_sha256"],
                item["batch_size"],
                item["prompt_length"],
                item["prefill_chunks"],
            )
            != contract
            for item in descriptors
        ):
            raise ValueError("Different L2 cases cannot share a scheduler batch")
        if len(active) < first["batch_size"]:
            # EngineCore may receive the four add-request messages separately.
            # Its normal paused scheduler emits no work until all are admitted.
            previous = self._pause_state
            self._pause_state = PauseState.PAUSED_ALL
            try:
                result = super().schedule(throttle_prefills)
            finally:
                self._pause_state = previous
            if result.total_num_scheduled_tokens:
                raise RuntimeError("The native scheduler violated the input batch barrier")
            return result
        if len(active) != first["batch_size"] or len(
            {item["sample_id"] for item in descriptors}
        ) != len(active):
            raise ValueError("L2 scheduler sample cardinality mismatch")
        counts = {request.num_computed_tokens for request in active}
        if len(counts) != 1:
            raise RuntimeError("Native requests lost their synchronized exact-chunk boundary")
        computed = counts.pop()
        count = next_scheduled_tokens(computed, first["prompt_length"], first["prefill_chunks"])
        self.scheduler_config.long_prefill_token_threshold = count
        result = super().schedule(throttle_prefills)
        expected = {request.request_id: count for request in active}
        if result.num_scheduled_tokens != expected or result.preempted_req_ids:
            raise RuntimeError(
                "exact_scheduler_mismatch: "
                f"expected {expected}, actual {result.num_scheduled_tokens}"
            )
        return result

    def _schedule_batch_invariance(self, active, throttle_prefills):
        """One traced target keeps its exact chunk plan; companions fill the step.

        All requests of a cell are admitted before the first step (as for L2).
        A request joins the waiting queue only at its ``arrival_step``, counted
        in scheduled steps from the first one, so arrivals are reproducible
        whatever the engine-core timing. Every prefilling request shares one
        ``long_prefill_token_threshold``: the target's next planned chunk while
        it prefills, the cell's chunk size otherwise. The target must get
        exactly its planned token count at every step until it has consumed its
        last input, and no request may be preempted.
        """
        descriptors = {}
        for request in self.requests.values():
            extra = request.sampling_params.extra_args or {}
            descriptor = extra.get("shared_l2") or extra.get("shared_bi_companion")
            if descriptor is None:
                raise ValueError("Exact Qwen scheduler accepts only identified requests")
            descriptors[request.request_id] = (request, descriptor, "shared_l2" in extra)
        cells = {item[1]["request_sha256"] for item in descriptors.values()}
        targets = [item for item in descriptors.values() if item[2]]
        if len(cells) != 1 or len(targets) > 1:
            raise ValueError("One batch-invariance cell runs exactly one traced target")
        # Once the target has finished, the scheduler frees it; companions go on.
        target, contract, _ = (
            targets[0] if targets else (None, *next(iter(descriptors.values()))[1:])
        )
        cell = contract["request_sha256"]
        if getattr(self, "_bi_cell", None) != cell:
            self._bi_cell, self._bi_step, self._bi_started = cell, 0, False
        if not self._bi_started and len(descriptors) < contract["total_requests"]:
            previous = self._pause_state
            self._pause_state = PauseState.PAUSED_ALL
            try:
                result = super().schedule(throttle_prefills)
            finally:
                self._pause_state = previous
            if result.total_num_scheduled_tokens:
                raise RuntimeError("The native scheduler violated the cell admission barrier")
            return result
        self._bi_started = True
        held = [
            r for r in self.waiting if descriptors[r.request_id][1]["arrival_step"] > self._bi_step
        ]
        if len(held) == len(self.waiting) and not self.running and held:
            # Nothing can run before the next arrival: that arrival is now.
            self._bi_step = min(descriptors[r.request_id][1]["arrival_step"] for r in held)
            held = [r for r in held if descriptors[r.request_id][1]["arrival_step"] > self._bi_step]
        self.waiting.remove_requests(held)
        target_active = target is not None and not target.is_finished() and target not in held
        count = None
        if target_active:
            count = next_scheduled_tokens(
                target.num_computed_tokens, contract["prompt_length"], contract["prefill_chunks"]
            )
        prefilling = target_active and target.num_computed_tokens < contract["prompt_length"]
        self.scheduler_config.long_prefill_token_threshold = (
            count if prefilling else contract["chunk_size"]
        )
        try:
            result = super().schedule(throttle_prefills)
        finally:
            for request in held:
                self.waiting.add_request(request)
        if result.total_num_scheduled_tokens:
            self._bi_step += 1
        if result.preempted_req_ids or (
            count is not None and result.num_scheduled_tokens.get(target.request_id) != count
        ):
            raise RuntimeError(
                "exact_scheduler_mismatch: target expected "
                f"{count}, actual {result.num_scheduled_tokens.get(target.request_id)}, "
                f"preempted {sorted(result.preempted_req_ids)}"
            )
        return result


class ForceFrozenTokens(AdapterLogitsProcessor):
    """Sampling-only teacher forcing; raw model logits are observed beforehand."""

    def is_argmax_invariant(self):
        return False

    def new_req_logits_processor(self, params):
        if (params.extra_args or {}).get("shared_bi_companion") is not None:
            return None  # companions generate greedily and are never traced
        descriptor = (params.extra_args or {}).get("shared_l2")
        if descriptor is None:
            raise ValueError("L2 sampling requires explicit request metadata")
        frozen = descriptor["response_tokens"]
        if frozen is None:
            return None
        if len(frozen) != 16 or any(
            type(token) is not int or not 0 <= token < 151936 for token in frozen
        ):
            raise ValueError("Teacher forcing requires sixteen frozen official token IDs")

        def force(output_ids, logits):
            index = len(output_ids)
            if index not in range(16):
                raise ValueError("Teacher forcing exceeded the fixed response length")
            selected = torch.full_like(logits, float("-inf"))
            selected[frozen[index]] = 0
            return selected

        return force


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RolloutTrace:
    """Worker-local observations of real engine buffers, keyed by immutable sample IDs."""

    def __init__(self, worker, request, output_directory):
        from rl_engine.integrations.engines.rollout.vllm.qwen3_next_vllm_model import (
            RLKernelQwen3NextForCausalLM,
        )

        self.batch, self.prompt_length = validate_request(request)
        self.worker, self.request, self.directory = worker, request, Path(output_directory)
        self.provider_binary = imported_provider_identity()
        if self.provider_binary != worker.shared_provider_binary:
            raise RuntimeError("Provider binary changed after native worker initialization")
        self.rank = worker.rank
        self.native_runtime_profile = runtime_profile(worker.vllm_config)
        self.provider_execution_mode = worker.shared_provider_execution_mode
        expected_mode = (
            "certified"
            if request["identity"].get("scheduler_profile_certificate")
            else "qualification"
        )
        if self.provider_execution_mode != expected_mode:
            raise ValueError("Worker provider certification differs from the immutable request")
        self.model = worker.get_model()
        if not isinstance(self.model, RLKernelQwen3NextForCausalLM) or worker.use_v2_model_runner:
            raise ValueError(
                "Trace requires the actual shared model and native V1 GPU model runner"
            )
        self.row_by_sample = {sample: row for row, sample in enumerate(request["sample_ids"])}
        self.request_rows, self.progress = {}, {}
        # Batch-invariance requests run untraced companions next to the traced
        # target; their tokens share every step but are never recorded.
        self.companions = request.get("companions")
        self.untraced = set()
        self.current, self.current_index, self.packed, self.schedule = [], [], [], []
        self.logits = torch.empty(self.batch, 16, 151936, dtype=torch.bfloat16)
        self.local_logits = torch.empty(self.batch, 16, 37984, dtype=torch.bfloat16)
        self.logprobs = torch.empty(self.batch, 16, dtype=torch.float32)
        self.responses = torch.full((self.batch, 16), -1, dtype=torch.int64)
        self.hidden = torch.empty(48, self.batch, 16, 2048, dtype=torch.bfloat16)
        self.hidden_written = torch.zeros(48, self.batch, 16, dtype=torch.bool)
        self.states = [
            (
                {"convolution": [None] * self.batch, "recurrent_state": [None] * self.batch}
                if (layer + 1) % 4
                else {"key": [None] * self.batch, "value": [None] * self.batch}
            )
            for layer in range(48)
        ]
        self.model.model.shared_forward_hook = self.begin_forward
        self.model.model.shared_trace_hook = self.layer
        self.model.shared_logits_hook = self.record_logits
        self.subnodes = subnode.SubnodeBuffer(self.batch)
        subnode.install(self.model.model.layers, subnode.vllm_modules, self.subnode)
        self.model.model.subnode_hook = self.embedding

    def embedding(self, name, packed):
        for row, begin, end, computed in self.current:
            for offset in range(end - begin):
                response = computed + offset - (self.prompt_length - 1)
                if 0 <= response < 16:
                    self.subnodes.put_embedding(row, response, packed[begin + offset])

    def subnode(self, layer_index, name, packed):
        for row, begin, end, computed in self.current:
            for offset in range(end - begin):
                response = computed + offset - (self.prompt_length - 1)
                if 0 <= response < 16:
                    self.subnodes.put(layer_index, name, row, response, packed[begin + offset])

    def before_execute(self, scheduled):
        for new in scheduled.scheduled_new_reqs:
            extra = new.sampling_params.extra_args or {}
            companion = extra.get("shared_bi_companion")
            if companion is not None:
                if (
                    self.companions is None
                    or companion.get("request_sha256") != self.request["request_sha256"]
                    or new.prompt_token_ids != self.companions[companion["index"]]
                    or new.num_computed_tokens != 0
                    or new.req_id in self.untraced
                ):
                    raise ValueError("Untraced companion does not belong to this request")
                self.untraced.add(new.req_id)
                self.progress[new.req_id] = 0
                continue
            descriptor = extra.get("shared_l2", {})
            if descriptor.get("request_sha256") != self.request["request_sha256"]:
                raise ValueError("Scheduler request belongs to a different L2 case")
            row = self.row_by_sample[descriptor["sample_id"]]
            if (
                new.prompt_token_ids != self.request["prompt_tokens"][row]
                or new.num_computed_tokens != 0
            ):
                raise ValueError("Native prompt differs or prefix reuse was applied")
            if new.req_id in self.request_rows or row in self.request_rows.values():
                raise ValueError("Native scheduler duplicated a sample")
            self.request_rows[new.req_id], self.progress[new.req_id] = row, 0
        self.scheduled = scheduled
        if not scheduled.total_num_scheduled_tokens:
            return
        scheduled_ids = set(scheduled.num_scheduled_tokens)
        traced = scheduled_ids & set(self.request_rows)
        unfinished = {
            req_id
            for req_id in self.request_rows
            if self.progress[req_id] < self.prompt_length + 15
        }
        # A batch-invariance target may wait for its arrival step; once it has
        # started it must take part in every step until its last input.
        started = {req_id for req_id in unfinished if self.progress[req_id] > 0}
        if (
            scheduled.preempted_req_ids
            or scheduled_ids - traced - self.untraced
            or (self.companions is None and len(traced) != self.batch)
            or (self.companions is not None and not started <= traced <= unfinished)
        ):
            raise RuntimeError("The actual native step did not preserve its required batch")
        step = {}
        for req_id, count in scheduled.num_scheduled_tokens.items():
            computed = self.progress[req_id]
            if req_id in self.untraced:
                step[req_id] = (None, computed, count)
                continue
            expected = next_scheduled_tokens(
                computed, self.prompt_length, self.request["prefill_chunks"]
            )
            if count != expected:
                raise RuntimeError(
                    "Observed worker tokens disagree with requested prefill/decode chunks"
                )
            step[req_id] = (self.request_rows[req_id], computed, count)
        self.step = step

    def begin_forward(self, input_ids, positions):
        ids = list(self.worker.model_runner.input_batch.req_ids)
        if set(ids) != set(self.step):
            raise ValueError("Actual model input batch differs from SchedulerOutput")
        self.current, self.current_index, self.packed = [], [], []
        begin = 0
        for index, req_id in enumerate(ids):
            row, computed, count = self.step[req_id]
            end = begin + count
            if positions[begin:end].cpu().tolist() != list(range(computed, computed + count)):
                raise ValueError("Native absolute positions differ from the scheduled offsets")
            self.packed.append((begin, end, computed))
            if row is None:
                self.progress[req_id] = computed + count
                begin = end
                continue
            known = self.request["prompt_tokens"][row] + self.responses[row].tolist()
            if input_ids[begin:end].cpu().tolist() != known[computed : computed + count]:
                raise ValueError(
                    "Native engine consumed tokens other than the prompt/frozen response prefix"
                )
            self.current.append((row, begin, end, computed))
            self.current_index.append(index)
            self.progress[req_id] = computed + count
            begin = end
        if begin != self.scheduled.total_num_scheduled_tokens:
            raise ValueError("Packed model input does not cover actual scheduled tokens")
        if not self.current:
            return
        self.schedule.append(
            {
                "samples": [self.request["sample_ids"][row] for row, _, _, _ in self.current],
                "computed_before": [computed for _, _, _, computed in self.current],
                "token_counts": [end - begin for _, begin, end, _ in self.current],
            }
        )

    def layer(self, layer_index, output, layer):
        from vllm.forward_context import get_forward_context
        from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

        from rl_engine.integrations.engines.rollout.vllm.qwen3_next_vllm import metadata_sequences
        from rl_engine.integrations.engines.rollout.vllm.qwen3_next_vllm_attention import (
            SharedAttentionMetadata,
        )

        if output.dtype != torch.bfloat16 or output.shape[1:] != (2048,):
            raise ValueError("Layer observation did not preserve original BF16 hidden states")
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict):
            raise ValueError("Native engine did not supply per-layer attention metadata")
        for row, begin, end, computed in self.current:
            for offset in range(end - begin):
                response = computed + offset - (self.prompt_length - 1)
                if 0 <= response < 16:
                    if self.hidden_written[layer_index, row, response]:
                        raise ValueError("A response hidden state was observed more than once")
                    self.hidden[layer_index, row, response].copy_(
                        output[begin + offset].detach().cpu()
                    )
                    self.hidden_written[layer_index, row, response] = True
        if not any(
            computed + end - begin == self.prompt_length + 15
            for _, begin, end, computed in self.current
        ):
            return
        if (layer_index + 1) % 4:
            module = layer.linear_attn
            meta = metadata[module.prefix]
            indices, cu, _reset = metadata_sequences(
                meta, self.scheduled.total_num_scheduled_tokens
            )
            if cu.tolist() != [0] + [end for _, end, _ in self.packed]:
                raise ValueError("GDN metadata reordered sequences without matching model inputs")
            conv, recurrent = module.kv_cache
            if not is_conv_state_dim_first():
                conv = conv.transpose(-1, -2)
            slots = indices.tolist()
            if len(slots) != len(self.packed):
                raise ValueError("GDN metadata does not cover every packed sequence")
            for (row, begin, end, computed), index in zip(self.current, self.current_index):
                slot = slots[index]
                if computed + end - begin == self.prompt_length + 15:
                    self.states[layer_index]["convolution"][row] = conv[slot].detach().cpu().clone()
                    self.states[layer_index]["recurrent_state"][row] = (
                        recurrent[slot].detach().cpu().clone()
                    )
        else:
            attention = layer.self_attn.attn
            meta, cache = metadata[attention.layer_name], attention.kv_cache
            if not isinstance(meta, SharedAttentionMetadata):
                raise ValueError("Full attention did not use shared native paged-KV metadata")
            if list(meta.query_start_loc) != [0] + [end for _, end, _ in self.packed]:
                raise ValueError("Full attention boundaries differ from actual packed input")
            block_size = cache.shape[2]
            for index, (row, begin, end, computed) in zip(self.current_index, self.current):
                length = computed + end - begin
                if length != self.prompt_length + 15:
                    continue
                if meta.seq_lens[index] != length:
                    raise ValueError("Full attention prefix length differs from consumed inputs")
                blocks = meta.block_table[index, : (length + block_size - 1) // block_size].long()
                values = cache[blocks, 0].reshape(-1, 512)[:length]
                self.states[layer_index]["key"][row] = (
                    values[:, :256].reshape(1, length, 256).detach().cpu().clone()
                )
                self.states[layer_index]["value"][row] = (
                    values[:, 256:].reshape(1, length, 256).detach().cpu().clone()
                )

    def record_logits(self, hidden, local, full):
        from rl_engine.contracts.operators.logprob import LogprobRole
        from rl_engine.models.qwen3_next.qwen3_next_forward import shared_logprob
        from rl_engine.models.qwen3_next.qwen3_next_logprob import qwen_logprob_contract

        if full.shape != (len(self.packed), 151936) or local.shape != (len(self.packed), 37984):
            raise ValueError(
                "Native sampling rows do not correspond one-to-one with active requests"
            )
        eligible = [
            (index, row, computed + end - begin - self.prompt_length)
            for index, (row, begin, end, computed) in zip(self.current_index, self.current)
            if computed + end - begin >= self.prompt_length
        ]
        if not eligible:
            return
        rows = torch.tensor([index for index, _, _ in eligible], device=local.device)
        targets = torch.tensor(
            [
                (
                    int(full[index].argmax())
                    if self.request["response_tokens"] is None
                    else self.request["response_tokens"][row][position]
                )
                for index, row, position in eligible
            ],
            device=local.device,
            dtype=torch.int64,
        )
        selected = local.index_select(0, rows)
        contract = qwen_logprob_contract(
            selected, rank=self.rank, world_size=4, role=LogprobRole.INFER
        )
        logp, _, _ = shared_logprob(
            selected, targets, contract=contract, tp_group=self.model.shared_tp_group
        )
        for count, (index, row, position) in enumerate(eligible):
            if position not in range(16) or self.responses[row, position] != -1:
                raise ValueError("Sampling observation has a duplicate or invalid response offset")
            self.logits[row, position].copy_(full[index].detach().cpu())
            self.local_logits[row, position].copy_(local[index].detach().cpu())
            self.logprobs[row, position].copy_(logp[count].detach().cpu())
            self.responses[row, position] = int(targets[count])

    def finish(self):
        if imported_provider_identity() != self.provider_binary:
            raise RuntimeError("Provider binary changed during the rollout case")
        if not self.hidden_written.all() or bool((self.responses < 0).any()):
            raise RuntimeError("Rollout did not observe every response position in every layer")
        for layer in range(48):
            self.subnodes.layer(layer)
        if len(self.schedule) != len(self.request["prefill_chunks"]) + 15:
            raise RuntimeError("Actual scheduler step count differs from the exact chunk contract")
        states = []
        for layer in self.states:
            if any(value is None for values in layer.values() for value in values):
                raise RuntimeError("A final convolution/recurrent/KV state was not observed")
            states.append({name: torch.stack(values) for name, values in layer.items()})
        payload = {
            "rank": self.rank,
            "request_sha256": self.request["request_sha256"],
            "logits": self.logits,
            "local_logits": self.local_logits,
            "logprobs": self.logprobs,
            "responses": self.responses,
            "hidden": self.hidden,
            "states": states,
            "subnodes": {name: self.subnodes.values[name] for name in subnode.SUBNODES},
            "embedding": self.subnodes.embedding_rows(),
            "schedule": self.schedule,
            "native_runtime_profile": self.native_runtime_profile,
            "provider_execution_mode": self.provider_execution_mode,
            "provider_binary": self.provider_binary,
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"rank-{self.rank}.pt"
        save_tensor_artifact(path, payload)
        self.model.model.shared_forward_hook = self.model.model.shared_trace_hook = None
        self.model.shared_logits_hook = None
        subnode.remove(self.model.model.layers, subnode.vllm_modules)
        self.model.model.subnode_hook = None
        return {
            "rank": self.rank,
            "path": str(path),
            "sha256": file_digest(path),
            "request_sha256": self.request["request_sha256"],
        }


def assemble_trace(request, records, sampled_responses):
    if len(records) != 4 or sorted(record["rank"] for record in records) != list(range(4)):
        raise ValueError("Rollout trace requires all four unique TP ranks")
    shards = []
    for record in sorted(records, key=lambda value: value["rank"]):
        if (
            record["request_sha256"] != request["request_sha256"]
            or file_digest(record["path"]) != record["sha256"]
        ):
            raise ValueError("Worker trace identity/hash mismatch")
        shard = load_tensor_artifact(record["path"])
        if shard["rank"] != record["rank"] or shard["request_sha256"] != request["request_sha256"]:
            raise ValueError("Worker trace content identity mismatch")
        shards.append(shard)
    first = shards[0]
    provider_binary = validate_provider_identity(first.get("provider_binary"))
    for shard in shards[1:]:
        if validate_provider_identity(shard.get("provider_binary")) != provider_binary:
            raise ValueError("TP workers imported different provider binaries")
    for shard in shards[1:]:
        for name in ("logits", "logprobs", "responses", "hidden"):
            exact(shard[name], first[name], name=f"TP replicated rollout {name}")
        for name in subnode.SUBNODES:
            exact(shard["subnodes"][name], first["subnodes"][name], name=f"TP replicated {name}")
        exact(shard["embedding"], first["embedding"], name="TP replicated embedding")
        if shard["schedule"] != first["schedule"]:
            raise ValueError("TP workers observed different scheduler decisions")
        if shard["native_runtime_profile"] != first["native_runtime_profile"]:
            raise ValueError("TP workers use different native runtime profiles")
        if shard["provider_execution_mode"] != first["provider_execution_mode"]:
            raise ValueError("TP workers use different provider certification modes")
    if first["native_runtime_profile"] != NATIVE_ROLLOUT_PROFILE:
        raise ValueError("Worker artifacts do not identify the pinned native runtime")
    expected_mode = (
        "certified" if request["identity"].get("scheduler_profile_certificate") else "qualification"
    )
    if first["provider_execution_mode"] != expected_mode:
        raise ValueError("Worker artifacts do not match the requested certification mode")
    exact(
        torch.cat([shard["local_logits"] for shard in shards], dim=-1),
        first["logits"],
        name="raw LM-head vocabulary gather",
    )
    exact(
        torch.tensor(sampled_responses, dtype=torch.int64),
        first["responses"],
        name="actual native sampled response",
    )
    layers, debug = [], {}
    for layer in range(48):
        states = {
            f"tp{rank}.{name}": value
            for rank, shard in enumerate(shards)
            for name, value in shard["states"][layer].items()
        }
        hidden = first["hidden"][layer]
        nodes = {name: first["subnodes"][name][layer] for name in subnode.SUBNODES}
        layers.append(layer_digest(layer, hidden, states, request=request, subnodes=nodes))
        if layer in request["trace_layers"]:
            debug[f"layer.{layer}.output"] = hidden
            debug.update({f"layer.{layer}.subnode.{name}": value for name, value in nodes.items()})
            debug.update({f"layer.{layer}.{name}": value for name, value in states.items()})
    artifact = {
        "metadata": {
            key: request[key]
            for key in ("schema_version", "case_id", "role", "identity", "request_sha256")
        },
        "execution": request["execution"],
        "prompt_tokens": torch.tensor(request["prompt_tokens"], dtype=torch.int64),
        "response_tokens": first["responses"],
        "prediction_positions": torch.tensor(request["prediction_positions"], dtype=torch.int64),
        "logits": first["logits"],
        "logprobs": first["logprobs"],
        "layer_digests": layers,
        "embedding_digest": embedding_digest(first["embedding"]),
        "debug_tensors": debug,
        "native_scheduler_steps": first["schedule"],
        "worker_traces": records,
        "native_runtime_profile": first["native_runtime_profile"],
        "provider_execution_mode": first["provider_execution_mode"],
        "provider_binary": provider_binary,
    }
    validate_model_artifact(artifact, request)
    validate_debug_capture(artifact, request)
    return artifact


def _new_engine(identity):
    from vllm import LLM

    from rl_engine.integrations.engines.rollout.vllm.qwen3_next_vllm_model import (
        register_shared_model,
    )
    from rl_engine.integrations.orchestrators.vime.experiments.qwen3_next.profile import (
        activate_for_identity,
    )

    activate_for_identity(identity)
    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") != "0":
        raise RuntimeError(
            "Set VLLM_USE_V2_MODEL_RUNNER=0 before importing the native rollout worker"
        )
    if torch.cuda.device_count() != 4:
        raise RuntimeError("Native rollout requires exactly four visible GPUs")
    return LLM(**engine_arguments(identity, register_shared_model()))


def engine_arguments(identity, architecture):
    """Keep explicit engine arguments independently checkable against pinned vLLM."""
    return dict(
        model=identity["checkpoint"],
        tokenizer=identity["checkpoint"],
        skip_tokenizer_init=True,
        trust_remote_code=False,
        hf_overrides={"architectures": [architecture]},
        additional_config={"qwen3_next_shared_identity": identity, "gdn_prefill_backend": "triton"},
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        distributed_executor_backend="mp",
        dtype="bfloat16",
        seed=1234,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        max_model_len=1040,
        max_num_seqs=NATIVE_ROLLOUT_PROFILE["max_num_seqs"],
        max_num_batched_tokens=NATIVE_ROLLOUT_PROFILE["max_num_batched_tokens"],
        long_prefill_token_threshold=1,
        scheduler_cls=(
            "rl_engine.integrations.engines.rollout.vllm.qwen3_next_native_rollout"
            ".ExactPrefillScheduler"
        ),
        async_scheduling=False,
        worker_cls=(
            "rl_engine.integrations.engines.rollout.vllm.qwen3_next_vllm_worker" ".SharedQwenWorker"
        ),
        logits_processors=[
            "rl_engine.integrations.engines.rollout.vllm.qwen3_next_native_rollout"
            ":ForceFrozenTokens"
        ],
        kv_cache_dtype="bfloat16",
        mamba_cache_dtype="bfloat16",
        mamba_ssm_cache_dtype="float32",
        mamba_cache_mode="none",
        generation_config="vllm",
        gpu_memory_utilization=0.85,
        kv_cache_memory_bytes=NATIVE_ROLLOUT_PROFILE["kv_cache_memory_bytes_per_gpu"],
        attention_config={"backend": "CUSTOM"},
        kernel_config={
            "moe_backend": "triton",
            "enable_flashinfer_autotune": False,
            "enable_jit_warmup": False,
        },
        compilation_config={"mode": 0, "cudagraph_mode": "NONE"},
    )


def run_rollout_cases(cases):
    from vllm import SamplingParams

    engine, identity = None, None
    try:
        for request, output in cases:
            batch, length = validate_request(request)
            if engine is None:
                identity = request["identity"]
                engine = _new_engine(identity)
            elif request["identity"] != identity:
                raise ValueError(
                    "One native rollout process cannot silently change weight/provider identity"
                )
            output = Path(output).resolve()
            records = engine.collective_rpc(
                "configure_l2_trace", args=(request, str(output.with_suffix(".worker-traces")))
            )
            if len(records) != 4 or any(
                record["request_sha256"] != request["request_sha256"] for record in records
            ):
                raise RuntimeError("All TP workers must acknowledge the exact trace request")
            sampling = []
            for row, sample in enumerate(request["sample_ids"]):
                descriptor = {
                    "request_sha256": request["request_sha256"],
                    "sample_id": sample,
                    "batch_size": batch,
                    "prompt_length": length,
                    "prefill_chunks": request["prefill_chunks"],
                    "response_tokens": (
                        None
                        if request["response_tokens"] is None
                        else request["response_tokens"][row]
                    ),
                }
                sampling.append(
                    SamplingParams(
                        temperature=0,
                        top_p=1,
                        top_k=-1,
                        seed=1234,
                        ignore_eos=True,
                        max_tokens=16,
                        detokenize=False,
                        extra_args={"shared_l2": descriptor},
                    )
                )
            generated = engine.generate(
                [{"prompt_token_ids": row} for row in request["prompt_tokens"]],
                sampling,
                use_tqdm=False,
            )
            if len(generated) != batch or any(
                len(item.outputs) != 1 or item.prompt_token_ids != request["prompt_tokens"][row]
                for row, item in enumerate(generated)
            ):
                raise RuntimeError(
                    "Native engine did not return one identified completion per prompt"
                )
            responses = [list(item.outputs[0].token_ids) for item in generated]
            if any(len(row) != 16 for row in responses):
                raise RuntimeError("Native engine did not generate sixteen tokens")
            records = engine.collective_rpc("collect_l2_trace")
            save_tensor_artifact(output, assemble_trace(request, records, responses))
    finally:
        if engine is not None:
            engine.llm_engine.engine_core.shutdown()
            del engine
            gc.collect()


def run_rollout_case(request, output):
    run_rollout_cases([(request, output)])


def batch_invariance_descriptors(request):
    """Scheduler descriptors for one version 2 request, in FCFS submission order."""
    common = {
        "request_sha256": request["request_sha256"],
        "total_requests": 1 + len(request["companions"]),
        "chunk_size": request["chunk_size"],
    }
    responses = request["response_tokens"]
    target = {
        **common,
        "bi": True,
        "sample_id": request["sample_ids"][0],
        "batch_size": 1,
        "prompt_length": len(request["prompt_tokens"][0]),
        "prefill_chunks": request["prefill_chunks"],
        "response_tokens": None if responses is None else responses[0],
        "arrival_step": request["arrival_steps"]["target"],
    }
    companions = [
        (
            prompt,
            count,
            {
                **common,
                "index": index,
                "arrival_step": request["arrival_steps"]["companions"][index],
            },
        )
        for index, (prompt, count) in enumerate(
            zip(request["companions"], request["companion_max_tokens"], strict=True)
        )
    ]
    entries = [(request["prompt_tokens"][0], RESPONSE_LENGTH, target, "shared_l2")] + [
        (prompt, count, descriptor, "shared_bi_companion")
        for prompt, count, descriptor in companions
    ]
    if request.get("target_last"):
        entries = entries[1:] + entries[:1]
    return entries


def run_batch_invariance_cases(cases):
    """Run version 2 cells in one engine; each artifact describes the traced target only."""
    from vllm import SamplingParams

    engine, identity = None, None
    try:
        for request, output in cases:
            validate_request(request)
            if request["schema_version"] != 2:
                raise ValueError("Batch-invariance cells use version 2 requests")
            if engine is None:
                identity = request["identity"]
                engine = _new_engine(identity)
            elif request["identity"] != identity:
                raise ValueError("One native rollout process cannot change its identity")
            output = Path(output).resolve()
            records = engine.collective_rpc(
                "configure_l2_trace", args=(request, str(output.with_suffix(".worker-traces")))
            )
            if len(records) != 4 or any(
                record["request_sha256"] != request["request_sha256"] for record in records
            ):
                raise RuntimeError("All TP workers must acknowledge the exact trace request")
            entries = batch_invariance_descriptors(request)
            sampling = [
                SamplingParams(
                    temperature=0,
                    top_p=1,
                    top_k=-1,
                    seed=1234,
                    ignore_eos=True,
                    max_tokens=count,
                    detokenize=False,
                    extra_args={key: descriptor},
                )
                for _, count, descriptor, key in entries
            ]
            generated = engine.generate(
                [{"prompt_token_ids": prompt} for prompt, _, _, _ in entries],
                sampling,
                use_tqdm=False,
            )
            responses = None
            for (prompt, count, _, key), item in zip(entries, generated, strict=True):
                tokens = list(item.outputs[0].token_ids)
                if item.prompt_token_ids != prompt or len(tokens) != count:
                    raise RuntimeError("Native engine did not complete every request of the cell")
                if key == "shared_l2":
                    responses = [tokens]
            records = engine.collective_rpc("collect_l2_trace")
            save_tensor_artifact(output, assemble_trace(request, records, responses))
    finally:
        if engine is not None:
            engine.llm_engine.engine_core.shutdown()
            del engine
            gc.collect()
