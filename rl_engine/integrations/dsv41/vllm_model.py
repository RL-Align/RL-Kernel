# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DeepSeek-V4.1 on vLLM main with RL-Kernel's batch-invariant model (rollout side of verl).

Design doc ``dsv41-design.md`` 10.1. vLLM main has no batch-invariant DSv4.1 path on Hopper,
so the plugin keeps only vLLM's scheduler, KV-cache allocation and attention metadata, and
every computation is :class:`Dsv41Model`'s, the actor's model:

- :class:`RlkDeepseekV41ForCausalLM` (registered under ``DeepseekV41ForCausalLM``) builds
  vLLM's attention layers on the meta device, so vLLM sees their cache specs and metadata
  builders, and nothing else of vLLM's model: no weights, Engram tables or MoE. Their
  parameters are dropped; the forward never calls them.
- The weights load from the checkpoint directory with RL-Kernel's loader (this rank's
  experts with expert parallelism, Engram tables mapped read-only from the shards); vLLM's
  weight iterator is not read.
- Each forward maps vLLM's caches and block tables to the model's
  (``engine_caches.StepCaches``) and its look-back token ids to Engram windows.
- Logits: FP32 from the BF16 head; the sampled token's log-prob is recomputed with the
  fixed-order kernel (``dsv4-logprob-v1``, :func:`_rlk_sampler_v2`), as for DSv4.

Supported: vLLM's V2 model runner, eager mode, no prefix caching, tensor and pipeline
parallel 1, no speculative decoding; data parallel attention with expert parallel MoE.

``RL_KERNEL_DSV41_TEST_LORA=rank,alpha,seed`` attaches random adapters (the gate's).
verl's adapter sync installs the actor's (``vllm_worker.py``,
:meth:`RlkDeepseekV41ForCausalLM.rlk_set_lora`).
``RL_KERNEL_DSV41_DUMP=<dir>`` (once the directory exists) saves each forward's tokens,
request layout, per-layer streams and final hidden state.
"""

from __future__ import annotations

import functools
import importlib
import os
import pathlib

import torch
from torch import nn

ENV_TEST_LORA = "RL_KERNEL_DSV41_TEST_LORA"
ENV_LORA = "RL_KERNEL_DSV41_LORA"
ENV_DUMP = "RL_KERNEL_DSV41_DUMP"
ENV_TABLES = "RL_KERNEL_DSV41_ENGRAM_TABLES"  # mmap (default) | cuda | cpu

_IN_DUMMY_RUN = [False]
_WARNED: set[str] = set()


def _use_vendored_deepgemm() -> None:
    """vLLM prefers an installed ``deep_gemm`` over its vendored build; installed here is
    RL-Kernel's pin, whose kernels vLLM main's metadata builders do not match."""
    import vllm.utils.deep_gemm as vdg

    vdg._import_deep_gemm = lambda: importlib.import_module("vllm.third_party.deep_gemm")


def _mark_dummy_runs(cls) -> None:
    """Flag the runner's dummy forwards (memory profiling, an idle data-parallel rank)."""
    if getattr(cls._dummy_run, "_rlk_marked", False):
        return
    orig = cls._dummy_run

    @functools.wraps(orig)
    def run(self, *a, **kw):
        _IN_DUMMY_RUN[0] = True
        try:
            return orig(self, *a, **kw)
        finally:
            _IN_DUMMY_RUN[0] = False

    setattr(run, "_rlk_marked", True)
    cls._dummy_run = run


def register() -> None:
    """Make vLLM build :class:`RlkDeepseekV41ForCausalLM` for DeepSeek-V4.1 checkpoints."""
    import vllm.v1.worker.gpu.model_runner as runner
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "DeepseekV41ForCausalLM",
        "rl_engine.integrations.dsv41.vllm_model:RlkDeepseekV41ForCausalLM",
    )
    _use_vendored_deepgemm()
    runner.Sampler = _rlk_sampler_v2()
    _mark_dummy_runs(runner.GPUModelRunner)


# ---------------------------------------------------------------------------- log-probs
def _warn_once(msg: str) -> None:
    if msg not in _WARNED:
        _WARNED.add(msg)
        import logging

        logging.getLogger(__name__).warning("RL-Kernel DSv4.1: %s", msg)


def _rlk_logprobs(z, ids, temps) -> torch.Tensor:
    """FP32 log-prob of ``ids`` under ``softmax(z / T)``; ``temps`` FP32 per row (``None`` =
    1), one kernel call per distinct ``T`` (the actor's ``selected_logprob`` bytes)."""
    from rl_engine.kernels.ops.cuda.logprob_dsv4 import logprob_from_logits_cuda

    if temps is None:
        return logprob_from_logits_cuda(z, ids, 1.0)[0]
    t = torch.where(temps < 1e-5, 1.0, temps).float()  # greedy rows: T = 1, as vLLM does
    logp = torch.empty(ids.shape[0], dtype=torch.float32, device=z.device)
    for tv in torch.unique(t).tolist():
        rows = (t == tv).nonzero().flatten()
        logp[rows] = logprob_from_logits_cuda(z[rows], ids[rows], tv)[0]
    return logp


def _rlk_sampler_v2():
    """vLLM's V2 sampler; the sampled token's reported log-prob is RL-Kernel's (the DSv4
    plugin's ``RlkSamplerV2``)."""
    from vllm.v1.worker.gpu.sample.sampler import Sampler

    class RlkSamplerV2(Sampler):
        def __call__(self, logits, input_batch):
            mode = self.logprobs_mode
            z = logits.float().clone() if mode in ("processed_logprobs", "raw_logprobs") else None
            out = super().__call__(logits, input_batch)
            lt = out.logprobs_tensors
            if z is None or lt is None or lt.logprobs.dim() != 2 or lt.logprobs.shape[0] == 0:
                return out
            if z.shape[0] != input_batch.idx_mapping_np.shape[0]:
                raise NotImplementedError("RL-Kernel DSv4.1 log-probs: speculative decoding")
            temps = None
            if mode == "processed_logprobs":
                top_k, top_p, *_ = self.sampling_states.get_top_k_top_p(
                    input_batch.expanded_idx_mapping, input_batch.idx_mapping_np, z.shape[1]
                )
                k_on = top_k is not None and bool((top_k < z.shape[1]).any())
                p_on = top_p is not None and bool((top_p < 1.0).any())
                if k_on or p_on:
                    _warn_once("top-k / top-p active: log-probs are vLLM's, not RL-Kernel's")
                    return out
                temps = self.sampling_states.temperature.gpu[input_batch.expanded_idx_mapping]
            ids = out.sampled_token_ids[:, 0].long()
            lt.logprobs[:, 0] = _rlk_logprobs(z, ids, temps)
            return out

    return RlkSamplerV2


# ---------------------------------------------------------------------------- the model
def model_dir(name: str) -> pathlib.Path:
    path = pathlib.Path(name)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return pathlib.Path(snapshot_download(name, allow_patterns=["*.json", "*.safetensors"]))


def _strip(module: nn.Module) -> None:
    """Drop the parameters and buffers of a meta-built module tree, and its quantization
    methods (vLLM's post-load processing would run them): only the cache layers matter."""
    for m in module.modules():
        for name in list(m._parameters):
            m._parameters[name] = None
        for name, b in list(m._buffers.items()):
            if b is not None and b.is_meta:
                m._buffers[name] = None
        if hasattr(m, "quant_method"):
            m.quant_method = None


class RlkDeepseekV41ForCausalLM(nn.Module):
    """vLLM's attention layers (caches and metadata only) + :class:`Dsv41Model`."""

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__()
        from vllm.models.deepseek_v41.nvidia.model import _select_dsv4_attn_cls

        from rl_engine.models.deepseek_v41 import Dsv41Config

        mc, pc = vllm_config.model_config, vllm_config.parallel_config
        problems = []
        if pc.tensor_parallel_size != 1:
            problems.append(f"tensor_parallel_size={pc.tensor_parallel_size}")
        if pc.pipeline_parallel_size != 1:
            problems.append(f"pipeline_parallel_size={pc.pipeline_parallel_size}")
        if getattr(pc, "prefill_context_parallel_size", 1) != 1:
            problems.append("prefill context parallel")
        if pc.enable_eplb:
            problems.append("EPLB")
        if vllm_config.cache_config.enable_prefix_caching:
            problems.append("prefix caching (enable_prefix_caching=False)")
        if vllm_config.speculative_config is not None:
            problems.append("speculative decoding")
        if not mc.enforce_eager:
            problems.append("CUDA graphs / torch.compile (enforce_eager=True)")
        if "dummy" in str(vllm_config.load_config.load_format):
            problems.append("load_format=dummy (load the checkpoint: load_format=auto)")
        if problems:
            raise NotImplementedError("RL-Kernel DSv4.1 rollout: " + ", ".join(problems))

        self.vllm_config = vllm_config
        # FlashInfer's autotune walks vLLM's layers for their weights (dropped here) and
        # tunes kernels the plugin does not run.
        vllm_config.kernel_config.enable_flashinfer_autotune = False
        self.config = mc.hf_config
        self.rlk_dir = model_dir(mc.model)
        cfg = Dsv41Config.from_pretrained(self.rlk_dir)
        text = getattr(self.config, "text_config", self.config)
        n_layers = text.num_hidden_layers
        self.rlk_config = cfg
        self.rlk_max_len = mc.max_model_len
        self.rlk_n_layers = n_layers
        self.rlk_model = None  # Dsv41Model, built in load_weights
        self.rlk_dump_step = 0

        # vLLM's attention layers, built on the meta device: they register their caches
        # (SWA, compressed KV, indexer K, compressor ring) under these prefixes.
        attn_cls = _select_dsv4_attn_cls(vllm_config)
        root = f"{prefix + '.' if prefix else ''}language_model.model.layers"
        self.rlk_prefixes = [f"{root}.{i}.attn" for i in range(n_layers)]
        index_topk = getattr(text, "index_topk", 512)
        blocks = getattr(text, "candidate_topk_blocks", 0) or 1
        tokens = vllm_config.scheduler_config.max_num_batched_tokens
        with torch.device("meta"):
            topk_buf = torch.empty(tokens, index_topk, dtype=torch.int32)
            cand_buf = torch.empty(tokens, blocks, dtype=torch.int32)
            hf = mc.hf_config
            mc.hf_config = text  # the layers read the text config
            try:
                self.attn = nn.ModuleList(
                    attn_cls(
                        vllm_config,
                        prefix=p,
                        topk_indices_buffer=topk_buf,
                        aux_stream_list=None,
                        candidate_block_buffer=cand_buf,
                    )
                    for p in self.rlk_prefixes
                )
            finally:
                mc.hf_config = hf
        _strip(self.attn)
        for layer in self.attn:
            layer.process_weights_after_loading = lambda *a, **k: None

    # ---- interfaces vLLM's runner reads
    def get_model_state_cls(self):
        from vllm.models.deepseek_v41.nvidia.model_state import DeepseekV41ModelState

        return DeepseekV41ModelState

    @property
    def token_lookback_depth(self) -> int:
        return max(self.rlk_config.engram_max_ngram_size - 1, 0)

    @property
    def decoder_replay_layers(self):
        return None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        # vLLM asks for embeddings only with prompt embeds / multimodal inputs (refused).
        return self.rlk_model.embed.index_select(0, input_ids.long())

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """FP32 ``[T, vocab]`` logits ``hidden @ head.T`` (DeepGEMM; ``dsv4-logprob-v1``)."""
        from rl_engine.kernels.ops.cuda.logprob_dsv4 import head_logits

        return head_logits(hidden_states, self.rlk_model.head)

    def get_expert_mapping(self) -> list:
        return []

    # ---- weights
    def load_weights(self, weights) -> set[str]:
        """Load :class:`Dsv41Model` from the checkpoint directory; vLLM's iterator is not
        read (it would stream every shard through the host). Returns every parameter name
        vLLM's loader may check (the attention layers have none left)."""
        from vllm.distributed import get_ep_group

        from rl_engine.models.deepseek_v41.engram import build_compressed_token_map
        from rl_engine.models.deepseek_v41.loader import load_model

        del weights
        ep = None
        pc = self.vllm_config.parallel_config
        if pc.enable_expert_parallel and get_ep_group().world_size > 1:
            ep = get_ep_group().device_group
        token_map = None
        if self.rlk_config.engram_layer_ids:
            token_map, _ = build_compressed_token_map(str(self.rlk_dir / "tokenizer.json"))
        tables = os.environ.get(ENV_TABLES, "mmap")
        self.rlk_model = load_model(
            self.rlk_dir,
            self.rlk_config,
            device=torch.device("cuda", torch.cuda.current_device()),
            max_model_len=self.rlk_max_len,
            num_layers=self.rlk_n_layers,
            ep_group=ep,
            token_map=token_map,
            engram_table_device=tables,
        )
        self._rlk_test_lora()
        return {n for n, _ in self.named_parameters()}

    def rlk_set_lora(self, adapters: list) -> None:
        """Install the actor's adapters, one ``(attention dict, MoeLora | None)`` per layer
        (``lora.adapters_from_tensors``). Adapters with the installed ones' targets and
        shapes are copied into them; others replace them."""
        layers = self.rlk_model.layers
        if len(adapters) != len(layers):
            raise ValueError(f"{len(adapters)} layers of adapters for {len(layers)} layers")

        def flat(per_layer):
            out = []
            for attn, moe in per_layer:
                out += [(f"attn.{k}", attn[k]) for k in sorted(attn)]
                out += [(f"moe.{n}", ad) for n, ad in moe.adapters()] if moe is not None else []
            return out

        cur = flat((layer.attn.w.lora, layer.moe.lora) for layer in layers)
        new = flat(adapters)
        same = len(cur) == len(new) and all(
            cn == nn_ and c.a.shape == n.a.shape and c.b.shape == n.b.shape
            and float(c.alpha) == float(n.alpha)
            for (cn, c), (nn_, n) in zip(cur, new)
        )  # fmt: skip
        if same:
            with torch.no_grad():
                for (_, c), (_, n) in zip(cur, new):
                    c.a.copy_(n.a)
                    c.b.copy_(n.b)
            return
        for layer, (attn, moe) in zip(layers, adapters):
            layer.attn.w.lora = dict(attn)
            layer.moe.lora = moe

    def _rlk_test_lora(self) -> None:
        spec = os.environ.get(ENV_TEST_LORA)
        if not spec:
            return
        from rl_engine.models.deepseek_v4.model import attach_lora

        rank, alpha, seed = spec.split(",")
        torch.manual_seed(int(seed))
        attach_lora(self.rlk_model, int(rank), float(alpha), seed=int(seed), zero_b=False)
        for p in self.rlk_model.lora_parameters():
            p.requires_grad_(False)

    # ---- forward
    def _engine_layers(self, md, reqs: int, qsl_cpu: torch.Tensor):
        from rl_engine.integrations.dsv41.engine_caches import EngineLayer

        layers = []
        for i, (p, attn) in enumerate(zip(self.rlk_prefixes, self.attn)):
            swa_md = md[f"{p}.swa_cache"]
            el = EngineLayer(swa=attn.swa_cache_layer.kv_cache, swa_table=swa_md.block_table)
            role = self.rlk_config.layer_role(i)
            if role.ratio and role.kv_source == i:
                el.kv, el.kv_table = attn.kv_cache, md[p].block_table
                idx_md = md[f"{p}.indexer.k_cache"]
                el.idx_k, el.idx_k_table = attn.indexer.k_cache.kv_cache, idx_md.block_table
                if role.ratio > 1:
                    ring = attn.compressor.state_cache.kv_cache
                    slots = md[f"{p}.compressor.state_cache"].slot_mapping
                    starts = qsl_cpu[:reqs].to(slots.device).long()
                    el.ring, el.ring_blocks = ring, slots[starts] // ring.shape[1]
            layers.append(el)
        return layers

    def _scratch_forward(self, input_ids, lookback):
        """A forward without vLLM metadata (memory profiling, an idle data-parallel rank):
        the tokens as one request on scratch caches, so the peak memory and every
        expert-parallel collective are a real forward's."""
        from rl_engine.models.deepseek_v41.attention import Dsv41Batch

        m = self.rlk_model
        n = input_ids.shape[0]
        n_run = max(min(n, self.rlk_max_len), 1)
        ids = input_ids[:n_run].clamp(min=0) if n else input_ids.new_zeros(1)
        batch = Dsv41Batch.from_lens([n_run], [n_run], m.config.sliding_window)
        caches = m.new_caches(1, max_len=n_run + 1)
        windows = None
        if m.hasher is not None:
            from rl_engine.models.deepseek_v41.engram import ngram_windows

            pos = torch.arange(n_run, device=ids.device)
            windows = ngram_windows([ids], [pos], m.config.engram_max_ngram_size)
        h = m.forward_batch(ids, batch, caches, engram_windows=windows)
        out = h.new_zeros(n, h.shape[1])
        out[: min(n, n_run)] = h[: min(n, n_run)]
        return out

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        lookback_token_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        from vllm.forward_context import get_forward_context

        from rl_engine.integrations.dsv41.engine_caches import StepCaches, engram_windows
        from rl_engine.models.deepseek_v41.attention import Dsv41Batch

        if inputs_embeds is not None:
            raise NotImplementedError("RL-Kernel DSv4.1 rollout: prompt embeddings")
        m = self.rlk_model
        md = get_forward_context().attn_metadata
        if not isinstance(md, dict) or f"{self.rlk_prefixes[0]}.swa_cache" not in md:
            return self._scratch_forward(input_ids, lookback_token_ids)
        swa = md[f"{self.rlk_prefixes[0]}.swa_cache"]
        qsl = swa.query_start_loc_cpu
        reqs = qsl.shape[0] - 1
        seq = swa.seq_lens[:reqs].cpu().tolist()
        qsl_list = qsl.tolist()
        qlen = [qsl_list[r + 1] - qsl_list[r] for r in range(reqs)]
        tokens = qsl_list[-1]
        step = StepCaches(m.config, self._engine_layers(md, reqs, qsl), seq, qlen)
        batch = Dsv41Batch.from_lens(seq, qlen, m.config.sliding_window)
        ids = input_ids[:tokens]
        windows = None
        if m.hasher is not None:
            if lookback_token_ids is None:
                raise RuntimeError("the model has Engram layers: vLLM passed no look-back ids")
            windows = engram_windows(
                ids, qsl, lookback_token_ids, m.config.engram_max_ngram_size
            )
        dump = os.environ.get(ENV_DUMP)
        if dump and not os.path.isdir(dump):
            dump = None
        res = m.forward_batch(
            ids, batch, step.caches, engram_windows=windows, return_streams=bool(dump)
        )
        h, per_layer = res if dump else (res, None)
        step.finish()
        if dump and not _IN_DUMMY_RUN[0]:
            self._rlk_dump(dump, ids, positions[:tokens], qsl, seq, per_layer, h)
        if tokens == input_ids.shape[0]:
            return h
        out = h.new_zeros(input_ids.shape[0], h.shape[1])
        out[:tokens] = h
        return out

    def _rlk_dump(self, root, ids, positions, qsl, seq, per_layer, hidden) -> None:
        from vllm.distributed import get_tensor_model_parallel_rank

        path = pathlib.Path(root)
        pc = self.vllm_config.parallel_config
        if pc.data_parallel_size > 1:
            path = path / f"dp{pc.data_parallel_rank}"
        path = path / f"rank{get_tensor_model_parallel_rank()}"
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "input_ids": ids.cpu(),
                "positions": positions.cpu(),
                "query_start_loc": qsl.clone(),
                "seq_lens": torch.tensor(seq),
                "streams": [s.cpu() for s in per_layer],
                "hidden": hidden.cpu(),
            },
            path / f"step{self.rlk_dump_step:05d}.pt",
        )
        self.rlk_dump_step += 1


