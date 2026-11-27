# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""verl training engine ``rlkernel``: the DSv4 actor on RL-Kernel's batch-invariant kernels.

A verl engine backend (``verl.workers.engine.BaseEngine``) around :class:`Dsv4Model`, so a
verl run trains DSv4 LoRA adapters with the same kernels, and the same bytes, as the vLLM
rollout plugin (``vllm_model.py``). verl's source is unchanged: this module registers the
backend in ``EngineRegistry`` when imported (``actor_rollout_ref.model.external_lib``, or
the ``_target_`` of :class:`RlkActorConfig`), and the hydra group files in ``verl_config/``
select it (``model_engine=rlkernel``, ``hydra.searchpath=[pkg://...verl_config]``).

What it does, per verl's engine contract:

- ``forward_backward_batch``: verl's micro-batching (``prepare_micro_batches``), then per
  micro-batch one packed full-prefill forward of its sequences (``forward_train`` under
  autograd, ``forward_batch`` without) and the selected-token log-prob of the next token
  at every position (``dsv4-logprob-v1``), returned jagged as ``model_output["log_probs"]``.
  Every kernel is batch-invariant, so a token's log-prob does not depend on the micro-batch
  it lands in, and equals the rollout's (``scripts/check_dsv4_vllm_gate.py``).
- Per-sample temperatures: rows are grouped by temperature (per-row kernels, so grouping
  changes no byte); a temperature below 1e-5 counts as 1, as in the rollout sampler.
- Entropy (verl asks for it with the old log-probs, as a metric): FP32 PyTorch from the
  head's logits, without gradient. Not batch-invariant, and never used in the loss: a
  training step that asks for entropy raises.
- Optimizer: verl's ``build_optimizer`` on FP32 master copies of the BF16 adapters.
  Micro-batch gradients are accumulated in FP32, averaged over the data-parallel group in
  one flat all-reduce, clipped, stepped, and rounded back to the BF16 adapters.
- ``get_per_tensor_param``: only the adapters (BF16, the bytes the actor runs), with a
  PEFT-style config. The base weights never change; the rollout loads them itself.
- ``disable_adapter``: the base model (verl's reference policy under LoRA).

verl sees no LoRA (``model.lora_rank`` 0): the reference policy, when a KL term needs one,
is a separate ``rlkernel`` engine without adapters (``ref/rlkernel_ref.yaml``), and the
rollout's sleep keeps its weights (``vllm_worker.py``).

Parallelism: data parallel attention with expert parallel MoE. ``expert_parallel_size``
E splits the world into groups of E consecutive ranks; each group holds every routed
expert once (rank r of a group keeps experts ``[r * N / E, (r + 1) * N / E)``), and every
other weight is replicated. verl sees a data-parallel group of the whole world. The MoE
exchange is a collective of the EP group, so every rank of a group runs the same number of
forwards (verl's ``same_micro_num_in_dp`` micro-batching). The routed adapters (one shared
by every expert) get their gradients summed over the EP group inside the MoE backward; they
are divided by E before the data-parallel average, so every adapter's gradient is the
world average. EP changes no forward byte (``Dsv4MoE``; ``scripts/check_dsv4_moe_ep.py``).

Memory: ``activation_recompute`` keeps one tensor per layer for the backward instead of the
layer's activations (``Dsv4Model.forward_train``).

Limits: no TP / PP / CP, LoRA only (base weights frozen), no parameter offload,
``use_remove_padding`` inputs.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from verl.workers.config import ActorConfig
from verl.workers.config.engine import EngineConfig
from verl.workers.engine.base import BaseEngine, BaseEngineCtx, EngineRegistry
from verl.workers.engine.utils import detach_tree, postprocess_batch_func, prepare_micro_batches

BACKEND = "rlkernel"
_MIN_TEMPERATURE = 1e-5


@dataclass
class RlkEngineConfig(EngineConfig):
    """``actor_rollout_ref.actor.rlkernel``.

    ``lora_rank`` / ``lora_alpha``: the adapters (here, not in verl's ``model.lora_rank``,
    which would also turn on vLLM's own LoRA in the rollout; ``vllm_worker.py`` installs
    the adapters on the plugin instead). ``lora_attention`` / ``lora_moe`` /
    ``lora_shared_expert``: which adapters :func:`attach_lora` creates. ``lora_seed``:
    their initialization seed. ``entropy_chunk_rows``: rows per FP32 logits chunk for the
    entropy metric. ``expert_parallel_size``: ranks per expert-parallel group (a divisor of
    the world size; 1 = every rank holds every expert). ``activation_recompute``: per-layer
    activation recompute in training forwards (``Dsv4Model.forward_train``): one BF16
    ``[T, 4, H]`` tensor per layer instead of the layer's activations, for one more forward
    per layer in the backward. Changes no forward byte. ``max_model_len``: the longest
    sequence (RoPE tables).
    """

    strategy: str = BACKEND
    lora_rank: int = 0
    lora_alpha: float = 16.0
    lora_attention: bool = True
    lora_moe: bool = True
    lora_shared_expert: bool = True
    lora_seed: int = 0
    entropy_chunk_rows: int = 1024
    expert_parallel_size: int = 1
    activation_recompute: bool = False
    max_model_len: int = 65536

    def __post_init__(self):
        assert self.strategy == BACKEND, f"strategy {self.strategy} is not {BACKEND}"


@dataclass
class RlkActorConfig(ActorConfig):
    """Actor (and reference) config for the ``rlkernel`` engine."""

    strategy: str = BACKEND
    rlkernel: RlkEngineConfig = field(default_factory=RlkEngineConfig)
    use_remove_padding: bool = True
    use_rollout_log_probs: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.engine = self.rlkernel


def model_family(path: str) -> str:
    """``"dsv41"`` for a DeepSeek-V4.1 checkpoint (``DeepseekV41ForCausalLM``), else
    ``"dsv4"``."""
    import json

    with open(os.path.join(path, "config.json")) as f:
        archs = json.load(f).get("architectures") or []
    return "dsv41" if "DeepseekV41ForCausalLM" in archs else "dsv4"


def _load_dsv41(path: str, device, ep_group, max_model_len: int):
    """:class:`Dsv41Model` with its Engram hasher; the Engram tables stay a read-only mmap
    of the shards, shared by every process on the node (``RL_KERNEL_DSV41_ENGRAM_TABLES``
    overrides: ``cuda`` / ``cpu``, as in the rollout plugin)."""
    from rl_engine.models.deepseek_v41 import Dsv41Config
    from rl_engine.models.deepseek_v41.engram import build_compressed_token_map
    from rl_engine.models.deepseek_v41.loader import load_model

    cfg = Dsv41Config.from_pretrained(path)
    token_map = None
    if cfg.engram_layer_ids:
        token_map, _ = build_compressed_token_map(os.path.join(path, "tokenizer.json"))
    return load_model(
        path,
        cfg,
        device=device,
        max_model_len=max_model_len,
        ep_group=ep_group,
        token_map=token_map,
        engram_table_device=os.environ.get("RL_KERNEL_DSV41_ENGRAM_TABLES", "mmap"),
    )


def _engram_windows(ids: torch.Tensor, lens: list[int], max_ngram: int) -> torch.Tensor:
    """DSv4.1's n-gram windows of packed full sequences (positions ``0..len-1``)."""
    from rl_engine.models.deepseek_v41.engram import ngram_windows

    seqs = list(ids.split(lens))
    return ngram_windows(seqs, [torch.arange(n, device=ids.device) for n in lens], max_ngram)


def named_adapters(model) -> list[tuple[str, Any]]:
    """``(name, LoraAdapter)`` for every adapter of ``model``, in
    :meth:`Dsv4Model.lora_parameters` order."""
    out = []
    for layer in model.layers:
        for target, ad in layer.attn.w.lora.items():
            out.append((f"layers.{layer.index}.attn.{target}", ad))
        if layer.moe.lora is not None:
            for name, ad in layer.moe.lora.adapters():
                out.append((f"layers.{layer.index}.moe.{name}", ad))
    return out


def _temperatures(micro_batch: TensorDict, n: int) -> list[float]:
    t = micro_batch["temperature"]
    if isinstance(t, torch.Tensor):
        vals = [float(v) for v in t.flatten().tolist()]
    else:
        vals = [float(t)] * n
    if len(vals) != n:
        raise ValueError(f"{len(vals)} temperatures for {n} sequences")
    return [1.0 if v < _MIN_TEMPERATURE else v for v in vals]


@EngineRegistry.register(model_type="language_model", backend=BACKEND, device="cuda")
class RlkernelEngine(BaseEngine):
    def __init__(self, model_config, engine_config, optimizer_config, checkpoint_config, **_):
        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config
        self.mode = None
        self.model = None
        self.adapters: list[tuple[str, Any]] = []
        self.master: list[torch.Tensor] = []
        self.optimizer = None
        self.lr_scheduler = None
        self._lora_off = False
        self.ep_group = None
        self.ep_size = 1
        self._routed: list[int] = []  # master indices of the routed (EP-summed) adapters
        self.family = "dsv4"

    # ------------------------------------------------------------------ setup
    @property
    def is_param_offload_enabled(self) -> bool:
        return False

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return False

    @property
    def device(self) -> torch.device:
        from verl.utils.device import get_device_id

        return torch.device("cuda", get_device_id())

    def initialize(self):
        from rl_engine.models.deepseek_v4.loader import load_model
        from rl_engine.models.deepseek_v4.model import attach_lora

        cfg = self.engine_config
        if cfg.param_offload or cfg.optimizer_offload:
            raise NotImplementedError("rlkernel: parameter / optimizer offload")
        merge = (getattr(self.model_config, "lora", None) or {}).get("merge", False)
        if merge:
            raise NotImplementedError("rlkernel: model.lora.merge (the rollout runs the adapters)")
        if int(getattr(self.model_config, "lora_rank", 0) or 0) > 0:
            raise ValueError(
                "rlkernel: set actor_rollout_ref.actor.rlkernel.lora_rank, not model.lora_rank "
                "(which turns on vLLM's own LoRA in the rollout)"
            )
        torch.cuda.set_device(self.device)
        self._init_expert_parallel(cfg.expert_parallel_size)
        path = self.model_config.local_path or self.model_config.path
        self.family = model_family(path)
        if self.family == "dsv41":
            self.model = _load_dsv41(path, self.device, self.ep_group, cfg.max_model_len)
        else:
            self.model = load_model(
                path, device=self.device, ep_group=self.ep_group, max_model_len=cfg.max_model_len
            )
        rank = cfg.lora_rank
        if rank <= 0:
            if not cfg.forward_only:
                raise NotImplementedError("rlkernel trains LoRA adapters only (lora_rank > 0)")
            return
        attach_lora(
            self.model,
            rank=rank,
            alpha=float(cfg.lora_alpha),
            attention=cfg.lora_attention,
            moe=cfg.lora_moe,
            shared_expert=cfg.lora_shared_expert,
            seed=cfg.lora_seed,
            zero_b=True,
        )
        self.adapters = named_adapters(self.model)
        self._routed = []
        i = 0
        for name, _ in self.adapters:
            if name.endswith((".moe.fc1", ".moe.fc2")):
                self._routed += [i, i + 1]  # A, B (lora_parameters order)
            i += 2
        self.master = []
        for p in self.model.lora_parameters():
            m = p.detach().float().clone()
            m.grad = torch.zeros_like(m)
            self.master.append(m)
        if not cfg.forward_only:
            from verl.workers.config.optimizer import build_optimizer

            self.optimizer = build_optimizer(self.master, self.optimizer_config)
            self.lr_scheduler = self._build_lr_scheduler(self.optimizer)

    def _init_expert_parallel(self, ep: int) -> None:
        world, rank = dist.get_world_size(), dist.get_rank()
        if ep < 1 or world % ep:
            raise ValueError(f"expert_parallel_size {ep} does not divide the world size {world}")
        self.ep_size = ep
        if ep == 1:
            return
        for start in range(0, world, ep):  # every rank creates every group
            group = dist.new_group(list(range(start, start + ep)))
            if start <= rank < start + ep:
                self.ep_group = group

    def _build_lr_scheduler(self, optimizer):
        from verl.utils.torch_functional import (
            get_constant_schedule_with_warmup,
            get_cosine_schedule_with_warmup,
        )

        oc = self.optimizer_config
        total = oc.total_training_steps
        warmup = oc.lr_warmup_steps
        if warmup is None or warmup <= 0:
            warmup = int(oc.lr_warmup_steps_ratio * total)
        kind = getattr(oc, "lr_scheduler_type", "constant")
        if kind == "constant":
            return get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup)
        if kind == "cosine":
            return get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=warmup,
                num_training_steps=total,
                min_lr_ratio=oc.min_lr_ratio,
                num_cycles=oc.num_cycles,
                zero_indexed_step=oc.zero_indexed_step,
            )
        raise NotImplementedError(f"lr_scheduler_type {kind}")

    # ------------------------------------------------------------ parallelism
    def get_data_parallel_rank(self):
        return dist.get_rank()

    def get_data_parallel_size(self):
        return dist.get_world_size()

    def get_data_parallel_group(self):
        return dist.group.WORLD

    def is_mp_src_rank_with_outputs(self):
        return True

    def train_mode(self, **kwargs):
        return BaseEngineCtx(self, "train", **kwargs)

    def eval_mode(self, **kwargs):
        return BaseEngineCtx(self, "eval", **kwargs)

    # ---------------------------------------------------------------- forward
    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False):
        return_model_output = tu.get_non_tensor_data(data, key="return_model_output", default=False)
        group = self.get_data_parallel_group()
        tokens = data["loss_mask"].sum().to(self.device)
        dist.all_reduce(tokens, op=dist.ReduceOp.SUM, group=group)
        tu.assign_non_tensor(data, batch_num_tokens=tokens.item())
        tu.assign_non_tensor(data, dp_size=self.get_data_parallel_size())

        micro_batches, indices = prepare_micro_batches(
            data=data, dp_group=group, same_micro_num_in_dp=True
        )
        outputs = []
        for i, mb in enumerate(micro_batches):
            ctx = torch.no_grad() if forward_only else contextlib.nullcontext()
            with ctx, torch.profiler.record_function(f"micro_batch{i}"):
                loss, out = self.forward_step(mb, loss_function, forward_only)
                if not forward_only:
                    loss.backward()
                    self._accumulate_grads()
                    if not return_model_output:
                        out.pop("model_output", None)
            outputs.append(out)
        return postprocess_batch_func(output_lst=outputs, indices=indices, data=data)

    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only):
        mb = micro_batch.to(self.device)
        if tu.get_non_tensor_data(mb, key="use_remove_padding", default=True) is False:
            raise NotImplementedError("rlkernel needs use_remove_padding inputs")
        model = self.model
        ids_nt = mb["input_ids"]
        offsets = ids_nt.offsets()
        lens = offsets.diff().tolist()
        ids = ids_nt.values().long()
        window = model.config.sliding_window
        if self.family == "dsv41":
            from rl_engine.models.deepseek_v41.attention import Dsv41Batch

            batch = Dsv41Batch.from_lens(lens, lens, window, device=self.device)
        else:
            from rl_engine.models.deepseek_v4.attention import Dsv4Batch

            batch = Dsv4Batch.from_lens(lens, lens, window, self.device)
        pos = mb["position_ids"]
        pos = pos.values() if pos.is_nested else pos
        if pos.dim() != 1 or not torch.equal(pos.long(), batch.positions):
            raise ValueError("rlkernel expects positions 0..len-1 for every sequence")
        targets = torch.roll(ids, -1)
        caches = model.new_caches(len(lens), max(lens))
        kw = {}
        if self.family == "dsv41" and model.hasher is not None:
            kw["engram_windows"] = _engram_windows(ids, lens, model.config.engram_max_ngram_size)
        if forward_only or not torch.is_grad_enabled():
            hidden = model.forward_batch(ids, batch, caches, **kw)
        else:
            recompute = self.engine_config.activation_recompute
            hidden = model.forward_train(ids, batch, caches, recompute=recompute, **kw)
        del caches

        temps = _temperatures(mb, len(lens))
        per_token = torch.repeat_interleave(
            torch.tensor(temps, dtype=torch.float32), torch.tensor(lens)
        ).to(self.device)
        logp = self._logprob(hidden, targets, temps, per_token)
        model_output = {"log_probs": torch.nested.nested_tensor_from_jagged(logp, offsets)}
        if tu.get_non_tensor_data(mb, key="calculate_entropy", default=False):
            if not forward_only:
                raise NotImplementedError(
                    "rlkernel: entropy in the loss (set entropy_coeff=0, calculate_entropy=False)"
                )
            ent = self._entropy(hidden, temps, per_token)
            model_output["entropy"] = torch.nested.nested_tensor_from_jagged(ent, offsets)

        if loss_function is not None:
            loss, metrics = loss_function(
                model_output=model_output, data=mb, dp_group=self.get_data_parallel_group()
            )
        else:
            assert forward_only, "forward_only must be True when loss_function is None"
            loss, metrics = torch.tensor(1.0, device=self.device), {}
        return loss, {
            "model_output": detach_tree(model_output),
            "loss": loss.detach().item(),
            "metrics": metrics,
        }

    def _logprob(self, hidden, targets, temps, per_token):
        uniq = sorted(set(temps))
        if len(uniq) == 1:
            return self.model.selected_logprob(hidden, targets, uniq[0])[0]
        out = torch.zeros(hidden.shape[0], dtype=torch.float32, device=hidden.device)
        for t in uniq:
            rows = torch.nonzero(per_token == t).flatten()
            lp = self.model.selected_logprob(hidden[rows], targets[rows], t)[0]
            out = out.index_copy(0, rows, lp)
        return out

    @torch.no_grad()
    def _entropy(self, hidden, temps, per_token):
        from rl_engine.kernels.ops.cuda.logprob_dsv4 import head_logits

        head = self.model.head
        step = self.engine_config.entropy_chunk_rows
        out = torch.empty(hidden.shape[0], dtype=torch.float32, device=hidden.device)
        h = hidden.detach()
        for lo in range(0, h.shape[0], step):
            hi = min(h.shape[0], lo + step)
            z = head_logits(h[lo:hi], head) / per_token[lo:hi, None]
            # Shifted by the row max: a trained model's peaked rows otherwise lose H to the
            # cancellation in lse - sum(p z). H >= 0; FP32 rounding can still land below 0.
            z = z - z.amax(dim=-1, keepdim=True)
            p = torch.softmax(z, dim=-1)
            out[lo:hi] = (torch.logsumexp(z, dim=-1) - (p * z).sum(-1)).clamp_min_(0.0)
        return out

    # -------------------------------------------------------------- optimizer
    def _accumulate_grads(self):
        for p, m in zip(self.model.lora_parameters(), self.master):
            if p.grad is not None:
                m.grad.add_(p.grad.float())
                p.grad = None

    def optimizer_zero_grad(self):
        for m in self.master:
            m.grad.zero_()

    def optimizer_step(self):
        grads = [m.grad for m in self.master]
        if self.ep_size > 1:
            # Summed over the EP group in the MoE backward: make them per-rank like the rest.
            for i in self._routed:
                grads[i].div_(self.ep_size)
        if self.get_data_parallel_size() > 1:
            flat = torch.cat([g.flatten() for g in grads])
            dist.all_reduce(flat, op=dist.ReduceOp.AVG, group=self.get_data_parallel_group())
            for g, v in zip(grads, flat.split([g.numel() for g in grads])):
                g.copy_(v.view_as(g))
        clip = self.optimizer_config.clip_grad
        grad_norm = torch.nn.utils.clip_grad_norm_(self.master, max_norm=clip)
        if not torch.isfinite(grad_norm):
            print(f"WARN: rlkernel grad_norm is not finite: {grad_norm}")
            self.optimizer_zero_grad()
        else:
            self.optimizer.step()
            with torch.no_grad():
                for p, m in zip(self.model.lora_parameters(), self.master):
                    p.copy_(m.to(p.dtype))
        return grad_norm.item()

    def lr_scheduler_step(self):
        self.lr_scheduler.step()
        return self.lr_scheduler.get_last_lr()[0]

    # ---------------------------------------------------------------- weights
    def peft_config(self) -> Optional[dict]:
        if not self.adapters:
            return None
        try:
            from peft import PeftType, TaskType

            peft_type, task_type = PeftType.LORA, TaskType.CAUSAL_LM
        except ImportError:
            peft_type, task_type = "LORA", "CAUSAL_LM"
        cfg = self.engine_config
        targets = []
        if cfg.lora_attention:
            targets += ["wq_a", "wkv", "wq_b", "wo_b"]
        if cfg.lora_moe:
            targets += ["fc1", "fc2"]
        if cfg.lora_shared_expert:
            targets += ["shared_fc1", "shared_fc2"]
        return {
            "peft_type": peft_type,
            "task_type": task_type,
            "r": int(cfg.lora_rank),
            "lora_alpha": float(cfg.lora_alpha),
            "target_modules": targets,
            "lora_dropout": 0.0,
            "bias": "none",
            "rlkernel_seed": cfg.lora_seed,
        }

    def get_per_tensor_param(self, layered_summon=False, base_sync_done=False, **kwargs):
        """The adapters only, ``<name>.lora_A`` ``[r, in]`` / ``<name>.lora_B`` ``[out, r]``
        (BF16, the bytes the actor runs; see :func:`named_adapters`)."""
        if not self.adapters:
            raise NotImplementedError("rlkernel exports LoRA adapters only")

        def gen():
            for name, ad in self.adapters:
                yield f"{name}.lora_A", ad.a.detach()
                yield f"{name}.lora_B", ad.b.detach()

        return gen(), self.peft_config()

    def set_rollout_weight_quantizer(self, quantizer) -> None:
        """Accepted and unused: verl offers it for a quantized checkpoint (DSv4's FP8 / FP4),
        to quantize base weights before the sync; this engine syncs only the BF16 adapters,
        which the rollout runs unquantized, as the actor does."""
        self._rollout_quantizer = quantizer

    @contextlib.contextmanager
    def disable_adapter(self):
        saved = [(layer.attn.w.lora, layer.moe.lora) for layer in self.model.layers]
        for layer in self.model.layers:
            layer.attn.w.lora = {}
            layer.moe.lora = None
        try:
            yield
        finally:
            for layer, (attn, moe) in zip(self.model.layers, saved):
                layer.attn.w.lora = attn
                layer.moe.lora = moe

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        """Nothing moves (verl's mode contexts call this with nothing to move when offload
        is off); asking to move the model or optimizer to the CPU raises."""
        super().to(device, model=model, optimizer=optimizer, grad=grad)
        if str(device) == "cpu" and (model or optimizer):
            raise NotImplementedError("rlkernel: offload to CPU")

    # ------------------------------------------------------------- checkpoint
    def _ckpt_file(self, local_path: str) -> str:
        return os.path.join(local_path, "rlkernel_adapters.pt")

    def save_checkpoint(
        self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None, **_
    ):
        if dist.get_rank() == 0:
            os.makedirs(local_path, exist_ok=True)
            state = {
                "global_step": global_step,
                "names": [n for n, _ in self.adapters],
                "master": [m.detach().cpu() for m in self.master],
                "optimizer": self.optimizer.state_dict() if self.optimizer else None,
                "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                "peft_config": {k: str(v) for k, v in (self.peft_config() or {}).items()},
            }
            torch.save(state, self._ckpt_file(local_path))
        dist.barrier()

    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True, **_):
        state = torch.load(self._ckpt_file(local_path), map_location="cpu", weights_only=False)
        if state["names"] != [n for n, _ in self.adapters]:
            raise ValueError("checkpoint adapters do not match the model's")
        with torch.no_grad():
            for m, saved in zip(self.master, state["master"]):
                m.copy_(saved)
            for p, m in zip(self.model.lora_parameters(), self.master):
                p.copy_(m.to(p.dtype))
        if self.optimizer is not None and state["optimizer"] is not None:
            self.optimizer.load_state_dict(state["optimizer"])
        if self.lr_scheduler is not None and state["lr_scheduler"] is not None:
            self.lr_scheduler.load_state_dict(state["lr_scheduler"])
