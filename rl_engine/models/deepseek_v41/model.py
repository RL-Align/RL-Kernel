# SPDX-License-Identifier: Apache-2.0
"""Text V4.1 original-layer graph, with frozen packed parameters and input STE.

Training and inference use the same ROCm kernels (shared-kernel parity). This
profile differentiates dense/nonquantized parameters, routing and input paths;
packed weights, token embeddings and the head are explicitly frozen. It is not
full-parameter training or an independent inference backend.
"""

from dataclasses import dataclass
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from rl_engine.backends.rocm.engram.ngram_hash import EngramHash
from rl_engine.backends.rocm.mhc.v41_delayed import collapse, inverse_rms, split_sinkhorn
from rl_engine.backends.rocm.mhc.triton_mhc import mhc_post_dsv4
from rl_engine.backends.rocm.moe.v41_router import route_top6
from rl_engine.backends.rocm.moe.v41_combine import combine
from rl_engine.backends.rocm.norm.v41_rmsnorm import rmsnorm
from .attention import Attention
from .checkpoint import LayerCheckpoint
from .config import DeepSeekV41Config
from .expert import V41Expert
from .loading import projection, engram


class MoE(torch.nn.Module):
    def __init__(self, config, checkpoint, layer_id, device):
        super().__init__()
        t = config.text
        if (
            t["n_routed_experts"],
            t["num_experts_per_tok"],
            t["n_shared_experts"],
            t["scoring_func"],
            t["routed_scaling_factor"],
            t["swiglu_limit"],
        ) != (384, 6, 1, "sqrtsoftplus", 1.5, 10.0):
            raise ValueError("unsupported MoE numerical profile")
        prefix = f"layers.{layer_id}.ffn"
        self.gate = torch.nn.Parameter(
            checkpoint.tensor(prefix + ".gate.weight", device=device).float()
        )
        self.register_buffer(
            "bias", checkpoint.tensor(prefix + ".gate.bias", device=device).float()
        )

        def expert(name):
            return V41Expert(
                *(projection(checkpoint, name + "." + k, device) for k in ("w1", "w2", "w3"))
            )

        self.experts = torch.nn.ModuleList([expert(prefix + f".experts.{i}") for i in range(384)])
        self.shared = expert(prefix + ".shared_experts")
        self.layer_id = layer_id

    def forward(self, x, trace):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        weights, ids = route_top6(F.linear(flat.float(), self.gate), self.bias)
        # Original implementation accumulates experts in ascending global ID.
        ordered, permutation = ids.sort(-1)
        ordered_weights = weights.gather(-1, permutation)
        slots = flat.new_zeros((flat.shape[0], 6, flat.shape[1]))
        for original_id in ordered.unique().tolist():
            tokens, slot = torch.where(ordered == original_id)
            slots[tokens, slot] = self.experts[original_id](
                flat[tokens], ordered_weights[tokens, slot]
            )
        trace(f"layers.{self.layer_id}.moe.expert_ids", ordered)
        trace(f"layers.{self.layer_id}.moe.routing_weights", ordered_weights)
        return combine(slots, self.shared(flat)).reshape(shape)


class Block(torch.nn.Module):
    def __init__(self, config, checkpoint, layer_id, device, tp=None):
        super().__init__()
        self.layer_id = layer_id
        self.t = config.text
        prefix = f"layers.{layer_id}"
        self.attn = Attention(config, checkpoint, layer_id, device, tp=tp)
        self.ffn = MoE(config, checkpoint, layer_id, device)
        self.engram = (
            engram(checkpoint, config, layer_id, device)
            if config.layer_plan(layer_id).engram
            else None
        )
        for name in ("attn", "ffn"):
            setattr(
                self,
                name + "_gain",
                torch.nn.Parameter(
                    checkpoint.tensor(prefix + f".{name}_norm.weight", device=device)
                ),
            )
            for tail in ("fn", "base", "scale"):
                key = f"hc_{name}_{tail}"
                setattr(
                    self,
                    key,
                    torch.nn.Parameter(
                        checkpoint.tensor(prefix + "." + key, device=device).float()
                    ),
                )

    def mixes(self, x, kind):
        flat = x.flatten(2).float()
        mixed = F.linear(flat, getattr(self, f"hc_{kind}_fn")) * inverse_rms(
            flat, eps=self.t["rms_norm_eps"]
        )
        return split_sinkhorn(
            mixed,
            getattr(self, f"hc_{kind}_scale"),
            getattr(self, f"hc_{kind}_base"),
            eps=self.t["hc_eps"],
            iters=self.t["hc_sinkhorn_iters"],
        )

    @staticmethod
    def post(y, residual, post, comb):
        b, s, n, d = residual.shape
        return mhc_post_dsv4(
            y.reshape(-1, d),
            residual.reshape(-1, n, d),
            post.reshape(-1, n),
            comb.reshape(-1, n, n),
        ).reshape(b, s, n, d)

    def forward(self, x, previous_pre, start_pos, state, hashes, trace):
        if self.engram is not None:
            x = self.engram(x, hashes)
            trace(f"layers.{self.layer_id}.engram", x)
        pre, post, comb = self.mixes(x, "attn")
        a = rmsnorm(collapse(x, previous_pre), self.attn_gain, eps=self.t["rms_norm_eps"])
        trace(f"layers.{self.layer_id}.attention.input", a)
        a = self.attn(a, start_pos, state, trace)
        x = self.post(a, x, post, comb)
        trace(f"layers.{self.layer_id}.attention.residual", x)
        next_pre, post, comb = self.mixes(x, "ffn")
        f = rmsnorm(collapse(x, pre), self.ffn_gain, eps=self.t["rms_norm_eps"])
        f = self.ffn(f, trace)
        x = self.post(f, x, post, comb)
        trace(f"layers.{self.layer_id}.hidden", x)
        trace(f"layers.{self.layer_id}.next_pre", next_pre)
        return x, next_pre


@dataclass
class ModelOutput:
    hidden: torch.Tensor
    logits: torch.Tensor
    raw_token_logprobs: torch.Tensor
    trace: dict
    state: dict


@dataclass
class StageOutput:
    residual: torch.Tensor
    previous_pre: torch.Tensor
    trace: dict
    state: dict


class DeepSeekV41TextModel(torch.nn.Module):
    profile = "dsv41-text-frozen-quant-embedding-head-input-ste-shared-kernels-v1"

    def __init__(
        self,
        config,
        checkpoint,
        *,
        layer_ids=(0, 1, 2, 3),
        device="cuda",
        max_seq_len=256,
        max_batch_size=2,
        first_stage=True,
        last_stage=True,
        attention_tp=None,
    ):
        super().__init__()
        config.plan_layers(tuple(layer_ids))
        self.config = config
        self.layer_ids = tuple(layer_ids)
        self.first_stage, self.last_stage = first_stage, last_stage
        device = torch.device(device)
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        if first_stage:
            self.register_buffer("embedding", checkpoint.tensor("embed.weight", device=device))
        if last_stage:
            self.register_buffer("head", checkpoint.tensor("head.weight", device=device).float())
            self.norm = torch.nn.Parameter(checkpoint.tensor("norm.weight", device=device))
        mapping = load_file(
            str(checkpoint.directory / "token-map.safetensors"), device=str(device)
        )["token_map"]
        self.hasher = EngramHash(
            config, mapping, max_seq_len=max_seq_len, max_batch_size=max_batch_size
        )
        self.blocks = torch.nn.ModuleDict()
        for layer in layer_ids:
            self.blocks[str(layer)] = Block(config, checkpoint, layer, device, tp=attention_tp)
            print("RLK_V41_LAYER_LOADED", layer, flush=True)
        self.reset()

    @classmethod
    def from_checkpoint(cls, directory, **kwargs):
        from pathlib import Path

        return cls(
            DeepSeekV41Config.from_json(Path(directory) / "config.json"),
            LayerCheckpoint(directory),
            **kwargs,
        )

    def reset(self):
        self.state = {}
        self._length = 0
        self._batch = None
        self.hasher.reset()
        for block in self.blocks.values():
            block.attn.reset()

    def forward(
        self,
        input_ids,
        *,
        start_pos=0,
        positions=None,
        mask=None,
        collect_trace=False,
        boundary=None,
        target_ids=None,
    ):
        b, s = input_ids.shape
        if start_pos == 0:
            self.reset()
            self._batch = b
        if start_pos != self._length or b != self._batch or (start_pos > 0 and s != 1):
            raise ValueError(
                "model requests require contiguous prefill/decode positions and fixed batch"
            )
        if positions is not None and not torch.equal(
            positions,
            torch.arange(start_pos, start_pos + s, device=input_ids.device)[None].expand(b, -1),
        ):
            raise ValueError("this profile requires contiguous positions")
        if mask is not None and (
            mask.shape != input_ids.shape or mask.dtype != torch.bool or not bool(mask.all())
        ):
            raise ValueError(
                "this profile requires all-valid text tokens; "
                "arbitrary padding masks are unsupported"
            )
        hashes = self.hasher(input_ids, start_pos)
        t = self.config.text
        if self.first_stage:
            if boundary is not None:
                raise ValueError("first stage does not accept an upstream boundary")
            x = (
                F.embedding(input_ids, self.embedding)[:, :, None, :]
                .expand(-1, -1, t["hc_mult"], -1)
                .contiguous()
            )
            pre = torch.zeros((b, s, t["hc_mult"]), device=x.device, dtype=torch.float32)
            pre[..., 0] = 1
        else:
            if boundary is None:
                raise ValueError("pipeline stage requires residual and delayed-pre boundary")
            x, pre = boundary
            if (
                x.shape != (b, s, t["hc_mult"], t["hidden_size"])
                or pre.shape != (b, s, t["hc_mult"])
                or x.dtype != torch.bfloat16
                or pre.dtype != torch.float32
                or x.device != self.device
                or pre.device != self.device
            ):
                raise ValueError("invalid pipeline boundary shape, dtype or device")
        traces = {}

        def trace(name, value):
            if collect_trace:
                traces[name] = value.detach().clone()

        if self.first_stage:
            trace("embedding", x)
        for layer, block in self.blocks.items():
            h = (
                hashes[:, :, t["engram_layer_ids"].index(int(layer))]
                if block.engram is not None
                else None
            )
            x, pre = block(x, pre, start_pos, self.state, h, trace)
        if self.last_stage:
            hidden = rmsnorm(collapse(x, pre), self.norm, eps=t["rms_norm_eps"])
            logits = F.linear(hidden.float(), self.head)
            if target_ids is None:
                logprobs = (
                    logits[:, :-1].log_softmax(-1).gather(-1, input_ids[:, 1:, None]).squeeze(-1)
                )
            else:
                if (
                    target_ids.shape != input_ids.shape
                    or target_ids.dtype != torch.int64
                    or target_ids.device != input_ids.device
                ):
                    raise ValueError(
                        "target_ids must match input_ids shape, int64 dtype and device"
                    )
                logprobs = logits.log_softmax(-1).gather(-1, target_ids[..., None]).squeeze(-1)
            trace("hidden", hidden)
            trace("logits", logits)
            trace("raw_token_logprobs", logprobs)
        snapshots = {}
        for key, value in self.state.items():
            owner, tensor = value
            snapshots[key] = (owner, tensor.detach().clone())
            trace(f"state.{key}.owner", torch.tensor([owner], device=x.device))
            trace(f"state.{key}.tensor", tensor)
        for layer, block in self.blocks.items():
            trace(f"state.layers.{layer}.window", block.attn.window)
            if hasattr(block.attn, "compressor"):
                compressor = block.attn.compressor
                if compressor.partial_kv is not None:
                    trace(f"state.layers.{layer}.partial_kv", compressor.partial_kv)
                    trace(f"state.layers.{layer}.partial_score", compressor.partial_score)
        self._length = start_pos + s
        if not self.last_stage:
            return StageOutput(x, pre, traces, snapshots)
        return ModelOutput(hidden, logits, logprobs, traces, snapshots)
