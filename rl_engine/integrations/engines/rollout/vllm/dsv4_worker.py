"""Real eager vLLM worker with explicit RLK bindings and versioned raw-score capture."""

import hashlib
import inspect
import json
import os
import types
from pathlib import Path

import torch
from vllm.v1.worker.gpu_worker import Worker

from rl_engine.integrations.orchestrators.verl.mhc import bind_mhc_worker


class RLKDeepSeekV4Worker(Worker):
    def set_case(self, case_id):
        if not case_id or not all(c.isalnum() or c in "-_" for c in case_id):
            raise ValueError("invalid case ID")
        self._rlk_case_id = case_id
        self._rlk_case_batches = []
        os.environ["DSV4_PHASE"] = "generation"
        return dict(
            pid=os.getpid(), weight_version=self._dsv4_diagnostic_p1.version, case_id=case_id
        )

    def case_batches(self):
        return dict(
            case_id=self._rlk_case_id,
            weight_version=self._dsv4_diagnostic_p1.version,
            batches=self._rlk_case_batches,
        )

    def begin_generation_trace(self):
        os.environ["DSV4_PHASE"] = "generation"
        return {"weight_version": self._dsv4_diagnostic_p1.version, "phase": "generation"}

    def load_actor_update(self, path):
        payload = torch.load(path, map_location="cuda", weights_only=True)
        p1 = self._dsv4_diagnostic_p1
        version = payload["weight_version"]
        if version != p1.version + 1:
            raise ValueError("actor update must advance exactly one version")
        wrapper = self.model_runner.get_model()

        def digest(tensor):
            return hashlib.sha256(
                tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest()

        before = digest(wrapper.model.norm.weight)
        expected = digest(payload["weights"]["norm.weight"])
        with p1.updating(version, torch.cuda.synchronize):
            loaded = wrapper.load_weights(payload["weights"].items())
            from .checkpoint_sync import verify_loaded_update

            verified = verify_loaded_update(wrapper, payload, loaded)
            actual = digest(wrapper.model.norm.weight)
            if actual != expected:
                raise RuntimeError("rollout final norm does not match actor update bytes")
        record = dict(
            weight_version=p1.version,
            updated_tensors=len(payload["weights"]),
            loaded_names=sorted(loaded),
            verified_original_tensors=verified,
            parameter_load_completed=True,
            pid=os.getpid(),
            norm_before=before,
            norm_after=actual,
            norm_expected=expected,
            weight_changed=before != actual,
        )
        Path("/outputs/rollout-update-ack.json").write_text(json.dumps(record, indent=2))
        return record

    def load_model(self, **kwargs):
        print(
            "ACTUAL_WORKER_IMPLEMENTATION",
            type(self),
            inspect.getfile(type(self)),
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            flush=True,
        )
        super().load_model(**kwargs)
        wrapper = self.model_runner.get_model()
        model = wrapper.model
        print(
            "ACTUAL_ROLLOUT_MODEL",
            type(wrapper),
            inspect.getfile(type(wrapper)),
            type(model),
            inspect.getfile(type(model)),
            flush=True,
        )
        print("ACTUAL_WEIGHT_VERSION", 0, os.environ.get("DSV4_CHECKPOINT_SHA256"), flush=True)
        root = Path("/outputs/rollout-first-execution")
        root.mkdir(exist_ok=True)
        context = {"layer": None}
        seen = set()
        trace_root = Path("/outputs/generation-trace")
        trace_root.mkdir(exist_ok=True)
        trace_index = [0]
        forward_index = [-1]
        (trace_root / "manifest.jsonl").write_text("")

        def snapshot(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().clone()
            if isinstance(value, tuple):
                return tuple(snapshot(x) for x in value)
            if isinstance(value, list):
                return [snapshot(x) for x in value]
            if isinstance(value, dict):
                return {k: snapshot(v) for k, v in value.items()}
            return (
                value
                if value is None or isinstance(value, (int, float, str, bool))
                else repr(type(value))
            )

        def dump(name, value):
            if os.environ.get("DSV4_PHASE") != "generation":
                return
            index = trace_index[0]
            trace_index[0] += 1
            filename = f"{index:04d}.pt"
            torch.save(snapshot(value), trace_root / filename)
            with (trace_root / "manifest.jsonl").open("a") as f:
                f.write(
                    json.dumps(
                        dict(
                            index=index,
                            forward=forward_index[0],
                            layer=context["layer"],
                            name=name,
                            file=filename,
                        )
                    )
                    + "\n"
                )

        boundaries = [("embedding", model.embed_tokens), ("final_norm", model.norm)] + [
            (f"layer.{i}.{part}", getattr(model.layers[i], part))
            for i in range(model.start_layer, model.end_layer)
            for part in ("attn", "ffn")
        ]
        for i in range(model.start_layer, model.end_layer):
            for name, module in model.layers[i].attn.named_modules():
                if name:
                    boundaries.append((f"layer.{i}.attn.{name}", module))
            ffn = model.layers[i].ffn
            from .router_projection import bind_rollout_router_projection

            bind_rollout_router_projection(ffn.gate)
            print(
                "MOE_MODULES",
                i,
                [(n, type(m).__module__ + "." + type(m).__name__) for n, m in ffn.named_modules()],
                flush=True,
            )
            for name, module in ffn.named_modules():
                if name:
                    boundaries.append((f"layer.{i}.ffn.{name}", module))
        for label, module in boundaries:

            def before(_m, args, label=label):
                dump(label + ".input", args)

            def after(_m, args, out, label=label):
                dump(label + ".output", out)

            module.register_forward_pre_hook(before)
            module.register_forward_hook(after)

        def observe(framework, role, version, backend, profile, name, tensor):
            dump("rlk." + name, tensor)
            identity = (version, context["layer"], name)
            if identity in seen:
                return
            seen.add(identity)
            record = dict(
                framework=framework,
                role=role,
                version=version,
                backend=backend,
                mode="split",
                profile=profile,
                layer=context["layer"],
                stage=name,
                dtype=str(tensor.dtype),
                shape=list(tensor.shape),
                source=inspect.getfile(bind_mhc_worker),
            )
            print("RLK_FIRST_EXECUTION", json.dumps(record), flush=True)
            with (root / "records.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            torch.save(tensor.detach().cpu(), root / f"layer-{context['layer']}-{name}.pt")

        p1 = bind_mhc_worker(
            model,
            "rollout",
            backend=os.environ.get("DSV4_MHC_BACKEND", "rocm"),
            mode="split",
            version=0,
            observer=observe,
        )
        from rl_engine.integrations.engines.rollout.vllm.outputs import (
            bind_output_head,
            selected_raw_logprobs,
        )

        bind_output_head(wrapper, p1)
        for i in range(model.start_layer, model.end_layer):

            def entering(_m, _args, i=i):
                context["layer"] = i
                dump(f"layer.{i}.input", _args)

            def leaving(_m, _args, _output, i=i):
                dump(f"layer.{i}.output", _output)
                context["layer"] = None

            model.layers[i].register_forward_pre_hook(entering)
            model.layers[i].register_forward_hook(leaving)
        from vllm.model_executor.kernels.linear.scaled_mm.aiter import AiterFp8BlockScaledMMKernel

        original_scaled_mm = AiterFp8BlockScaledMMKernel.apply_block_scaled_mm
        replayed = [False]

        def capture_scaled_mm(kernel, A, B, As, Bs):
            out = original_scaled_mm(kernel, A, B, As, Bs)
            if (
                os.environ.get("DSV4_PHASE") == "generation"
                and forward_index[0] == 0
                and context["layer"] == 0
            ):
                dump(f"fp8.mm.{B.shape[0]}.{B.shape[1]}", {"A": A, "As": As, "output": out})
            if (
                not replayed[0]
                and os.environ.get("DSV4_PHASE") == "generation"
                and forward_index[0] == 1
                and context["layer"] == 0
                and tuple(B.shape) == (4096, 2048)
            ):
                replayed[0] = True
                operands = dict(
                    A=A.detach().cpu().clone(),
                    B=B.detach().cpu().clone(),
                    As=As.detach().cpu().clone(),
                    Bs=Bs.detach().cpu().clone(),
                    output=out.detach().cpu().clone(),
                    profile=type(kernel).__module__ + "." + type(kernel).__name__,
                    use_triton=kernel.use_triton,
                    use_bpreshuffle=kernel.use_bpreshuffle,
                )
                torch.save(operands, Path("/outputs") / "decode0-shared-down-operands.pt")
                counts = []
                for _ in range(20):
                    again = original_scaled_mm(kernel, A, B, As, Bs)
                    counts.append(
                        int(
                            (
                                out.contiguous().view(torch.int16)
                                != again.contiguous().view(torch.int16)
                            ).sum()
                        )
                    )
                (Path("/outputs") / "decode0-shared-down-replay.json").write_text(
                    json.dumps(
                        dict(
                            scope="rollout self repeat, not actor comparison",
                            replays=20,
                            mismatch_counts=counts,
                            all_operands_identical=True,
                        )
                    )
                )
                print("NATIVE_DOWN_PROJ_REPLAY", counts, flush=True)
            return out

        AiterFp8BlockScaledMMKernel.apply_block_scaled_mm = capture_scaled_mm
        if os.environ.get("DSV4_FIXED_SHARED_DOWN") == "1":
            from rl_engine.integrations.engines.rollout.vllm.shared_down import bind_shared_down

            print("RLK_FIXED_SHARED_DOWN_BOUND", bind_shared_down(model, p1), flush=True)
            down_kernel = model.layers[
                model.start_layer
            ].ffn.shared_experts.down_proj.quant_method.fp8_linear
            current_down = down_kernel.apply_block_scaled_mm
            replay_done = [False]

            def paired_shared_down(kernel, A, B, As, Bs):
                result = current_down(A, B, As, Bs)
                if (
                    not replay_done[0]
                    and os.environ.get("DSV4_PHASE") == "generation"
                    and forward_index[0] == 1
                ):
                    replay_done[0] = True
                    legacy = original_scaled_mm(kernel, A, B, As, Bs)
                    old_counts = []
                    fixed_counts = []
                    for _ in range(20):
                        old_counts.append(
                            int(
                                (
                                    legacy.view(torch.int16)
                                    != original_scaled_mm(kernel, A, B, As, Bs).view(torch.int16)
                                ).sum()
                            )
                        )
                        fixed_counts.append(
                            int(
                                (
                                    result.view(torch.int16)
                                    != current_down(A, B, As, Bs).view(torch.int16)
                                ).sum()
                            )
                        )
                    torch.save(
                        dict(A=A.cpu(), B=B.cpu(), As=As.cpu(), Bs=Bs.cpu(), output=result.cpu()),
                        Path("/outputs/decode0-shared-down-operands.pt"),
                    )
                    record = dict(
                        scope="rollout self repeat, separate from actor comparison",
                        original_mismatch_counts=old_counts,
                        fixed_mismatch_counts=fixed_counts,
                        replays=20,
                        all_operands_identical=True,
                    )
                    Path("/outputs/decode0-shared-down-paired.json").write_text(
                        json.dumps(record, indent=2)
                    )
                    print("SHARED_DOWN_PAIRED_REPLAY", record, flush=True)
                return result

            down_kernel.apply_block_scaled_mm = types.MethodType(paired_shared_down, down_kernel)

        for i in range(model.start_layer, model.end_layer):
            attn = model.layers[i].attn
            runner = model.layers[i].ffn.experts
            original_select = runner.router.select_experts

            def select(*args, _original=original_select, _i=i, **kwargs):
                dump(f"layer.{_i}.ffn.router.select.input", kwargs)
                out = _original(*args, **kwargs)
                dump(f"layer.{_i}.ffn.router.select.output", out)
                return out

            runner.router.select_experts = select
            backend = runner.routed_experts.quant_method.moe_kernel.fused_experts
            old_apply = backend.apply.__func__
            native_matmul = old_apply.__globals__["matmul"]

            def matmul(*args, _original=native_matmul, _i=i, **kwargs):
                dump(
                    f"layer.{_i}.ffn.packed.matmul.input",
                    dict(x=args[0], gammas=kwargs.get("gammas"), weight_type=str(type(args[1]))),
                )
                out = _original(*args, **kwargs)
                dump(f"layer.{_i}.ffn.packed.matmul.output", out)
                return out

            copied = types.FunctionType(
                old_apply.__code__,
                dict(old_apply.__globals__, matmul=matmul),
                old_apply.__name__,
                old_apply.__defaults__,
                old_apply.__closure__,
            )
            copied.__kwdefaults__ = old_apply.__kwdefaults__
            backend.apply = types.MethodType(copied, backend)
            old_routing_data = backend._make_routing_data

            def routing_data(*args, _original=old_routing_data, _i=i, **kwargs):
                out = _original(*args, **kwargs)
                data = out[0] if isinstance(out, tuple) else out
                dump(
                    f"layer.{_i}.ffn.packed.routing_data",
                    {
                        k: getattr(data, k)
                        for k in ("gather_tok", "gate_scal", "scatter_tok", "n_valid")
                        if hasattr(data, k)
                    },
                )
                return out

            backend._make_routing_data = routing_data
            original_modular = runner.routed_experts.forward_modular

            def modular(*args, _original=original_modular, _i=i, **kwargs):
                dump(f"layer.{_i}.ffn.routed_modular.input", kwargs)
                out = _original(*args, **kwargs)
                dump(f"layer.{_i}.ffn.routed_modular.output", out)
                return out

            runner.routed_experts.forward_modular = modular
            from .native_sparse import bind_rollout_sparse

            bind_rollout_sparse(
                attn, observer=lambda name, value, i=i: dump(f"layer.{i}.attn.{name}", value)
            )
            original_o_proj = attn._o_proj.__func__
            original_inverse_group = original_o_proj.__globals__["rocm_inv_rope_einsum"]

            def trace_inverse_group(*args, _original=original_inverse_group, _i=i, **kwargs):
                out = _original(*args, **kwargs)
                dump(f"layer.{_i}.attn.wo_a.bf16_einsum.output", out)
                return out

            cloned_o_proj = types.FunctionType(
                original_o_proj.__code__,
                dict(original_o_proj.__globals__, rocm_inv_rope_einsum=trace_inverse_group),
                original_o_proj.__name__,
                original_o_proj.__defaults__,
                original_o_proj.__closure__,
            )
            cloned_o_proj.__kwdefaults__ = original_o_proj.__kwdefaults__
            attn._o_proj = types.MethodType(cloned_o_proj, attn)
            from .group_projection import bind_original_group_projection

            bind_original_group_projection(
                attn,
                self.model_config.model,
                i,
                observer=lambda name, value, i=i: dump(f"layer.{i}.attn.{name}", value),
            )
            from .native_decode import bind_rollout_decode

            bind_rollout_decode(
                attn,
                observer=lambda name, value, i=i: dump(f"layer.{i}.attn.{name}", value),
                record_if=lambda: os.environ.get("DSV4_PHASE") == "generation",
            )
            original_prefill = attn._forward_prefill.__func__
            sparse_prefill = original_prefill.__globals__["rocm_sparse_attn_prefill"]

            def trace_prefill(*args, _original=sparse_prefill, _i=i, **kwargs):
                dump(f"layer.{_i}.attn.sparse_prefill.input", kwargs)
                result = _original(*args, **kwargs)
                dump(f"layer.{_i}.attn.sparse_prefill.output", kwargs["output"])
                return result

            cloned = types.FunctionType(
                original_prefill.__code__,
                dict(original_prefill.__globals__, rocm_sparse_attn_prefill=trace_prefill),
                original_prefill.__name__,
                original_prefill.__defaults__,
                original_prefill.__closure__,
            )
            cloned.__kwdefaults__ = original_prefill.__kwdefaults__
            attn._forward_prefill = types.MethodType(cloned, attn)
            print(
                "RLK_ACTUAL_ATTENTION_DISPATCH",
                i,
                attn._use_aiter_sparse_mla_prefill,
                attn._use_aiter_sparse_mla_decode,
                flush=True,
            )
            for method_name in (
                "_fused_wqa_wkv_gemm",
                "_split_qkv_and_norm",
                "_wq_b_proj",
                "_o_proj",
                "forward_mqa",
            ):
                original_method = getattr(attn, method_name)

                def traced(*args, _original=original_method, _name=method_name, _i=i, **kwargs):
                    observed_args = args[:3] if _name == "forward_mqa" else args
                    dump(f"layer.{_i}.attn.method.{_name}.input", observed_args)
                    result = _original(*args, **kwargs)
                    dump(f"layer.{_i}.attn.method.{_name}", result)
                    if _name == "forward_mqa":
                        dump(
                            f"layer.{_i}.attn.method.{_name}.filled_output",
                            args[3] if len(args) > 3 else kwargs["output"],
                        )
                    return result

                setattr(attn, method_name, traced)
        # Capture the real production wo_a quantizer/GEMM, without changing values.
        import aiter.ops.batched_gemm_op_a8w8 as grouped_gemm
        import aiter.ops.inverse_rope_group_quant as inverse_quant

        original_grouped = grouped_gemm.batched_gemm_a8w8_mxscale
        original_inverse = inverse_quant.inverse_rope_group_quant

        def grouped(*args, **kwargs):
            result = original_grouped(*args, **kwargs)
            dump("wo_a.grouped_fp8_input", (args[0], args[2], args[3]))
            dump("wo_a.grouped_output", result)
            return result

        def inverse(*args, **kwargs):
            result = original_inverse(*args, **kwargs)
            dump("wo_a.inverse_quant_input", args[:2])
            dump("wo_a.inverse_quant_output", result)
            return result

        grouped_gemm.batched_gemm_a8w8_mxscale = grouped
        inverse_quant.inverse_rope_group_quant = inverse
        original = model.forward
        forward_signature = inspect.signature(original)

        def forward(*args, **kwargs):
            if os.environ.get("DSV4_PHASE") == "generation":
                forward_index[0] += 1
                from vllm.forward_context import get_forward_context

                metadata = get_forward_context().attn_metadata
                if not isinstance(metadata, dict):
                    raise ValueError("ordinary eager attention metadata is required")
                batch = next(m for m in metadata.values() if hasattr(m, "num_reqs"))
                starts = batch.query_start_loc.detach().cpu().tolist()
                tokens = (
                    forward_signature.bind(*args, **kwargs)
                    .arguments["input_ids"]
                    .detach()
                    .cpu()
                    .tolist()
                )
                self._rlk_case_batches.append(
                    dict(
                        forward=forward_index[0],
                        num_requests=int(batch.num_reqs),
                        query_start_loc=starts,
                        input_token_rows=[
                            tokens[a:b] for a, b in zip(starts[:-1], starts[1:], strict=True)
                        ],
                    )
                )
            dump("model.input", {"args": args, "kwargs": kwargs})
            with p1.request(version=p1.version):
                result = original(*args, **kwargs)
            dump("model.output", result)
            return result

        model.forward = forward
        self._dsv4_diagnostic_p1 = p1
        print("RLK_BOUND_AFTER_LOAD_BEFORE_WARMUP", p1.backend, p1.mode, p1.version, flush=True)

        print("ACTUAL_RAW_LOGPROB_MODE", self.model_runner.sampler.logprobs_mode, flush=True)
        from vllm.v1.worker.gpu.sample import logprob as lp

        raw_count = [0]

        def raw_logprobs(logits, token_ids):
            values = selected_raw_logprobs(wrapper, logits, token_ids, weight_version=p1.version)
            dump("logits", logits)
            dump("selected_raw_logprobs", values)
            if os.environ.get("DSV4_PHASE") == "generation":
                index = raw_count[0]
                raw_count[0] += 1
                (Path("/outputs/raw") / getattr(self, "_rlk_case_id", "unassigned")).mkdir(
                    parents=True, exist_ok=True
                )
                torch.save(
                    {
                        "logits": logits.detach().cpu(),
                        "token_ids": token_ids.detach().cpu(),
                        "raw_logprobs": values.detach().cpu(),
                        "mode": self.model_runner.sampler.logprobs_mode,
                        "weight_version": p1.version,
                    },
                    (Path("/outputs/raw") / getattr(self, "_rlk_case_id", "unassigned"))
                    / f"{index:05d}.pt",
                )
                print(
                    "RAW_LOGPROB_TENSOR",
                    index,
                    str(values.dtype),
                    list(values.shape),
                    inspect.getfile(selected_raw_logprobs),
                    flush=True,
                )
            return values

        lp.compute_token_logprobs = raw_logprobs
