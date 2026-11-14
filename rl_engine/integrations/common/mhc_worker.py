# SPDX-License-Identifier: Apache-2.0
"""Per-worker P1 binding. The orchestrator owns communication and version acknowledgments."""
from contextlib import contextmanager
from threading import RLock, local

import torch

from rl_engine.ops.mhc.post_pre import post_pre
from rl_engine.ops.mhc.training import EPS, PROFILE, mhc_head, mhc_post, mhc_pre
from rl_engine.ops.norm.rmsnorm_residual import rmsnorm_residual


class P1Worker:
    def __init__(self, framework, role, *, backend, mode="split", version=0, observer=None):
        if framework not in ("verl", "vime") or role not in ("training", "rollout"):
            raise ValueError("unknown framework or role")
        if backend not in ("reference", "rocm") or mode not in ("split", "fused"):
            raise ValueError("unsupported P1 profile")
        if mode == "fused" and backend != "rocm":
            raise ValueError("fused post/pre requires the owned ROCm profile")
        if type(version) is not int or version < 0:
            raise ValueError("invalid initial weight version")
        self.framework, self.role, self.backend, self.mode = framework, role, backend, mode
        self.version, self.observer = version, observer
        self._lock = RLock()
        self._ready = True
        self._scope = local()
        self._updating = False
        self._generation = 0

    def _require(self, version):
        if not self._ready or self.version != version:
            raise RuntimeError("P1 weights unavailable or stale")

    @contextmanager
    def request(self, version):
        """Hold across the entire engine call, including explicit graph replay."""
        with self._lock:
            self._require(version)
            self._scope.depth = getattr(self._scope, "depth", 0) + 1
            try:
                yield self
            finally:
                self._scope.depth -= 1

    @contextmanager
    def updating(self, version, synchronize):
        """Drain work, mutate/repack weights, wait for copies, then acknowledge v+1.

        A failed update invalidates the worker; a retry must reload the complete
        version. The callback must synchronize all local streams participating in
        the update, including any work enqueued by an earlier request.
        """
        with self._lock:
            if type(version) is not int or version <= self.version:
                raise ValueError("weights must advance")
            if getattr(self._scope, "depth", 0) or self._updating:
                raise RuntimeError("cannot nest an update inside a request or another update")
            self._updating = True
            self._ready = False
            self._generation += 1
            try:
                synchronize()
                yield
                synchronize()
                self.version = version
                self._ready = True
            finally:
                self._updating = False

    def _active(self):
        if not getattr(self._scope, "depth", 0):
            raise RuntimeError("run model calls inside worker.request(weight_version)")
        self._require(self.version)

    def _record(self, name, value):
        if self.observer is not None:
            self.observer(
                self.framework, self.role, self.version, self.backend, PROFILE, name, value.detach()
            )
        return value

    def check_config(self, rms_eps, hc_eps, post_multiplier=2.0):
        if (rms_eps, hc_eps, post_multiplier) != (EPS, EPS, 2.0):
            raise ValueError("unsupported DSV4 controller constants")

    def pre(self, x, fn, scale, base, rounds):
        self._active()
        if x.ndim == 2:
            x = x[:, None, :].expand(-1, 4, -1)
        p, b, h = mhc_pre(
            x, fn, scale, base, backend=self.backend, recompute=True, sinkhorn_repeat=rounds
        )
        for n, v in (("post_mix", p), ("comb_mix", b), ("layer_input", h)):
            self._record(n, v)
        return x, p, b, h

    def norm(self, h, gamma):
        self._active()
        # Training may store the gain in BF16. Promote at the boundary without
        # caching: every optimizer update is visible and autograd retains the cast.
        if gamma.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("expected BF16/FP32 norm weight")
        y, _ = rmsnorm_residual(
            h,
            gamma.float(),
            backend="reference" if self.backend == "reference" else "rocm",
            recompute=True,
        )
        return self._record("normalized", y)

    def post(self, y, x, p, b):
        self._active()
        return self._record("residual", mhc_post(y, x, p, b, backend=self.backend))

    def adjacent(self, y, x, p, b, fn, s, base, rounds):
        self._active()
        out = post_pre(
            y, x, p, b, fn, s, base, backend=self.backend, mode=self.mode, sinkhorn_repeat=rounds
        )
        for n, v in zip(("residual", "post_mix", "comb_mix", "layer_input"), out, strict=True):
            self._record(n, v)
        return out

    def head(self, x, fn, s, base):
        self._active()
        return self._record("head", mhc_head(x, fn, s, base, backend=self.backend, recompute=True))

    def capture(self, fn, *, version, device):
        """Capture static compute. Training closures must reset grads and exclude the optimizer."""
        if self.observer is not None or self.backend == "reference":
            raise ValueError("capture requires a ROCm worker with tracing off")
        policy = torch.enable_grad if self.role == "training" else torch.inference_mode
        with self.request(version), torch.cuda.device(device), policy():
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                fn()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                outputs = fn()
            generation = self._generation

        def replay():
            with self.request(version), torch.cuda.device(device):
                if generation != self._generation:
                    raise RuntimeError("P1 graph invalidated by weight update")
                graph.replay()
                return outputs

        return replay
