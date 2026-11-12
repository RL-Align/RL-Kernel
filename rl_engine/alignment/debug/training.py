# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Native backward/optimizer evidence without replacing the optimizer or autograd."""

from __future__ import annotations

from typing import Callable, Mapping

import torch

from .events import EventCapture


class TrainingCapture:
    """Compare diagnostic training arms using the existing training/rollout sides.

    ``backward`` delegates once to native autograd, preserving accumulation.
    Ordinary ``optimizer.step()`` is observed by removable pre/post hooks. This
    records parameter gradients and actual optimizer state, not hidden backward
    kernel internals. No zero_grad, clipping, loss scaling or step is synthesized.

    Optional state/gradient readers can return TensorShard views for distributed
    optimizers, but must expose actual weights, master weights, moments and
    gradients. Optimizers without PyTorch step hooks need a native adapter.
    """

    @staticmethod
    def prepare(directory, *, scope: str, ranks=None):
        EventCapture.prepare(
            directory,
            [
                {"id": stage, "module": "training", "stage": stage}
                for stage in (
                    "state.initial",
                    "backward.input",
                    "backward.output",
                    "step.input",
                    "step.output",
                    "state.final",
                )
            ],
            scope=scope,
            ranks=ranks,
            coverage={
                "execution": "native autograd and optimizer steps",
                "claim": "loss, parameter gradients, weights, buffers and exposed optimizer state",
            },
        )

    def __init__(
        self,
        directory,
        side,
        identity,
        model,
        optimizer,
        *,
        rank=0,
        state: Callable[[], Mapping] | None = None,
        gradients: Callable[[], Mapping] | None = None,
    ):
        if not all(
            callable(getattr(optimizer, name, None))
            for name in ("register_step_pre_hook", "register_step_post_hook")
        ):
            raise TypeError("optimizer must expose native step pre/post hooks")
        self.capture = EventCapture(directory, side, identity, rank=rank)
        self.model, self.optimizer = model, optimizer
        self.parameters = dict(model.named_parameters())
        self.names = {id(param): name for name, param in self.parameters.items()}
        self.state_reader = state or self._state
        self.gradient_reader = gradients or self._gradients
        self.handles = []
        self.steps = self.microbatches = 0
        self.pending = None
        self.step_depth = 0
        self.opened = False

    def _gradients(self):
        return {name: param.grad for name, param in self.parameters.items()}

    def _state(self):
        groups, states = [], {}
        for group in self.optimizer.param_groups:
            names = []
            for param in group["params"]:
                if id(param) not in self.names:
                    raise ValueError("optimizer/master parameter requires an explicit state reader")
                name = self.names[id(param)]
                names.append(name)
                states[name] = self.optimizer.state.get(param, {})
            groups.append({**{k: v for k, v in group.items() if k != "params"}, "params": names})
        return {
            "model": self.model.state_dict(),
            "optimizer": {"groups": groups, "state": states},
        }

    def __enter__(self):
        self.capture.__enter__()
        self.opened = True
        try:
            self.capture.record("state.initial", self.state_reader(), frame="training")
            self.handles.append(self.optimizer.register_step_pre_hook(self._before_step))
            self.handles.append(self.optimizer.register_step_post_hook(self._after_step))
            return self
        except Exception as exc:
            self.__exit__(type(exc), exc, exc.__traceback__)
            raise

    def backward(self, loss: torch.Tensor, gradient=None, **kwargs):
        if not self.opened:
            raise RuntimeError("open the training capture before backward")
        frame = f"step:{self.steps}.backward:{self.microbatches}"
        try:
            self.capture.record(
                "backward.input",
                {"loss": loss, "gradient": gradient, "accumulated": self.gradient_reader()},
                frame=frame,
                contracts={
                    "create_graph": kwargs.get("create_graph", False),
                    "retain_graph": kwargs.get("retain_graph"),
                },
                route="torch.autograd.backward",
            )
            torch.autograd.backward(loss, grad_tensors=gradient, **kwargs)
            # Equal loss and incoming gradients do not prove equality of the
            # forward graph's saved tensors. Keep input evidence UNKNOWN.
            self.capture.record(
                "backward.output",
                self.gradient_reader(),
                frame=frame,
                route="torch.autograd.backward",
            )
            self.microbatches += 1
        except Exception as exc:
            self.capture.errors.append(f"native backward failed: {exc}")
            raise

    def _before_step(self, optimizer, args, kwargs):
        try:
            # PyTorch wraps both an overridden step and super().step(). Observe
            # the outer call, including updates performed after the super call.
            if self.step_depth:
                self.step_depth += 1
                return
            if self.pending is not None or not self.microbatches:
                raise ValueError("optimizer step requires observed backward and no unfinished step")
            self.pending = self.capture.record(
                "step.input",
                {"state": self.state_reader(), "gradients": self.gradient_reader()},
                frame=f"step:{self.steps}",
                route=type(optimizer).__qualname__,
            )
            self.step_depth = 1
        except Exception as exc:
            self.capture.errors.append(f"optimizer pre-step failed: {exc}")
            raise

    def _after_step(self, optimizer, args, kwargs):
        try:
            if self.step_depth > 1:
                self.step_depth -= 1
                return
            if self.pending is None:
                raise ValueError("optimizer post-step has no matching pre-step")
            self.capture.record(
                "step.output",
                {"state": self.state_reader(), "gradients": self.gradient_reader()},
                frame=f"step:{self.steps}",
                inputs=[self.pending],
                route=type(optimizer).__qualname__,
            )
            self.steps += 1
            self.microbatches = 0
            self.pending = None
            self.step_depth = 0
        except Exception as exc:
            self.capture.errors.append(f"optimizer post-step failed: {exc}")
            raise

    def __exit__(self, exc_type, exc, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.opened = False
        try:
            if self.pending is not None or self.microbatches or not self.steps:
                self.capture.errors.append("unfinished backward/optimizer update")
            self.capture.record("state.final", self.state_reader(), frame="training")
        except Exception as error:
            self.capture.errors.append(f"training state verification failed: {error}")
        return self.capture.__exit__(exc_type, exc, traceback)
