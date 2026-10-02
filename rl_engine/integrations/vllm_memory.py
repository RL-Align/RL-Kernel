# SPDX-License-Identifier: Apache-2.0
"""Compatibility for vLLM releases that accidentally skip the weight pool."""
import ast
import inspect
import textwrap
from functools import wraps


def prepare_worker_ipc(worker):
    """Create resident IPC arenas before entering vLLM's offloadable pool."""
    import os

    import torch
    from vllm.distributed import get_tp_group

    from rl_engine.distributed.collectives import collective_for_group

    config = worker.vllm_config
    tp = get_tp_group()
    if tp.world_size > 1:
        collective_for_group(tp.device_group)
    cp = int(getattr(config.parallel_config, "prefill_context_parallel_size", 1))
    if cp > 1:
        from vllm.distributed import get_pcp_group

        group = get_pcp_group()
        block = int(config.cache_config.block_size)
        capacity = ((config.model_config.max_model_len + cp * block - 1) // (cp * block)) * block
        requests = min(
            config.scheduler_config.max_num_seqs,
            int(
                os.getenv(
                    "RL_KERNEL_VLLM_CUDAGRAPH_MAX_CAPTURE_SIZE",
                    str(config.scheduler_config.max_num_seqs),
                )
            ),
        )
        kv_heads = config.model_config.get_num_kv_heads(config.parallel_config)
        head_size = config.model_config.get_head_size()
        element_size = torch.empty((), dtype=config.model_config.dtype).element_size()
        collective_for_group(
            group.device_group,
            min_size_bytes=2 * requests * capacity * kv_heads * head_size * element_size,
        )


def patch_weight_pool(worker_class=None, prepare=None):
    """Repair only the known ``with pool and config`` load-model pattern.

    Entering the weight pool makes vLLM's ordinary sleep/wake mechanism own
    model storage. Fixed upstream versions and sleep-disabled workers retain
    their original load path.
    """
    if worker_class is None:
        from vllm.v1.worker.gpu_worker import Worker

        worker_class = Worker
    original = worker_class.load_model
    if getattr(original, "_rlk_weight_pool_fix", False):
        return False
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    except (OSError, TypeError, IndentationError, SyntaxError):
        return False
    broken = any(
        isinstance(node, ast.withitem)
        and isinstance(node.context_expr, ast.BoolOp)
        and isinstance(node.context_expr.op, ast.And)
        and any(
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "_maybe_get_memory_pool_context"
            for value in node.context_expr.values
        )
        for node in ast.walk(tree)
    )
    if not broken:
        return False

    @wraps(original)
    def load_model(self, *args, **kwargs):
        if not self.vllm_config.model_config.enable_sleep_mode:
            return original(self, *args, **kwargs)
        if prepare is not None:
            prepare(self)
        with self._maybe_get_memory_pool_context(tag="weights"):
            return original(self, *args, **kwargs)

    load_model._rlk_weight_pool_fix = True
    worker_class.load_model = load_model
    return True
