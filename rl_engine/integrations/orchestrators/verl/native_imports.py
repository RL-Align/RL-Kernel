# SPDX-License-Identifier: Apache-2.0
"""Opt-in, process-local import ABI adapters; dependency files stay pristine.

The two narrowly targeted hooks run before the real verl/Bridge imports. They do
not replace model computation, installed files, or unrelated engine registrations.
"""

import importlib.abc
import importlib.machinery
import sys
from contextlib import contextmanager
from functools import wraps


@contextmanager
def _gemma_checkpoint_alias():
    from megatron.core.extensions.transformer_engine import te_checkpoint
    from megatron.core.transformer.moe import moe_layer

    existed = hasattr(moe_layer, "te_checkpoint")
    if not existed:
        moe_layer.te_checkpoint = te_checkpoint
    try:
        yield
    finally:
        if not existed:
            del moe_layer.te_checkpoint


def _guard_legacy_checkpoint(module):
    import megatron.core
    import torch
    from packaging.version import parse

    original = module.apply_patch_megatron_v012_with_torch_v28_v29

    @wraps(original)
    def guarded():
        # Match upstream's own intended version guard, before its removed imports.
        if (
            parse(torch.__version__).base_version not in ("2.8.0", "2.9.0")
            or parse(megatron.core.__version__).base_version != "0.12.1"
        ):
            return
        return original()

    module.apply_patch_megatron_v012_with_torch_v28_v29 = guarded


class _Loader:
    def __init__(self, loader, name):
        self.loader, self.name = loader, name

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        if self.name.endswith("modeling_gemma4"):
            with _gemma_checkpoint_alias():
                self.loader.exec_module(module)
        else:
            self.loader.exec_module(module)
            if self.name == "verl.models.mcore.patch":
                _guard_legacy_checkpoint(module)
            else:
                classes = tuple(
                    c for c in module.ALL_MODULE_WRAPPER_CLASSNAMES if isinstance(c, type)
                )
                original = module.unwrap_model

                @wraps(original)
                def unwrap(model, module_instances=classes):
                    return original(model, module_instances=module_instances)

                module.ALL_MODULE_WRAPPER_CLASSNAMES = classes
                module.unwrap_model = unwrap

    def __getattr__(self, name):
        return getattr(self.loader, name)


class _Finder(importlib.abc.MetaPathFinder):
    names = frozenset(
        (
            "megatron.bridge.models.gemma.modeling_gemma4",
            "verl.models.mcore.patch",
            "verl.utils.megatron_utils",
        )
    )

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.names:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot find locked dependency module {fullname}")
        spec.loader = _Loader(spec.loader, fullname)
        return spec


def install_native_import_adapters():
    """Call before importing the real TrainingWorker in an opt-in DSV4 process."""
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
