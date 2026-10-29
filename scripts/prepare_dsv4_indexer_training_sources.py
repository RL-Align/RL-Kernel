#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Explicitly provision hash-pinned source dependencies and validation references."""

import argparse
import importlib.util
from pathlib import Path

AIT = "326cffa38c9eda16bef19375ba55334e4d3acd59"
MLITE = "327a238239448de6771beea185dc6902a8de3eaa"
AIT_ROOT = f"https://raw.githubusercontent.com/ROCm/aiter/{AIT}/aiter/ops/triton"
MLITE_ROOT = (
    f"https://raw.githubusercontent.com/ISEEKYAN/Megatron-LM/{MLITE}/"
    "experimental/lite/megatron/lite"
)

SOURCES = (
    (
        "sparse_kernels.py",
        f"{AIT_ROOT}/_triton_kernels/attention/sparse_mla_dsv4_train.py",
        "56db3b46c58f8fb30912d4564189b214e2c9a37fa2138a155b45a4ce51b5c844",
        "RLK_AITER_DSV4_SPARSE_KERNEL_SOURCE",
    ),
    (
        "sparse_wrapper.py",
        f"{AIT_ROOT}/attention/sparse_mla_dsv4_train.py",
        "ede08814720b9c783c7375cab9f95ee38bf6212ca77ba23149e33309b72842ff",
        "RLK_AITER_DSV4_SPARSE_WRAPPER_SOURCE",
    ),
    (
        "indexer_reference.py",
        f"{AIT_ROOT}/attention/dsv4_indexer.py",
        "409e304e604c5e8b780a5d370fc138865e6f7fe92ed4c12554b2578078ced9af",
        "RLK_AITER_INDEXER_REFERENCE",
    ),
    (
        "mlite_backward.py",
        f"{MLITE_ROOT}/model/deepseek_v4/vllm/primitive/attention/backward.py",
        "6a1e78ca89b83f56391b5e5ac409e765338f1e3a625251d0d4c877c1e80b4a68",
        "RLK_MLITE_DSV4_BACKWARD_SOURCE",
    ),
    (
        "mlite_dense.py",
        f"{MLITE_ROOT}/model/deepseek_v4/vllm/primitive/dense.py",
        "cd3eb935b6783bae806859f0e810692cc13240e4ef2c04b93e0dcfdb6f8ba6c4",
        "RLK_MEGATRON_DENSE_SOURCE",
    ),
    (
        "mlite_dsa.py",
        f"{MLITE_ROOT}/primitive/kernels/dsa_kernels.py",
        "0df650e687fa0018dd1bc1dd1f0694d8ebd3ce6a61eb9466af58678dd235fdee",
        "RLK_MLITE_DSA_SOURCE",
    ),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    helper = Path(__file__).with_name("prepare_dsv4_sparse_provider.py")
    spec = importlib.util.spec_from_file_location("_rlk_source_provision", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for filename, url, digest, variable in SOURCES:
        module.provision(args.output_dir / filename, url, digest, variable)
    print("Also export TRITON_F32_DEFAULT=ieee BEFORE starting the training process.")
    print("This helper does not install or upgrade Torch, ROCm, aiter, vLLM or Megatron.")


if __name__ == "__main__":
    main()
