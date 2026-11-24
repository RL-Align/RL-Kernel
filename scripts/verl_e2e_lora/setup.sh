#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Host setup for the 8-GPU DSv4 LoRA comparison (README.md, section 1). Idempotent: a finished
# step is skipped on a rerun.
#
#   bash setup.sh rlkernel   # conda env: torch 2.13 / CUDA 13, vLLM fork, DeepGEMM, FlashMLA,
#                            # verl, RL-Kernel (source builds, ~2-3 h)
#   bash setup.sh baseline   # verl #7483's uv env + the Megatron-Bridge / Megatron-LM commits
#   bash setup.sh data       # checkpoint (~160 GB), DAPO / AIME parquet, the L$LAYERS view
#   bash setup.sh all        # the three, in this order
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh"
export MAX_JOBS=${MAX_JOBS:-32}
mkdir -p "$SRC_DIR" "$DATA_DIR" "$OUT_DIR" "$(dirname "$RLK_ENV")"

say() { printf '\n=== %s\n' "$*"; }

clone() { # url commit dir [submodules]
  local url=$1 commit=$2 dir=$3
  [[ -d $dir/.git ]] || git clone -q "$url" "$dir"
  git -C "$dir" fetch -q origin "$commit" 2>/dev/null || git -C "$dir" fetch -q origin
  git -C "$dir" checkout -q "$commit"
  [[ ${4:-} == submodules ]] && git -C "$dir" submodule update --init --recursive -q
  echo "$(basename "$dir") at $(git -C "$dir" rev-parse --short HEAD)"
}

preflight() {
  say preflight
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
  nvidia-smi topo -m | sed -n 1,10p   # not head: SIGPIPE under pipefail on a long table
  free -g | head -2
  df -h "$LORA_E2E_ROOT" | tail -1
  command -v conda >/dev/null || { echo "need conda (or mamba) on PATH"; exit 1; }
}

# A CUDA_HOME with the standard layout. The conda cuda-toolkit keeps headers and libraries
# under targets/x86_64-linux, which DeepGEMM's runtime JIT and the extension builds expect at
# include/ and lib64/.
cuda_home() {
  if [[ -n ${SYSTEM_CUDA_HOME:-} ]]; then echo "$SYSTEM_CUDA_HOME"; return; fi
  local h=$LORA_E2E_ROOT/envs/cuda130-home
  mkdir -p "$h"
  ln -sfn "$RLK_ENV/bin" "$h/bin"
  ln -sfn "$RLK_ENV/targets/x86_64-linux/include" "$h/include"
  ln -sfn "$RLK_ENV/targets/x86_64-linux/lib" "$h/lib"
  ln -sfn "$RLK_ENV/targets/x86_64-linux/lib" "$h/lib64"
  ln -sfn "$RLK_ENV/nvvm" "$h/nvvm"
  ln -sfn "$RLK_ENV/targets" "$h/targets"
  echo "$h"
}

setup_rlkernel() {
  say "RL-Kernel env at $RLK_ENV"
  if [[ ! -x $RLK_ENV/bin/python ]]; then
    if [[ -n ${SYSTEM_CUDA_HOME:-} ]]; then
      conda create -y -p "$RLK_ENV" -c conda-forge python=3.12 ninja
    else
      conda create -y -p "$RLK_ENV" -c nvidia -c conda-forge python=3.12 cuda-toolkit=13.0 ninja
    fi
  fi
  local py=$RLK_ENV/bin/python
  export CUDA_HOME
  CUDA_HOME=$(cuda_home)
  export PATH=$CUDA_HOME/bin:$RLK_ENV/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0
  export CPLUS_INCLUDE_PATH=$CUDA_HOME/include/cccl${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}
  nvcc --version | tail -1

  say "torch + the locked packages"
  $py -m pip install -q torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 \
    --index-url "$TORCH_INDEX"
  # The exact package set the 2-GPU run used, without the source-built pieces below.
  $py -m pip install -q --no-deps -r "$HERE/rlkernel-env.lock.txt" \
    --extra-index-url "$VERL_WHEELHOUSE"

  say "FlashMLA $FLASHMLA_COMMIT (RL-Kernel's pinned build; vLLM compiles the same sources)"
  clone "$FLASHMLA_REPO" "$FLASHMLA_COMMIT" "$SRC_DIR/FlashMLA" submodules
  if ! $py -c 'import flash_mla' 2>/dev/null; then
    (cd "$SRC_DIR/FlashMLA" && FLASH_MLA_DISABLE_SM100=1 \
      $py -m pip install -q --no-build-isolation --no-deps .)
  fi

  say "DeepGEMM $DEEPGEMM_COMMIT (+ the paged-MQA empty-batch fix)"
  clone "$DEEPGEMM_REPO" "$DEEPGEMM_COMMIT" "$SRC_DIR/DeepGEMM" submodules
  # Build from the clean tree: on a dirty one the version becomes 2.6.1+local, which
  # rl_engine/kernels/ops/cuda/matmul/deepgemm.py refuses. The fix goes into the installed headers,
  # which are what DeepGEMM's runtime JIT compiles.
  git -C "$SRC_DIR/DeepGEMM" checkout -q -- .
  # setup.py imports its namespace package `scripts`; once verl is installed (editable), verl's
  # regular `scripts` package shadows it. An __init__.py, excluded from git status, fixes that.
  touch "$SRC_DIR/DeepGEMM/scripts/__init__.py"
  grep -qx scripts/__init__.py "$SRC_DIR/DeepGEMM/.git/info/exclude" ||
    echo scripts/__init__.py >> "$SRC_DIR/DeepGEMM/.git/info/exclude"
  if [[ $($py -c 'import importlib.metadata as m; print(m.version("deep_gemm"))' \
          2>/dev/null) != 2.6.1+e21c821 ]]; then
    (cd "$SRC_DIR/DeepGEMM" && $py -m pip install -q --no-build-isolation --no-deps .)
  fi
  local patch=$HERE/deepgemm-sm90-paged-mqa-empty.patch site
  site=$($py -c 'import importlib.util as u, os
print(os.path.dirname(list(u.find_spec("deep_gemm").submodule_search_locations)[0]))')
  if patch -p1 -d "$site" -N --dry-run -s < "$patch" > /dev/null 2>&1; then
    patch -p1 -d "$site" -N -s < "$patch"
  fi
  patch -p1 -d "$site" -R --dry-run -s < "$patch" > /dev/null  # fails if the fix is missing

  say "vLLM fork $VLLM_FORK_COMMIT (source build, 1-2 h)"
  clone "$VLLM_FORK_REPO" "$VLLM_FORK_COMMIT" "$SRC_DIR/vllm"
  if ! $py -c 'import vllm' 2>/dev/null; then
    $py -m pip install -q "cmake>=3.26.1" "setuptools>=77.0.3,<81.0.0" "setuptools-scm>=8.0" \
      "setuptools-rust>=1.9.0" jinja2 "packaging>=24.2" wheel
    (cd "$SRC_DIR/vllm" && TORCH_CUDA_ARCH_LIST=9.0a NVCC_THREADS=1 VLLM_TARGET_DEVICE=cuda \
      FLASH_MLA_SRC_DIR=$SRC_DIR/FlashMLA \
      $py -m pip install --no-build-isolation --no-deps -e . > "$SRC_DIR/vllm-build.log" 2>&1) ||
      { tail -30 "$SRC_DIR/vllm-build.log"; exit 1; }
  fi

  say "vLLM's DS4 deterministic top-k (VLLM_BATCH_INVARIANT=1 loads it from DS4_BI_TOPK_LIB)"
  # Not part of vLLM's build. The rollout needs it once a C4 layer holds more than
  # index_topk = 512 entries (sequences past ~2048 tokens); RL-Kernel's stable top-k gives
  # the same bytes (scripts/check_stable_topk_vllm_parity.py).
  if [[ ! -f $DS4_BI_TOPK_LIB ]]; then
    $py - "$SRC_DIR/vllm/csrc/ds4_bi_topk_extension.cu" "$(dirname "$DS4_BI_TOPK_LIB")" <<'PY'
import os, sys
from torch.utils.cpp_extension import load
os.makedirs(sys.argv[2], exist_ok=True)
load(name="ds4_bi_topk", sources=[sys.argv[1]], is_python_module=False,
     build_directory=sys.argv[2],
     extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])
PY
  fi
  ls -l "$DS4_BI_TOPK_LIB"

  say "verl $VERL_COMMIT (editable, no deps)"
  clone "$VERL_REPO" "$VERL_COMMIT" "$VERL_DIR"
  $py -m pip install -q --no-deps -e "$VERL_DIR"

  say "RL-Kernel (editable: builds _C with the routed-expert kernels, registers the vLLM plugin)"
  (cd "$RLK_ROOT" && KERNEL_ALIGN_MOE_SM90=1 RL_KERNEL_REQUIRE_EXT=1 \
    $py -m pip install -q --no-build-isolation --no-deps -e .)

  say "RL-Kernel env check"
  (cd /tmp && $py - <<'EOF'
import importlib.metadata as m
import torch, vllm, verl, deep_gemm, flash_mla
import rl_engine._C as c
assert hasattr(c, "sm90_moe_fc1_swiglu_quant_forward"), "_C built without KERNEL_ALIGN_MOE_SM90=1"
eps = [e.value for e in m.entry_points(group="vllm.general_plugins")]
assert any(v.startswith("rl_engine.") for v in eps), f"RL-Kernel vLLM plugin not registered: {eps}"
print("torch", torch.__version__, torch.version.cuda, "| vllm", vllm.__version__,
      "| deep_gemm", m.version("deep_gemm"), "| flash_mla", m.version("flash_mla"),
      "| gpus", torch.cuda.device_count(), torch.cuda.get_device_name(0))
print("rlkernel env ok")
EOF
  )
}

setup_baseline() {
  say "baseline env: verl #7483 uv env"
  command -v uv >/dev/null || { echo "need uv (https://docs.astral.sh/uv/)"; exit 1; }
  clone "$VERL_REPO" "$VERL_COMMIT" "$VERL_DIR"
  (cd "$VERL_DIR" && uv sync --frozen --all-packages --extra vllm --extra megatron)
  local py=$BASELINE_VENV/bin/python
  say "Megatron-Bridge $BRIDGE_COMMIT and Megatron-LM $MEGATRON_COMMIT (the launcher's pins)"
  # Over the lock's Bridge 0.6.2 / core 0.19.2, without deps (as the lock installs Bridge).
  # Run with VERL_USE_UV=0 from now on: `uv run --frozen` would put the lock's versions back.
  uv pip install --python "$py" --no-deps "git+$BRIDGE_REPO@$BRIDGE_COMMIT"
  uv pip install --python "$py" --no-deps "git+$MEGATRON_REPO@$MEGATRON_COMMIT"
  # flash-attn 2.8.3's FA4 preview (flash_attn.cute) fails to import against vLLM 0.32's pinned
  # cutlass-dsl 4.8.0 with an AttributeError, which Megatron's `except ImportError` around its
  # FA4 import does not catch, so `import megatron.bridge` fails. DSv4 runs no FA4: set the
  # package aside so Megatron sees FA4 as absent.
  local fa
  fa=$($py -c 'import importlib.util as u
print(u.find_spec("flash_attn").submodule_search_locations[0])')
  if [[ -d $fa/cute ]]; then mv "$fa/cute" "$fa/cute.off-cutlass-dsl-4.8"; fi
  # This Bridge commit wants transformers < 5.9; vLLM 0.32 needs >= 5.16.1, which has its own
  # qwen3_asr, and Bridge's registration of it (a model DSv4 does not touch) then raises.
  local asr
  asr=$(dirname "$fa")/megatron/bridge/models/qwen3_asr/hf_qwen3_asr/__init__.py
  sed -i -E 's/^(Auto(Config|Model|Processor)\.register\(.*)\)$/\1, exist_ok=True)/' "$asr"
  # On a host with glibc < 2.38 (RHEL 8 / 9), TransformerEngine, apex, DeepEP and flash-attn
  # wheels do not load: set CONTAINER to an Ubuntu 24.04 CUDA 13 image (README 1).
  local run=()
  if [[ -n ${CONTAINER:-} ]]; then
    # --nv puts the host driver's libcuda first; the compat one has to go before it.
    local libs=${CUDA_COMPAT_DIR:+$CUDA_COMPAT_DIR:}/.singularity.d/libs
    run=(env "SINGULARITYENV_LD_LIBRARY_PATH=$libs"
         singularity exec --nv -B "$LORA_E2E_ROOT" -B "$RLK_ROOT" "$CONTAINER")
  fi
  say "FlashMLA nv_dev $BL_FLASHMLA_COMMIT and fast-hadamard-transform (fused DSA kernels)"
  clone "$BL_FLASHMLA_REPO" "$BL_FLASHMLA_COMMIT" "$SRC_DIR/FlashMLA-nv_dev" submodules
  clone "$HADAMARD_REPO" "$HADAMARD_COMMIT" "$SRC_DIR/fast-hadamard-transform"
  if ! (cd /tmp && "${run[@]}" $py -c 'import flash_mla, fast_hadamard_transform') 2>/dev/null
  then
    local cuda=${CONTAINER:+/usr/local/cuda}
    cuda=${cuda:-$CUDA_HOME}
    # The CUDA container has no git; FlashMLA's setup.py only runs `git submodule update`
    # (done above) and `git rev-parse --short HEAD`.
    local stub=$LORA_E2E_ROOT/tools/gitstub d
    mkdir -p "$stub"
    printf '#!/bin/sh\n[ "$*" = "rev-parse --short HEAD" ] && echo nv_dev\nexit 0\n' \
      > "$stub/git"
    chmod +x "$stub/git"
    for d in FlashMLA-nv_dev fast-hadamard-transform; do
      (cd "$SRC_DIR/$d" && "${run[@]}" env CUDA_HOME="$cuda" PATH="$stub:$cuda/bin:$PATH" \
         CPATH="$cuda/targets/x86_64-linux/include/cccl" TORCH_CUDA_ARCH_LIST=9.0 \
         FLASH_MLA_DISABLE_SM100=1 FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE \
         MAX_JOBS=32 NVCC_THREADS=8 \
         uv pip install --python "$py" --no-build-isolation --no-deps .)
    done
  fi
  (cd /tmp && "${run[@]}" $py - <<'EOF'
import importlib.metadata as m
import torch, vllm, verl, megatron.core, megatron.bridge, flash_mla, fast_hadamard_transform
print("torch", torch.__version__, "| vllm", vllm.__version__,
      "| megatron-core", m.version("megatron-core"),
      "| megatron-bridge", m.version("megatron-bridge"))
print("baseline env ok")
EOF
  )
}

setup_data() {
  local py=${PY:-$RLK_ENV/bin/python}
  say "checkpoint $MODEL_REPO -> $MODEL_DIR"
  HF_HUB_ENABLE_HF_TRANSFER=0 $py - "$MODEL_REPO" "$MODEL_DIR" <<'EOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2], max_workers=16)
EOF
  $py - "$MODEL_DIR" <<'EOF'
import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
want = set(json.load(open(d / "model.safetensors.index.json"))["weight_map"].values())
missing = sorted(f for f in want if not (d / f).exists())
print("missing shards", missing)
sys.exit(1 if missing else 0)
EOF

  say "data -> $DATA_DIR"
  local f
  f=$DATA_DIR/dapo-math-17k.parquet; [[ -s $f ]] || curl -fL -o "$f" "$TRAIN_URL"
  f=$DATA_DIR/aime-2024.parquet; [[ -s $f ]] || curl -fL -o "$f" "$VAL_URL"
  ln -sfn dapo-math-17k.parquet "$DATA_DIR/train.parquet"
  ln -sfn aime-2024.parquet "$DATA_DIR/test.parquet"

  say "first $LAYERS layers -> $MODEL_CUT"
  # Symlinks to the checkpoint plus an edited config. No MTP layer (neither run uses it).
  mkdir -p "$MODEL_CUT"
  for f in "$MODEL_DIR"/*; do
    [[ $(basename "$f") == config.json ]] || ln -sfn "$f" "$MODEL_CUT/$(basename "$f")"
  done
  $py - "$MODEL_DIR/config.json" "$MODEL_CUT/config.json" "$LAYERS" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))
c["num_hidden_layers"] = int(sys.argv[3])
c["num_nextn_predict_layers"] = 0
json.dump(c, open(sys.argv[2], "w"), indent=2)
print("layer types (compress ratio):", c["compress_ratios"][: c["num_hidden_layers"]])
EOF
}

case ${1:-} in
  rlkernel) preflight; setup_rlkernel ;;
  baseline) preflight; setup_baseline ;;
  data) setup_data ;;
  all) preflight; setup_rlkernel; setup_baseline; setup_data ;;
  *) echo "usage: setup.sh rlkernel|baseline|data|all"; exit 64 ;;
esac
say "setup ${1} done"
