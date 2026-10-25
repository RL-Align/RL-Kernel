"""Isolate reciprocal-square-root differences with shared FP32 input variances.

Run in this directory using /venv/main/bin/python probe.py. No fast math.
FP64 correctness here evaluates pow on the same FP32 inputs, not the whole model.
"""
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def probe(X, Y, N: tl.constexpr, MODE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, other=1.0)
    if MODE == 0:
        y = libdevice.pow(x, -0.5)
    elif MODE == 1:
        y = libdevice.rsqrt(x)
    elif MODE == 2:
        y = tl.inline_asm_elementwise(
            "{ .reg .f32 s; sqrt.rn.f32 s, $1; div.rn.f32 $0, 0f3F800000, s; }",
            constraints="=f,f", args=[x], dtype=tl.float32, is_pure=True, pack=1)
    else:
        y = tl.rsqrt(x)
    tl.store(Y + i, y, i < N)


def bits(x):
    return x.contiguous().view(torch.int32 if x.dtype == torch.float32 else torch.int16)


def compare(x, ref):
    return {"mismatch": int((bits(x) != bits(ref)).sum()), "count": x.numel()}


def main():
    root = Path(__file__).resolve().parent
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    flags = ["-std=c++17", "-O3", "-arch=sm_89", "--fmad=false", "--ftz=false",
             "--prec-div=true", "--prec-sqrt=true"]
    cmd = ["/usr/local/cuda/bin/nvcc", *flags, "-shared", "-Xcompiler=-fPIC",
           str(root / "probe.cu"), "-o", str(root / "probe.so")]
    subprocess.run(cmd, check=True)
    subprocess.run(["/usr/local/cuda/bin/nvcc", *flags, "--ptx", str(root / "probe.cu"),
                    "-o", str(root / "cuda.ptx")], check=True)
    library = ctypes.CDLL(str(root / "probe.so"))
    library.launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                              ctypes.c_int, ctypes.c_void_p]
    library.launch.restype = ctypes.c_int
    saved_ptx = set()

    def candidates(v):
        outputs = {"torch_pow": torch.pow(v, -0.5), "torch_rsqrt": torch.rsqrt(v)}
        for mode, name in enumerate(("powf", "rsqrtf", "sqrt_div_diagnostic")):
            out = torch.empty_like(v)
            err = library.launch(v.data_ptr(), out.data_ptr(), v.numel(), mode,
                                 torch.cuda.current_stream().cuda_stream)
            if err:
                raise RuntimeError(f"CUDA launch failed: {err}")
            outputs["cuda_" + name] = out
        for mode, name in enumerate(("pow", "rsqrt", "sqrt_div_diagnostic", "tl_rsqrt")):
            out = torch.empty_like(v)
            compiled = probe[(triton.cdiv(v.numel(), 256),)](
                v, out, v.numel(), mode, 256, enable_fp_fusion=False, enable_reflect_ftz=False)
            if name not in saved_ptx:
                (root / ("triton_" + name + ".ptx")).write_text(compiled.asm["ptx"])
                saved_ptx.add(name)
            outputs["triton_" + name] = out
        torch.cuda.synchronize()
        return outputs

    gen = torch.Generator(device="cuda").manual_seed(501)
    # Broad positive normal range; separate from actual RMSNorm variance domain.
    raw = torch.exp2(torch.rand(1048576, device="cuda", generator=gen) * 240 - 120)
    eps = torch.tensor([1e-5], device="cuda")
    boundaries = torch.cat([eps, torch.nextafter(eps, torch.zeros_like(eps)),
                            torch.nextafter(eps, torch.full_like(eps, float("inf"))),
                            torch.tensor([1., 2., 4., 16.], device="cuda")])
    # BF16 input rows spanning near-zero through large but finite squares.
    rows = 65536
    x = torch.randn(rows, 128, generator=gen, device="cuda")
    scales = torch.exp2(torch.linspace(-20, 20, rows, device="cuda"))
    x = (x * scales[:, None]).bfloat16()
    x[:4] = 0
    x[4:8] = 1
    x[8:12] = torch.finfo(torch.bfloat16).tiny
    a = x.float() * x.float()
    for half in (64, 32, 16, 8, 4, 2, 1):
        a = a[:, :half] + a[:, half:]
    v = (a[:, 0] * (1. / 128) + 1e-5).contiguous()

    report = {"environment": {"gpu": torch.cuda.get_device_name(),
                "capability": torch.cuda.get_device_capability(), "torch": torch.__version__,
                "torch_git": torch.version.git_version, "cuda": torch.version.cuda,
                "triton": triton.__version__, "nvcc_flags": flags,
                "triton_fp_fusion": False, "triton_reflect_ftz": False},
              "seed": 501, "datasets": {}}
    for label, values in (("positive_normal_sweep", raw), ("boundaries", boundaries),
                          ("bf16_fixed_tree_variances", v)):
        outputs = candidates(values)
        ref = outputs["torch_pow"]
        exact = values.double().pow(-0.5)
        result = {"count": values.numel(), "min": float(values.min()),
                  "max": float(values.max()), "vs_torch_pow": {}, "backend_pairs": {}}
        for name, out in outputs.items():
            stats = compare(out, ref)
            stats["max_ulp_vs_torch"] = int((bits(out).long() - bits(ref).long()).abs().max())
            stats["max_relative_error_fp64"] = float(((out.double() - exact).abs() / exact).max())
            bad = torch.nonzero(bits(out) != bits(ref)).flatten()[:5]
            stats["examples"] = [{"input": float(values[j]), "reference": float(ref[j]),
                                  "candidate": float(out[j])} for j in bad]
            if label == "bf16_fixed_tree_variances":
                baseline = (x.float() * ref[:, None]).bfloat16()
                candidate = (x.float() * out[:, None]).bfloat16()
                stats["bf16_norm"] = compare(candidate, baseline)
                stats["bf16_q_scale"] = compare((candidate.float()*3.87).bfloat16(),
                                                 (baseline.float()*3.87).bfloat16())
            result["vs_torch_pow"][name] = stats
        for left, right in (("cuda_powf", "triton_pow"), ("cuda_rsqrtf", "triton_rsqrt"),
                            ("cuda_rsqrtf", "triton_tl_rsqrt"),
                            ("cuda_sqrt_div_diagnostic", "triton_sqrt_div_diagnostic")):
            result["backend_pairs"][left + "__" + right] = compare(outputs[left], outputs[right])
        report["datasets"][label] = result
        print(label, {name: stats["mismatch"] for name, stats in result["vs_torch_pow"].items()}, flush=True)
    report["source_hashes"] = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                               for name in ("probe.py", "probe.cu")}
    (root / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print("RESULTS", root / "results.json", flush=True)


if __name__ == "__main__":
    main()
