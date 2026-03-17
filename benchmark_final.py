import argparse
import csv
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
import triton
import torch._functorch.config

# import Triton implementation and helpers
from fused_attention import (
    attention as triton_attention,
    DEVICE,
    is_hopper,
    is_blackwell,
)

try:
    from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func as flash_attn_func
    HAS_FLASH = True
except ImportError:
    HAS_FLASH = False

TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")

torch._functorch.config.donated_buffer = False

# def naive_attention(q, k, v, causal):
#     S = q @ k.transpose(-2, -1)
#     if causal:
#         N_CTX = q.shape[-2]
#         mask = torch.tril(torch.ones((N_CTX, N_CTX), device=q.device))
#         S = S.masked_fill(mask == 0, float("-inf"))
#     P = torch.softmax(S, dim=-1)
#     o = P @ v
#     return o

def naive_attention(q, k, v, causal, sm_scale=None):
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])

    S = (q @ k.transpose(-2, -1)) * sm_scale
    if causal:
        N_CTX = q.shape[-2]
        mask = torch.tril(torch.ones((N_CTX, N_CTX), device=q.device))
        S = S.masked_fill(mask == 0, float("-inf"))
    P = torch.softmax(S, dim=-1)
    o = P @ v
    return o

compiled_attention = torch.compile(naive_attention)

BATCH, N_HEADS = 4, 32
configs = []

for HEAD_DIM in [64]:
    for mode in ["fwd", "bwd"]:
        for causal in [True, False]:
            enable_ws = mode == "fwd" and (is_blackwell() or (is_hopper() and not causal))
            for warp_specialize in [False, True] if enable_ws else [False]:

                line_vals = ["triton-fp16", "naive", "torch.compile"]
                line_names = ["Triton [FP16]", "Naive PyTorch", "Torch Compile"]
                styles = [("red", "-"), ("blue", "-"), ("purple", "-")]

                if TORCH_HAS_FP8:
                    line_vals.append("triton-fp8")
                    line_names.append("Triton [FP8]")
                    styles.append(("green", "-"))

                if HAS_FLASH:
                    line_vals.append("flash")
                    line_names.append("Flash-2")
                    styles.append(("orange", "-"))

                configs.append(
                    triton.testing.Benchmark(
                        x_names=["N_CTX"],
                        x_vals=[2 ** i for i in range(10, 12)],
                        line_arg="provider",
                        line_vals=line_vals,
                        line_names=line_names,
                        styles=styles,
                        ylabel="TFLOPS",
                        plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{mode}-causal={causal}-ws={warp_specialize}",
                        args={
                            "H": N_HEADS,
                            "BATCH": BATCH,
                            "HEAD_DIM": HEAD_DIM,
                            "mode": mode,
                            "causal": causal,
                            "warp_specialize": warp_specialize,
                        },
                    )
                )


# ============================================================
# helpers
# ============================================================

def str2bool(x):
    if isinstance(x, bool):
        return x
    x = x.lower()
    if x in ("true", "1", "yes", "y", "t"):
        return True
    if x in ("false", "0", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {x}")


def parse_int_list(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def zero_grads(*tensors):
    for t in tensors:
        if t is not None and t.grad is not None:
            t.grad = None


def standard_sm_scale(head_dim):
    return 1.0 / math.sqrt(head_dim)


def provider_list(include_fp8=True, include_flash=True):
    vals = ["triton-fp16", "naive", "torch.compile"]
    if include_fp8 and TORCH_HAS_FP8:
        vals.append("triton-fp8")
    if include_flash and HAS_FLASH:
        vals.append("flash")
    return vals


def clone_qkv(q, k, v):
    q2 = q.detach().clone().requires_grad_(q.requires_grad)
    k2 = k.detach().clone().requires_grad_(k.requires_grad)
    v2 = v.detach().clone().requires_grad_(v.requires_grad)
    return q2, k2, v2


def make_inputs(batch, heads, n_ctx, head_dim, dtype, device, seed=0):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    q = torch.randn((batch, heads, n_ctx, head_dim), dtype=dtype, device=device, generator=g, requires_grad=True)
    k = torch.randn((batch, heads, n_ctx, head_dim), dtype=dtype, device=device, generator=g, requires_grad=True)
    v = torch.randn((batch, heads, n_ctx, head_dim), dtype=dtype, device=device, generator=g, requires_grad=True)
    return q, k, v


def compute_tflops(batch, heads, n_ctx, head_dim, causal, mode, ms):
    flops_per_matmul = 2.0 * batch * heads * n_ctx * n_ctx * head_dim
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5
    return total_flops * 1e-12 / (ms * 1e-3)


def get_flash_qkv(batch, heads, n_ctx, head_dim, dtype, device, seed=0):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    qkv = torch.randn(
        (batch, n_ctx, 3, heads, head_dim),
        dtype=dtype,
        device=device,
        generator=g,
        requires_grad=True,
    )
    return qkv


def provider_forward(provider, q, k, v, causal, sm_scale, warp_specialize):
    if "triton" in provider:
        q_in, k_in, v_in = q, k, v
        if provider == "triton-fp8":
            q_in = q.to(torch.float8_e5m2)
            k_in = k.to(torch.float8_e5m2)
            v_in = v.permute(0, 1, 3, 2).contiguous().permute(0, 1, 3, 2).to(torch.float8_e5m2)
        out = triton_attention(q_in, k_in, v_in, causal, sm_scale, warp_specialize)
        return out, {"q": q, "k": k, "v": v}

    elif provider == "flash":
        qkv = torch.stack([q, k, v], dim=2).transpose(1, 2)  # [B, 3, H, N, D]
        qkv = qkv.permute(0, 3, 1, 2, 4).contiguous()        # [B, N, 3, H, D]
        qkv.requires_grad_(True)
        out = flash_attn_func(qkv, causal=causal)
        return out, {"qkv": qkv}

    elif provider == "naive":
        out = naive_attention(q, k, v, causal, sm_scale)
        return out, {"q": q, "k": k, "v": v}

    elif provider == "torch.compile":
        _ = compiled_attention(q, k, v, causal, sm_scale)
        out = compiled_attention(q, k, v, causal, sm_scale)
        return out, {"q": q, "k": k, "v": v}

    else:
        raise ValueError(f"Unknown provider: {provider}")


def benchmark_one(provider, batch, heads, n_ctx, head_dim, causal, warp_specialize, mode, dtype, device):
    sm_scale = standard_sm_scale(head_dim)

    q, k, v = make_inputs(batch, heads, n_ctx, head_dim, dtype, device, seed=123)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    if mode == "fwd":
        def fn():
            out, _ = provider_forward(provider, q, k, v, causal, sm_scale, warp_specialize)
            return out

        ms = triton.testing.do_bench(fn)
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated(device)

    else:
        out, aux = provider_forward(provider, q, k, v, causal, sm_scale, warp_specialize)
        do = torch.randn_like(out)

        if provider == "torch.compile":
            out.backward(do, retain_graph=True)
            zero_grads(q, k, v)

        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

        def fn():
            if provider == "flash":
                zero_grads(aux["qkv"])
            else:
                zero_grads(q, k, v)
            out.backward(do, retain_graph=True)

        ms = triton.testing.do_bench(fn)
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated(device)

    tflops = compute_tflops(batch, heads, n_ctx, head_dim, causal, mode, ms)

    return {
        "provider": provider,
        "BATCH": batch,
        "H": heads,
        "N_CTX": n_ctx,
        "HEAD_DIM": head_dim,
        "causal": causal,
        "warp_specialize": warp_specialize,
        "mode": mode,
        "dtype": str(dtype),
        "sm_scale": sm_scale,
        "latency_ms": ms,
        "peak_memory_bytes": peak_mem,
        "peak_memory_mb": peak_mem / (1024 ** 2),
        "tflops": tflops,
    }


def run_correctness_check(batch, heads, n_ctx, head_dim, causal, warp_specialize, dtype, device):
    if dtype != torch.float16:
        print(f"[Correctness] Skipping unsupported dtype {dtype} for current reference path")
        return []

    q, k, v = make_inputs(batch, heads, n_ctx, head_dim, dtype, device, seed=999)

    q_ref, k_ref, v_ref = clone_qkv(q, k, v)
    q_tri, k_tri, v_tri = clone_qkv(q, k, v)
    
    sm_scale = standard_sm_scale(head_dim)
    # ref = naive_attention(q_ref, k_ref, v_ref, causal)
    # tri = triton_attention(q_tri, k_tri, v_tri, causal, standard_sm_scale(head_dim), warp_specialize)
    ref = naive_attention(q_ref, k_ref, v_ref, causal, sm_scale)
    tri = triton_attention(q_tri, k_tri, v_tri, causal, sm_scale, warp_specialize)

    out_err = (ref - tri.float()).abs().max().item()

    do = torch.randn_like(ref)
    ref.backward(do)
    tri.backward(do)

    dq_err = (q_ref.grad - q_tri.grad.float()).abs().max().item()
    dk_err = (k_ref.grad - k_tri.grad.float()).abs().max().item()
    dv_err = (v_ref.grad - v_tri.grad.float()).abs().max().item()

    results = [{
        "provider": "triton-fp16",
        "BATCH": batch,
        "H": heads,
        "N_CTX": n_ctx,
        "HEAD_DIM": head_dim,
        "causal": causal,
        "warp_specialize": warp_specialize,
        "mode": "correctness",
        "output_max_abs_err": out_err,
        "dq_max_abs_err": dq_err,
        "dk_max_abs_err": dk_err,
        "dv_max_abs_err": dv_err,
    }]

    if HAS_FLASH:
        q_f, k_f, v_f = clone_qkv(q, k, v)
        qkv = torch.stack([q_f, k_f, v_f], dim=2).transpose(1, 2)
        qkv = qkv.permute(0, 3, 1, 2, 4).contiguous().requires_grad_(True)
        flash_out = flash_attn_func(qkv, causal=causal)
        flash_err = (ref - flash_out.float()).abs().max().item()

        q_ref2, k_ref2, v_ref2 = clone_qkv(q, k, v)
        ref2 = naive_attention(q_ref2, k_ref2, v_ref2, causal, sm_scale)
        #ref2 = naive_attention(q_ref2, k_ref2, v_ref2, causal)
        do2 = torch.randn_like(ref2)
        ref2.backward(do2)
        flash_out.backward(do2)

        qkv_grad = qkv.grad
        dq_flash = (q_ref2.grad - qkv_grad[:, :, 0].permute(0, 2, 1, 3).float()).abs().max().item()
        dk_flash = (k_ref2.grad - qkv_grad[:, :, 1].permute(0, 2, 1, 3).float()).abs().max().item()
        dv_flash = (v_ref2.grad - qkv_grad[:, :, 2].permute(0, 2, 1, 3).float()).abs().max().item()

        results.append({
            "provider": "flash",
            "BATCH": batch,
            "H": heads,
            "N_CTX": n_ctx,
            "HEAD_DIM": head_dim,
            "causal": causal,
            "warp_specialize": warp_specialize,
            "mode": "correctness",
            "output_max_abs_err": flash_err,
            "dq_max_abs_err": dq_flash,
            "dk_max_abs_err": dk_flash,
            "dv_max_abs_err": dv_flash,
        })

    return results


def write_csv(rows, path):
    if not rows:
        return
    fieldnames = sorted(set().union(*[row.keys() for row in rows]))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_speedup_rows(rows, baseline="naive"):
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row["BATCH"],
            row["H"],
            row["N_CTX"],
            row["HEAD_DIM"],
            row["causal"],
            row["warp_specialize"],
            row["mode"],
        )
        grouped[key].append(row)

    speedup_rows = []
    for key, group in grouped.items():
        base_rows = [r for r in group if r["provider"] == baseline]
        if not base_rows:
            continue
        base_ms = base_rows[0]["latency_ms"]
        for r in group:
            rr = dict(r)
            rr["speedup_vs_" + baseline] = base_ms / r["latency_ms"] if r["latency_ms"] > 0 else float("nan")
            speedup_rows.append(rr)
    return speedup_rows


def make_custom_plots(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    grouped = defaultdict(list)
    for row in rows:
        key = (row["mode"], row["causal"], row["HEAD_DIM"], row["BATCH"], row["H"], row["warp_specialize"])
        grouped[key].append(row)

    for key, group in grouped.items():
        mode, causal, head_dim, batch, heads, ws = key
        providers = sorted(set(r["provider"] for r in group))
        nctxs = sorted(set(r["N_CTX"] for r in group))

        # Latency plot
        plt.figure(figsize=(8, 5))
        for p in providers:
            xs = []
            ys = []
            for n in nctxs:
                match = [r for r in group if r["provider"] == p and r["N_CTX"] == n]
                if match:
                    xs.append(n)
                    ys.append(match[0]["latency_ms"])
            if xs:
                plt.plot(xs, ys, marker="o", label=p)
        plt.xlabel("Sequence Length (N_CTX)")
        plt.ylabel("Latency (ms)")
        plt.title(f"Latency | mode={mode} causal={causal} d={head_dim} B={batch} H={heads} ws={ws}")
        plt.xscale("log", base=2)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"latency_mode={mode}_causal={causal}_d={head_dim}_B={batch}_H={heads}_ws={ws}.png"))
        plt.close()

        # TFLOPS plot
        plt.figure(figsize=(8, 5))
        for p in providers:
            xs = []
            ys = []
            for n in nctxs:
                match = [r for r in group if r["provider"] == p and r["N_CTX"] == n]
                if match:
                    xs.append(n)
                    ys.append(match[0]["tflops"])
            if xs:
                plt.plot(xs, ys, marker="o", label=p)
        plt.xlabel("Sequence Length (N_CTX)")
        plt.ylabel("TFLOPS")
        plt.title(f"Throughput | mode={mode} causal={causal} d={head_dim} B={batch} H={heads} ws={ws}")
        plt.xscale("log", base=2)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"tflops_mode={mode}_causal={causal}_d={head_dim}_B={batch}_H={heads}_ws={ws}.png"))
        plt.close()

        # Memory plot
        plt.figure(figsize=(8, 5))
        for p in providers:
            xs = []
            ys = []
            for n in nctxs:
                match = [r for r in group if r["provider"] == p and r["N_CTX"] == n]
                if match:
                    xs.append(n)
                    ys.append(match[0]["peak_memory_mb"])
            if xs:
                plt.plot(xs, ys, marker="o", label=p)
        plt.xlabel("Sequence Length (N_CTX)")
        plt.ylabel("Peak Memory (MB)")
        plt.title(f"Memory | mode={mode} causal={causal} d={head_dim} B={batch} H={heads} ws={ws}")
        plt.xscale("log", base=2)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"memory_mode={mode}_causal={causal}_d={head_dim}_B={batch}_H={heads}_ws={ws}.png"))
        plt.close()

        # Speedup plot vs naive
        plt.figure(figsize=(8, 5))
        base = [r for r in group if r["provider"] == "naive"]
        if base:
            for p in providers:
                xs = []
                ys = []
                for n in nctxs:
                    target = [r for r in group if r["provider"] == p and r["N_CTX"] == n]
                    base_r = [r for r in group if r["provider"] == "naive" and r["N_CTX"] == n]
                    if target and base_r:
                        xs.append(n)
                        ys.append(base_r[0]["latency_ms"] / target[0]["latency_ms"])
                if xs:
                    plt.plot(xs, ys, marker="o", label=p)
            plt.xlabel("Sequence Length (N_CTX)")
            plt.ylabel("Speedup vs Naive")
            plt.title(f"Speedup | mode={mode} causal={causal} d={head_dim} B={batch} H={heads} ws={ws}")
            plt.xscale("log", base=2)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"speedup_mode={mode}_causal={causal}_d={head_dim}_B={batch}_H={heads}_ws={ws}.png"))
            plt.close()


# ============================================================
# Perf report function 
# ============================================================

@triton.testing.perf_report(configs)
def run_benchmark(BATCH, H, N_CTX, HEAD_DIM, causal, warp_specialize, mode, provider, device=DEVICE):
    assert mode in ["fwd", "bwd"]
    dtype = torch.float16
    sm_scale = standard_sm_scale(HEAD_DIM)

    q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)

    if "triton" in provider:
        if mode == "fwd" and "fp8" in provider:
            q = q.to(torch.float8_e5m2)
            k = k.to(torch.float8_e5m2)
            v = v.permute(0, 1, 3, 2).contiguous().permute(0, 1, 3, 2).to(torch.float8_e5m2)

        fn = lambda: triton_attention(q, k, v, causal, sm_scale, warp_specialize)

    elif provider == "flash":
        qkv = torch.randn((BATCH, N_CTX, 3, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: flash_attn_func(qkv, causal=causal)

    elif provider in ["naive", "torch.compile"]:
        target_fn = compiled_attention if provider == "torch.compile" else naive_attention

        if provider == "torch.compile":
            _ = target_fn(q, k, v, causal, sm_scale)

        fn = lambda: target_fn(q, k, v, causal, sm_scale)

    else:
        raise ValueError(f"Unknown provider {provider}")

    if mode == "bwd":
        o = fn()
        do = torch.randn_like(o)

        if provider == "torch.compile":
            o.backward(do, retain_graph=True)
            q.grad, k.grad, v.grad = None, None, None

        fn = lambda: o.backward(do, retain_graph=True)
    
    if mode == "fwd":
        for _ in range(2):
            _ = fn()
        torch.cuda.synchronize()
    else:
        _ = fn()
        torch.cuda.synchronize()
        
    ms = triton.testing.do_bench(fn)
    

    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5

    return total_flops * 1e-12 / (ms * 1e-3)


# ============================================================
# CLI experiment runner
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=str, default="4")
    parser.add_argument("--head-counts", type=str, default="32")
    parser.add_argument("--seq-lengths", type=str, default="512,1024,2048")
    parser.add_argument("--head-dims", type=str, default="64")
    parser.add_argument("--modes", type=str, default="fwd,bwd")
    parser.add_argument("--causal-values", type=str, default="true,false")
    parser.add_argument("--providers", type=str, default="")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--run-correctness", type=str2bool, default=True)
    parser.add_argument("--run-bench", type=str2bool, default=True)
    parser.add_argument("--run-triton-report", type=str2bool, default=True)
    parser.add_argument("--outdir", type=str, default="benchmark_results_final")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    plots_dir = os.path.join(args.outdir, "custom_plots")
    os.makedirs(plots_dir, exist_ok=True)

    batch_sizes = parse_int_list(args.batch_sizes)
    head_counts = parse_int_list(args.head_counts)
    seq_lengths = parse_int_list(args.seq_lengths)
    head_dims = parse_int_list(args.head_dims)
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    causal_values = [str2bool(x.strip()) for x in args.causal_values.split(",") if x.strip()]

    if args.providers.strip():
        providers = [x.strip() for x in args.providers.split(",") if x.strip()]
    else:
        providers = provider_list(include_fp8=True, include_flash=True)

    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if args.dtype.lower() not in dtype_map:
        raise ValueError(f"Unsupported dtype: {args.dtype}")
    dtype = dtype_map[args.dtype.lower()]

    correctness_rows = []
    benchmark_rows = []

    if args.run_correctness:
        print("\nRunning correctness checks")
        for batch in batch_sizes:
            for heads in head_counts:
                for n_ctx in seq_lengths:
                    for head_dim in head_dims:
                        for causal in causal_values:
                            enable_ws = is_blackwell() or (is_hopper() and not causal)
                            ws_values = [False, True] if enable_ws else [False]
                            for warp_specialize in ws_values:
                                print(
                                    f"\n[Correctness] B={batch}, H={heads}, N={n_ctx}, D={head_dim}, "
                                    f"causal={causal}, ws={warp_specialize}"
                                )
                                try:
                                    rows = run_correctness_check(
                                        batch=batch,
                                        heads=heads,
                                        n_ctx=n_ctx,
                                        head_dim=head_dim,
                                        causal=causal,
                                        warp_specialize=warp_specialize,
                                        dtype=torch.float16,
                                        device=DEVICE,
                                    )
                                    correctness_rows.extend(rows)
                                    for r in rows:
                                        print(
                                            f"{r['provider']}: "
                                            f"out={r['output_max_abs_err']:.6e}, "
                                            f"dq={r['dq_max_abs_err']:.6e}, "
                                            f"dk={r['dk_max_abs_err']:.6e}, "
                                            f"dv={r['dv_max_abs_err']:.6e}"
                                        )
                                except Exception as e:
                                    print(f"Correctness failed for config: {e}")

        write_csv(correctness_rows, os.path.join(args.outdir, "correctness_results.csv"))

    if args.run_bench:
        print("\nRunning benchmark measurements")
        for batch in batch_sizes:
            for heads in head_counts:
                for n_ctx in seq_lengths:
                    for head_dim in head_dims:
                        for causal in causal_values:
                            for mode in modes:
                                enable_ws = mode == "fwd" and (is_blackwell() or (is_hopper() and not causal))
                                ws_values = [False, True] if enable_ws else [False]
                                for warp_specialize in ws_values:
                                    for provider in providers:
                                        if provider == "triton-fp8" and not TORCH_HAS_FP8:
                                            continue
                                        if provider == "flash" and not HAS_FLASH:
                                            continue
                                        try:
                                            row = benchmark_one(
                                                provider=provider,
                                                batch=batch,
                                                heads=heads,
                                                n_ctx=n_ctx,
                                                head_dim=head_dim,
                                                causal=causal,
                                                warp_specialize=warp_specialize,
                                                mode=mode,
                                                dtype=dtype,
                                                device=DEVICE,
                                            )
                                            benchmark_rows.append(row)
                                            print(
                                                f"[Bench] provider={provider:12s} "
                                                f"B={batch} H={heads} N={n_ctx} D={head_dim} "
                                                f"mode={mode} causal={causal} ws={warp_specialize} | "
                                                f"{row['latency_ms']:.4f} ms | "
                                                f"{row['peak_memory_mb']:.2f} MB | "
                                                f"{row['tflops']:.2f} TFLOPS"
                                            )
                                        except Exception as e:
                                            print(
                                                f"[Bench Failed] provider={provider} "
                                                f"B={batch} H={heads} N={n_ctx} D={head_dim} "
                                                f"mode={mode} causal={causal} ws={warp_specialize}: {e}"
                                            )

        benchmark_rows = add_speedup_rows(benchmark_rows, baseline="naive")
        write_csv(benchmark_rows, os.path.join(args.outdir, "benchmark_results.csv"))
        make_custom_plots(benchmark_rows, plots_dir)

    if args.run_triton_report:
        print("\nRunning Triton perf report")
        run_benchmark.run(save_path=args.outdir, print_data=True)


if __name__ == "__main__":
    main()