import torch
import triton
import os
import torch._functorch.config

# import Triton implementation and helpers
from fused_attention import (
    attention as triton_attention, 
    DEVICE, 
    is_hopper, 
    is_blackwell
)

try:
    from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func as flash_attn_func
    HAS_FLASH = True
except ImportError:
    HAS_FLASH = False

TORCH_HAS_FP8 = hasattr(torch, 'float8_e5m2')

torch._functorch.config.donated_buffer = False
# naive
def naive_attention(q, k, v, causal):
    S = q @ k.transpose(-2, -1)
    if causal:
        N_CTX = q.shape[-2]
        mask = torch.tril(torch.ones((N_CTX, N_CTX), device=q.device))
        S = S.masked_fill(mask == 0, float("-inf"))
    P = torch.softmax(S, dim=-1)
    o = P @ v
    return o

# torch.compile
compiled_attention = torch.compile(naive_attention)
compiled_attention_max = torch.compile(naive_attention, mode='max_autotune')

BATCH, N_HEADS = 4, 32
configs = []

for HEAD_DIM in [64]:
    for mode in ["fwd", "bwd"]:
        for causal in [True, False]:
            enable_ws = mode == "fwd" and (is_blackwell() or (is_hopper() and not causal))
            for warp_specialize in [False, True] if enable_ws else [False]:
                
                # Define the providers we want to compare
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
                        x_vals=[2**i for i in range(10, 12)], # e.g., 1024 to 16384
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
                    ))

@triton.testing.perf_report(configs)
def run_benchmark(BATCH, H, N_CTX, HEAD_DIM, causal, warp_specialize, mode, provider, device=DEVICE):
    assert mode in ["fwd", "bwd"]
    dtype = torch.float16
    sm_scale = 1.3

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
            _ = target_fn(q, k, v, causal)
            
        fn = lambda: target_fn(q, k, v, causal)

    if mode == "bwd":
        o = fn()
        do = torch.randn_like(o)
        
        if provider == "torch.compile":
            # Warmup backward compiler
            o.backward(do, retain_graph=True)
            q.grad, k.grad, v.grad = None, None, None
            
        fn = lambda: o.backward(do, retain_graph=True)

    ms = triton.testing.do_bench(fn)

    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
        
    return total_flops * 1e-12 / (ms * 1e-3)

if __name__ == "__main__":
    os.makedirs("benchmark_results", exist_ok=True)
    run_benchmark.run(save_path="benchmark_results", print_data=True)