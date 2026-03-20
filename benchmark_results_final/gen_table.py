#!/usr/bin/env python3
"""
Generate the LaTeX results table from benchmark_results.csv.
Filters to causal=True at N_CTX in {256, 512, 1024, 2048} and prints
a table* snippet ready to paste into report.tex.

Usage: python3 gen_table.py [path/to/benchmark_results.csv]
"""
import csv
import sys

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results.csv"
KEEP_CTX = {256, 512, 1024, 2048}

rows = list(csv.DictReader(open(CSV_PATH)))
kept = [r for r in rows if r["causal"] == "True" and int(r["N_CTX"]) in KEEP_CTX]

# Write filtered CSV alongside the script for inspection
out_csv = CSV_PATH.replace(".csv", "_table_data.csv")
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(kept)

# Build lookup: (provider, n_ctx, mode) -> row
data = {(r["provider"], int(r["N_CTX"]), r["mode"]): r for r in kept}

PROVIDERS = [
    ("naive",         "Naive"),
    ("torch.compile", r"\texttt{torch.compile}"),
    ("triton",        "Triton"),
    ("flash",         "FlashAttention (CUDA)"),
]
CTXS = sorted(KEEP_CTX)

lines = [
    r"\begin{table*}[h!]",
    r"  \centering",
    r"  \caption{Summary of benchmark results for causal attention (batch=4, $H$=32, head\_dim=64)."
    r" Speedup is relative to the naive PyTorch baseline."
    r" Peak memory is the maximum GPU allocation during each pass.}",
    r"  \label{tab:results}",
    r"  \begin{tabular}{l r r r r r r r}",
    r"    \hline",
    r"    \multirow{2}{*}{\textbf{Implementation}} & \multirow{2}{*}{$\mathbf{N_{ctx}}$}"
    r" & \multicolumn{2}{c}{\textbf{Forward Pass}}"
    r" & \multicolumn{2}{c}{\textbf{Backward Pass}}"
    r" & \multicolumn{2}{c}{\textbf{Peak Memory (MB)}} \\",
    r"     & & \textbf{TFLOPS} & \textbf{Speedup} & \textbf{TFLOPS} & \textbf{Speedup} & \textbf{Fwd} & \textbf{Bwd} \\",
    r"    \hline",
]

for pkey, plabel in PROVIDERS:
    for i, ctx in enumerate(CTXS):
        fwd = data[(pkey, ctx, "fwd")]
        bwd = data[(pkey, ctx, "bwd")]
        impl_col = rf"    \multirow{{{len(CTXS)}}}{{*}}{{{plabel}}}" if i == 0 else "    "
        lines.append(
            f"{impl_col} & {ctx}"
            f" & {float(fwd['tflops']):.1f} & ${float(fwd['speedup_vs_naive']):.2f}\\times$"
            f" & {float(bwd['tflops']):.1f} & ${float(bwd['speedup_vs_naive']):.2f}\\times$"
            f" & {round(float(fwd['peak_memory_mb']))} & {round(float(bwd['peak_memory_mb']))} \\\\"
        )
    lines.append(r"    \hline")

lines += [r"  \end{tabular}", r"\end{table*}"]

print("\n".join(lines))
