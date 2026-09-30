#!/usr/bin/env python3
"""Aggregate a rocprofv3 kernel trace by phase: prefill vs decode.

The decode begins at the verify window's first polling kernel, so `wait_flag_ge_kernel` is a reliable
separator (the prefill and the graph capture precede it).  With no phase markers in the trace this is the
cheapest way to answer "which kernel owns the prompt path".

    python bench/tools/prof_prefill.py /tmp/prof/k_kernel_trace.csv [--top N] [--group]
"""
import argparse, csv, collections, re, sys

def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                s, e = int(r["Start_Timestamp"]), int(r["End_Timestamp"])
            except (KeyError, ValueError):
                continue
            rows.append((s, e, r.get("Kernel_Name", "?")))
    rows.sort()
    return rows

def short(name):
    n = re.sub(r"strata::(kernels|prefill)::(anonymous namespace)::", "", name)
    n = re.sub(r"<.*", "", n)
    return n[:44]

def group(name):
    n = name
    if "dequant" in n: return "dequant (weight->f16)"
    if n.startswith("Cijk") or "rocblas" in n.lower(): return "rocBLAS GEMM"
    if "mul_mat_q" in n or "mmq" in n: return "MMQ (quantized GEMM)"
    if "gdn_rec" in n: return "gdn recurrence"
    if "gdn_" in n: return "gdn other"
    if "qsa" in n or "attn" in n: return "attention"
    if "wait_flag" in n: return "wait (doorbell poll)"
    if "streamOps" in n: return "stream-ops shader"
    if "copyBuffer" in n or "fillBuffer" in n: return "runtime copy/fill"
    if "fetch_blobs" in n: return "fetch blobs"
    return "other"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--top", type=int, default=14)
    ap.add_argument("--group", action="store_true", help="aggregate into functional groups instead of raw names")
    a = ap.parse_args()
    rows = load(a.trace)
    if not rows:
        print("no kernels", file=sys.stderr); return 1
    cut = next((i for i, r in enumerate(rows) if "wait_flag" in r[2]), len(rows))
    for part, label in ((rows[:cut], "PREFILL (+capture)"), (rows[cut:], "DECODE")):
        d, c = collections.Counter(), collections.Counter()
        for s, e, n in part:
            key = group(n) if a.group else short(n)
            d[key] += (e - s) / 1e3
            c[key] += 1
        tot = sum(d.values())
        print(f"\n== {label}: {len(part)} kernels, {tot/1e3:.1f} ms of kernel time")
        for n, ms in d.most_common(a.top):
            print(f"   {ms/1e3:9.2f} ms  x{c[n]:6d}  {100*ms/tot:5.1f}%  {n}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
