#!/usr/bin/env python3
"""Interleaved confirmation of the x16 tuning candidates.

Alternates configurations within each repetition so that machine drift (the host-side interference that makes 4K
noisy) cannot masquerade as a configuration effect.  Reports medians and min/max per configuration.

Usage: python3 bench/tools/tune_x16_confirm.py
"""
import statistics
import subprocess
import re

BASE = [
    "HIP_VISIBLE_DEVICES=0",
    "./build-hip/strata-hip",
    "--pack", "/home/jianwei/Strata-data/packs/iq3_s",
    "--native", "/home/jianwei/Strata-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
    "--ple-gguf", "/home/jianwei/Strata-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf",
    "--expert-profile", "data/expert-profile.bin", "--expert-cache", "auto", "--prefill", "2048",
    "--spec", "4", "--spec-min-p", "0.5", "--adapt-every", "2",
    "--mtp", "/home/jianwei/Strata-data/models/IQ3_S/mtp/rt",
]


def run(ctx, max_new, prompt, args=(), env=()):
    cmd = ["env"] + list(env) + list(BASE) + ["--max-context", str(ctx), "--max-new", str(max_new),
                                              "--tokens-file", prompt] + list(args)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    t = r.stdout + r.stderr
    pp = re.search(r"prefill\s+\d+ tokens in [\d.]+ ms\s+->\s+([\d.]+) tok/s", t)
    tg = re.search(r"decode\s+\d+ tokens in [\d.]+ ms\s+->\s+([\d.]+) tok/s", t)
    return (float(pp.group(1)) if pp else None, float(tg.group(1)) if tg else None)


def report(tag, keys, pp, tg):
    print(f"\n== {tag}")
    print(f"{'config':>10} {'pp median':>10} {'pp range':>18} {'tg median':>10} {'tg range':>16}")
    for k in keys:
        p = [x for x in pp[k] if x]
        g = [x for x in tg[k] if x]
        pr = f"{min(p):.0f}-{max(p):.0f}" if p else "-"
        gr = f"{min(g):.1f}-{max(g):.1f}" if g else "-"
        print(f"{k:>10} {statistics.median(p) if p else float('nan'):>10.1f} {pr:>18} "
              f"{statistics.median(g) if g else float('nan'):>10.1f} {gr:>16}", flush=True)


if __name__ == "__main__":
    # 1K decode is where pcie-frac matters most; 5 interleaved reps over the three candidates
    keys = ["0.30", "0.55", "0.70"]
    pp = {k: [] for k in keys}
    tg = {k: [] for k in keys}
    for i in range(5):
        for k in keys:
            p, g = run(4096, 128, "bench/tools/prompts/1k.txt", ["--pcie-frac", k])
            pp[k].append(p)
            tg[k].append(g)
        print(f"  rep {i + 1}/5 done", flush=True)
    report("pcie-frac @1K, 128 tokens, interleaved x5", keys, pp, tg)

    # 32K: does the opposite end really win there?
    keys32 = ["0.55", "0.85"]
    pp32 = {k: [] for k in keys32}
    tg32 = {k: [] for k in keys32}
    for i in range(3):
        for k in keys32:
            p, g = run(36864, 32, "bench/tools/prompts/32k.txt", ["--pcie-frac", k])
            pp32[k].append(p)
            tg32[k].append(g)
        print(f"  rep {i + 1}/3 done", flush=True)
    report("pcie-frac @32K, 32 tokens, interleaved x3", keys32, pp32, tg32)

    # ring at 4K, interleaved
    keysr = ["192", "384", "512"]
    ppr = {k: [] for k in keysr}
    tgr = {k: [] for k in keysr}
    for i in range(4):
        for k in keysr:
            env = () if k == "384" else ("STRATA_PREFILL_RING=" + k,)
            p, g = run(8192, 1, "bench/tools/prompts/4k.txt", (), env)
            ppr[k].append(p)
            tgr[k].append(g)
        print(f"  rep {i + 1}/4 done", flush=True)
    report("STRATA_PREFILL_RING @4K, interleaved x4", keysr, ppr, tgr)
    print("\nDONE", flush=True)
