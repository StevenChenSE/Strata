#!/usr/bin/env python3
"""Round-2 tuning after the x16 link: resolve the ring at 4K and settle pcie-frac at 32K.

Uses the same interleaving as tune_x16_confirm.py.  The 32K decode uses a fixed 64-token protocol because the
short 32-token decode is dominated by warm-up and acceptance variance, which is what made the first two 32K
sweeps contradict each other.

Usage: python3 bench/tools/tune_x16_round2.py
"""
import statistics
import subprocess
import re

BASE = [
    "HIP_VISIBLE_DEVICES=0", "./build-hip/strata-hip",
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
    print(f"{'config':>10} {'pp median':>10} {'pp range':>16} {'tg median':>10} {'tg range':>16}")
    for k in keys:
        p = [x for x in pp[k] if x]
        g = [x for x in tg[k] if x]
        print(f"{k:>10} {statistics.median(p) if p else float('nan'):>10.1f} "
              f"{(f'{min(p):.0f}-{max(p):.0f}' if p else '-'):>16} "
              f"{statistics.median(g) if g else float('nan'):>10.1f} "
              f"{(f'{min(g):.1f}-{max(g):.1f}' if g else '-'):>16}", flush=True)


if __name__ == "__main__":
    keys = ["384", "512"]
    pp = {k: [] for k in keys}
    for i in range(8):
        for k in keys:
            env = () if k == "384" else ("STRATA_PREFILL_RING=" + k,)
            p, _ = run(8192, 1, "bench/tools/prompts/4k.txt", (), env)
            pp[k].append(p)
        print(f"  rep {i + 1}/8 done", flush=True)
    report("STRATA_PREFILL_RING @4K, interleaved x8", keys, pp, {k: [] for k in keys})

    keys2 = ["0.30", "0.55"]
    pp2 = {k: [] for k in keys2}
    tg2 = {k: [] for k in keys2}
    for i in range(3):
        for k in keys2:
            p, g = run(36864, 64, "bench/tools/prompts/32k.txt", ["--pcie-frac", k])
            pp2[k].append(p)
            tg2[k].append(g)
        print(f"  rep {i + 1}/3 done", flush=True)
    report("pcie-frac @32K, 64 tokens (fixed protocol), interleaved x3", keys2, pp2, tg2)
    print("\nDONE", flush=True)
