#!/usr/bin/env python3
"""Determine the best configuration for the 32K tier on the PCIe 4.0 x16 link.

32K is the link-heaviest tier (537 GB of expert blobs streamed), so it is where both remaining knobs should
matter most: --pcie-frac (how much of each layer's missed experts the GPU pulls over PCIe instead of the CPU
computing them) and STRATA_PREFILL_RING (the prefetch ring, whose time-depth halved when the link doubled).

Design notes:
  * every configuration is measured interleaved within a repetition, so host-side interference - which is what
    makes this box's timings heavy-tailed - cannot masquerade as a configuration effect;
  * the decode uses a FIXED 64-token protocol, because the 32-token decode used earlier is dominated by warm-up
    and acceptance variance and that is why two earlier 32K pcie-frac sweeps disagreed;
  * pcie-frac is swept first (it is expected to be the larger effect), the ring second.

Usage: python3 bench/tools/tune_32k.py
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
    "--max-context", "36864", "--max-new", "64", "--tokens-file", "bench/tools/prompts/32k.txt",
]

REPS = 3


def run(args=(), env=()):
    cmd = ["env"] + list(env) + list(BASE) + list(args)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    t = r.stdout + r.stderr
    pp = re.search(r"prefill\s+\d+ tokens in [\d.]+ ms\s+->\s+([\d.]+) tok/s", t)
    tg = re.search(r"decode\s+\d+ tokens in [\d.]+ ms\s+->\s+([\d.]+) tok/s", t)
    st = re.search(r"experts streamed (\d+).*?resident (\d+)", t)
    return (float(pp.group(1)) if pp else None,
            float(tg.group(1)) if tg else None,
            st.group(1) if st else "-")


def sweep(tag, keys, mk):
    pp = {k: [] for k in keys}
    tg = {k: [] for k in keys}
    streamed = "-"
    for i in range(REPS):
        for k in keys:
            a, e = mk(k)
            p, g, st = run(a, e)
            pp[k].append(p)
            tg[k].append(g)
            if st != "-":
                streamed = st
        print(f"  rep {i + 1}/{REPS} done", flush=True)
    print(f"\n== {tag}   (experts streamed {streamed})")
    print(f"{'config':>8} {'pp med':>8} {'pp range':>15} {'tg med':>8} {'tg range':>15}")
    best = None
    for k in keys:
        p = [x for x in pp[k] if x]
        g = [x for x in tg[k] if x]
        pm = statistics.median(p) if p else float("nan")
        gm = statistics.median(g) if g else float("nan")
        print(f"{k:>8} {pm:>8.1f} {(f'{min(p):.0f}-{max(p):.0f}' if p else '-'):>15} "
              f"{gm:>8.1f} {(f'{min(g):.1f}-{max(g):.1f}' if g else '-'):>15}", flush=True)
        if best is None or pm > best[1]:
            best = (k, pm, gm)
    print(f"  -> best pp: {best[0]} (pp {best[1]:.1f}, tg {best[2]:.1f})", flush=True)


if __name__ == "__main__":
    sweep("pcie-frac @32K, 64 tokens, interleaved", ["0.00", "0.15", "0.30", "0.55"],
          lambda k: (["--pcie-frac", k], ()))
    sweep("STRATA_PREFILL_RING @32K, 64 tokens, interleaved", ["192", "384", "512"],
          lambda k: ((), () if k == "384" else ("STRATA_PREFILL_RING=" + k,)))
    print("\nDONE", flush=True)
