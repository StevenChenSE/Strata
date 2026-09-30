#!/usr/bin/env python3
"""Tune the engine for the PCIe 4.0 x16 link.

Two candidates, both zero-code:
  * --pcie-frac  (share of each layer's missed experts the GPU reads over PCIe while the CPU computes the rest;
                  a static model default of 0.55 that cannot know the link doubled)
  * STRATA_PREFILL_RING (the prefetch ring's *time* depth halved when the link doubled: 384 slots was 58 ms of
                  copy work at 13.5 GB/s, it is ~27 ms at 25.8)

Prints one line per configuration as it completes, then a summary.  Usage: python3 bench/tools/tune_x16.py
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


def run(ctx, max_new, prompt, extra_args=(), extra_env=(), profile=False):
    env = ["env"] + list(extra_env) + list(BASE)
    if profile:
        env.insert(1, "STRATA_VERIFY_PROFILE=1")
    cmd = env + ["--max-context", str(ctx), "--max-new", str(max_new), "--tokens-file", prompt] + list(extra_args)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    t = r.stdout + r.stderr
    out = {"pp": None, "tg": None, "share": None}
    m = re.search(r"prefill\s+\d+ tokens in [\d.]+ ms\s+->\s+([\d.]+) tok/s", t)
    if m:
        out["pp"] = float(m.group(1))
    d = re.search(r"decode\s+\d+ tokens in [\d.]+ ms\s+->\s+([\d.]+) tok/s", t)
    if d:
        out["tg"] = float(d.group(1))
    s = re.search(r"pcie experts\s+([\d.]+) distinct experts per layer read over PCIe \(share (\d+)/256", t)
    if s:
        out["share"] = f"{s.group(1)} ({s.group(2)}/256)"
    return out


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def sweep(label, ctx, max_new, prompt, values, reps, extra_of, env_of=None, profile=False):
    print(f"\n== {label}: values {values}, {reps} reps", flush=True)
    res = {}
    for v in values:
        pps, tgs, share = [], [], "-"
        for _ in range(reps):
            r = run(ctx, max_new, prompt, extra_of(v), (env_of(v) if env_of else ()), profile=profile)
            if r["pp"] is not None:
                pps.append(r["pp"])
            if r["tg"] is not None:
                tgs.append(r["tg"])
            if r["share"]:
                share = r["share"]
        res[v] = (med(pps), med(tgs), share)
        print(f"  {v:>6}: pp {med(pps):7.1f}  tg {med(tgs):6.1f}  share {share}"
              f"   raw pp {['%.0f' % x for x in pps]} tg {['%.1f' % x for x in tgs]}", flush=True)
    return res


if __name__ == "__main__":
    frac1k = sweep("pcie-frac @1K (128 tokens)", 4096, 128, "bench/tools/prompts/1k.txt",
                   ["0.30", "0.55", "0.70", "0.85", "1.00"], 3,
                   lambda v: ["--pcie-frac", v], profile=True)
    best = max(frac1k, key=lambda k: frac1k[k][1])
    print(f"\nbest pcie-frac at 1K by tg: {best} ({frac1k[best][1]:.1f} tok/s, default is 0.55)", flush=True)

    frac32k = sweep("pcie-frac @32K (32 tokens)", 36864, 32, "bench/tools/prompts/32k.txt",
                    ["0.55", "0.85", "1.00"], 2, lambda v: ["--pcie-frac", v])
    best32 = max(frac32k, key=lambda k: frac32k[k][1])
    print(f"\nbest pcie-frac at 32K by tg: {best32} ({frac32k[best32][1]:.1f} tok/s)", flush=True)

    rings = sweep("STRATA_PREFILL_RING @4K", 8192, 1, "bench/tools/prompts/4k.txt",
                  ["192", "384", "512"], 3, lambda v: (),
                  env_of=lambda v: (() if v == "384" else ("STRATA_PREFILL_RING=" + v,)))
    best_ring = max(rings, key=lambda k: rings[k][0])
    print(f"\nbest ring at 4K by pp: {best_ring} ({rings[best_ring][0]:.1f} tok/s)", flush=True)
    print("\nDONE", flush=True)
