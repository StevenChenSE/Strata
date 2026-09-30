#!/usr/bin/env python3
"""Two decode questions left open by the x16 work.

A) Does the prompt-diversity effect hold for DECODE throughput, not just prefill?  The earlier prompt test ran
   --max-new 1, so tg was never measured on the narrow/diverse/prose prompts.

B) Is the speculative draft length still optimal at x16?  The window sweep that chose --spec 4 was run on the x8
   link, and that sweep also compared --spec 2 WITHOUT MTP against 3/4/6 WITH it, so the short end was
   confounded.  This tests --spec 2/3/4/6 with MTP for all of them, at 1K and at 32K, with a fixed decode
   protocol per tier, and reports acceptance and tokens/round alongside throughput because those are what the
   window is trading against.

Configurations are interleaved within each repetition.  Usage: python3 bench/tools/tg_x16.py
"""
import re
import statistics
import subprocess

BASE = [
    "env", "HIP_VISIBLE_DEVICES=0", "./build-hip/strata-hip",
    "--pack", "/home/jianwei/Strata-data/packs/iq3_s",
    "--native", "/home/jianwei/Strata-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
    "--ple-gguf", "/home/jianwei/Strata-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf",
    "--expert-profile", "data/expert-profile.bin", "--expert-cache", "auto", "--prefill", "2048",
    "--spec-min-p", "0.5", "--adapt-every", "2", "--pcie-frac", "0.30",
    "--mtp", "/home/jianwei/Strata-data/models/IQ3_S/mtp/rt",
]


def run(ctx, max_new, prompt, args=()):
    cmd = BASE + ["--max-context", str(ctx), "--max-new", str(max_new), "--tokens-file", prompt] + list(args)
    with open("/tmp/tg_run.log", "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=1800)
    t = open("/tmp/tg_run.log", errors="ignore").read()
    tg = re.search(r"decode\s+\d+ tokens in [\d.]+ ms\s+->\s+([\d.]+) tok/s", t)
    sp = re.search(r"drafts accepted (\d+) of (\d+) \(([\d.]+)\), ([\d.]+) tokens per round", t)
    return (float(tg.group(1)) if tg else None,
            float(sp.group(3)) if sp else None,
            float(sp.group(4)) if sp else None)


def sweep(tag, keys, mk, reps):
    tg = {k: [] for k in keys}
    acc = {k: [] for k in keys}
    tpr = {k: [] for k in keys}
    for i in range(reps):
        for k in keys:
            g, a, r = run(*mk(k))
            tg[k].append(g)
            acc[k].append(a)
            tpr[k].append(r)
        print(f"  rep {i + 1}/{reps} done", flush=True)
    print(f"\n== {tag}")
    print(f"{'config':>8} {'tg median':>10} {'tg range':>14} {'accept':>8} {'tok/round':>10}")
    best = None
    for k in keys:
        xs = [x for x in tg[k] if x]
        if not xs:
            print(f"{k:>8}  no data")
            continue
        m = statistics.median(xs)
        a = statistics.median([x for x in acc[k] if x]) if any(acc[k]) else float("nan")
        r = statistics.median([x for x in tpr[k] if x]) if any(tpr[k]) else float("nan")
        print(f"{k:>8} {m:>10.1f} {f'{min(xs):.1f}-{max(xs):.1f}':>14} {a:>8.3f} {r:>10.2f}", flush=True)
        if best is None or m > best[1]:
            best = (k, m)
    print(f"  -> best: {best[0]} ({best[1]:.1f} tok/s)", flush=True)


if __name__ == "__main__":
    # A) prompt diversity x decode
    sweep("A) prompt diversity, decode @4K prompts, 64 tokens", ["narrow", "diverse", "prose"],
          lambda k: (8192, 64, f"bench/tools/prompts/4k_{k}.txt", ["--spec", "4"]), 3)

    # B) draft length at x16, MTP on for every value
    sweep("B) --spec @1K, 128 tokens", ["2", "3", "4", "6"],
          lambda k: (4096, 128, "bench/tools/prompts/1k.txt", ["--spec", k]), 3)
    sweep("B) --spec @32K, 64 tokens", ["2", "3", "4", "6"],
          lambda k: (36864, 64, "bench/tools/prompts/32k.txt", ["--spec", k]), 2)
    print("\nDONE", flush=True)
