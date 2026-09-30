#!/usr/bin/env python3
"""Decode throughput against context DEPTH, at the user's configuration.

Configuration held fixed at every point: --max-context 262144 (256K capacity), --kv int8, --spec 2 (the draft
length that won at long context), --pcie-frac 0.30, 64 decoded tokens.

First attempt used nested prefixes of one corpus, which confounds depth with CONTENT: each depth continued from a
different place in the text, so MTP acceptance - and therefore tg - varied because of what was being continued,
not because of how much context preceded it (4K continued TypeScript at 0.923 acceptance, 32K continued the
Python/CUDA part at 1.000).

So each point here is [history of N tokens] + [the SAME 512-token tail], the tail taken from a disjoint part of
the corpus (tokens 39,000-39,512) so it is never already inside the history.  Every depth therefore continues
from the same text, and only the amount of preceding context changes.  depth 0 is the tail alone.

Points are interleaved within each repetition; each run logs to /tmp/depth2_<tag>.log.

Usage: python3 bench/tools/tg_depth.py
"""
import os
import re
import statistics
import subprocess

PACK = "/home/jianwei/Strata-data/packs/iq3_s"
CORP = "/tmp/corp/mixed.txt"
FULL = "bench/tools/prompts/42k_mixed.txt"
CACHE = "/tmp/depth2_cache"

BASE = [
    "env", "HIP_VISIBLE_DEVICES=0", "./build-hip/strata-hip",
    "--pack", PACK,
    "--native", "/home/jianwei/Strata-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
    "--ple-gguf", "/home/jianwei/Strata-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf",
    "--expert-profile", "data/expert-profile.bin", "--expert-cache", "auto", "--prefill", "2048",
    "--spec", "2", "--spec-min-p", "0.5", "--adapt-every", "2", "--pcie-frac", "0.30",
    "--mtp", "/home/jianwei/Strata-data/models/IQ3_S/mtp/rt",
    "--kv", "int8", "--max-context", "262144", "--max-new", "64",
]

DEPTHS = [("0", 0), ("4K", 4096), ("8K", 8192), ("32K", 32768)]
REPS = 3
TAIL = (39000, 39512)


def build():
    if not os.path.exists(CORP):
        parts = []
        for cmd in (
            "find /home/jianwei/Documents/deepseek-harness/packages -name '*.ts' -type f | head -160 | xargs cat",
            "find /home/jianwei/Documents/llama.cpp/ggml/src/ggml-cuda -name '*.cu' -o -name '*.cuh' | head -60 | xargs cat",
            "find /home/jianwei/Documents/vllm-serving -name '*.py' | head -90 | xargs cat",
            "cat src/core/*.cpp src/prefill/*.cpp src/kernels/cuda/*.cu",
        ):
            out = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout
            parts.append(out[:200000])
        os.makedirs("/tmp/corp", exist_ok=True)
        open(CORP, "w").write("\n".join(parts))
    if not os.path.exists(FULL):
        subprocess.run(["python3", "bench/tools/tok_ascii.py", "--pack", PACK, "--text", CORP,
                        "--max-tokens", "42000", "--out", FULL], check=True)
    ids = open(FULL).read().strip().split(",")
    tail = ids[TAIL[0]:TAIL[1]]
    os.makedirs(CACHE, exist_ok=True)
    paths = {}
    for tag, n in DEPTHS:
        p = f"{CACHE}/depth_{tag}.txt"
        open(p, "w").write(",".join(ids[:n] + tail) + "\n")
        paths[tag] = p
    return ids, tail, paths


def run(tag, path):
    log = f"/tmp/depth2_{tag}.log"
    with open(log, "w") as f:
        subprocess.run(BASE + ["--tokens-file", path], stdout=f, stderr=subprocess.STDOUT, timeout=3600)
    t = open(log, errors="ignore").read()
    pp = re.search(r"prefill\s+(\d+) tokens in ([\d.]+) ms\s+->\s+([\d.]+) tok/s", t)
    tg = re.search(r"decode\s+(\d+) tokens in ([\d.]+) ms\s+->\s+([\d.]+) tok/s", t)
    sp = re.search(r"drafts accepted (\d+) of (\d+) \(([\d.]+)\), ([\d.]+) tokens per round", t)
    ca = re.search(r"expert cache (\d+) slots, ([\d.]+) GiB", t)
    return {"pp": float(pp.group(3)) if pp else None,
            "tg": float(tg.group(3)) if tg else None,
            "acc": float(sp.group(3)) if sp else None,
            "tpr": float(sp.group(4)) if sp else None,
            "cache": f"{ca.group(1)} slots, {ca.group(2)} GiB" if ca else "-"}


if __name__ == "__main__":
    ids, tail, paths = build()
    print(f"corpus {len(ids)} tokens; tail is tokens {TAIL[0]}-{TAIL[1]} ({len(tail)} tokens)", flush=True)
    data = {tag: [] for tag, _ in DEPTHS}
    for i in range(REPS):
        for tag, _ in DEPTHS:
            data[tag].append(run(tag, paths[tag]))
        print(f"  rep {i + 1}/{REPS} done", flush=True)

    print("\n== --max-context 262144 --kv int8 --spec 2, decode 64 tokens, same 512-token tail at every depth")
    print(f"{'depth':>6} {'pp tok/s':>9} {'tg median':>10} {'tg range':>14} "
          f"{'accept':>7} {'tok/round':>10}  expert cache")
    for tag, _ in DEPTHS:
        rs = data[tag]
        tgs = [r["tg"] for r in rs if r["tg"]]
        if not tgs:
            print(f"{tag:>6}   no decode line (see /tmp/depth2_{tag}.log)")
            continue
        pp = statistics.median([r["pp"] for r in rs if r["pp"]]) if any(r["pp"] for r in rs) else float("nan")
        acc = statistics.median([r["acc"] for r in rs if r["acc"]]) if any(r["acc"] for r in rs) else float("nan")
        tpr = statistics.median([r["tpr"] for r in rs if r["tpr"]]) if any(r["tpr"] for r in rs) else float("nan")
        print(f"{tag:>6} {pp:>9.1f} {statistics.median(tgs):>10.1f} "
              f"{f'{min(tgs):.1f}-{max(tgs):.1f}':>14} {acc:>7.3f} {tpr:>10.2f}  {rs[0]['cache']}", flush=True)
    print("\nDONE", flush=True)
