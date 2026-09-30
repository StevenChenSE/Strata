#!/usr/bin/env python3
"""TG at long context with 256K capacity and INT8 KV.

The user's configuration: --max-context 262144 --kv int8.  Measuring tg there requires the sequence to actually
BE that long, so this builds a genuinely long prompt (a mix of four codebases, so the routing and the MTP
acceptance are not flattered by a single-repo prompt) and then decodes at three filled-context sizes to give a
tg-vs-context curve rather than one point:

    64K  -> --max-context 65536
    128K -> --max-context 131072
    256K -> --max-context 262144   (the requested configuration)

--spec 2 is used because the earlier sweep showed the optimum draft length shrinks with context (4 at 1K, 2 at
32K).  Each run logs to /tmp/kv256_<tier>.log so the evidence survives, and prints pp, tg, acceptance and
tokens/round.

Usage: python3 bench/tools/kv256_tg.py
"""
import os
import re
import subprocess
import sys

PACK = "/home/jianwei/Strata-data/packs/iq3_s"
CORP = "/tmp/corp/mixed.txt"
FULL = "bench/tools/prompts/256k_mixed.txt"

BASE = [
    "env", "HIP_VISIBLE_DEVICES=0", "./build-hip/strata-hip",
    "--pack", PACK,
    "--native", "/home/jianwei/Strata-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
    "--ple-gguf", "/home/jianwei/Strata-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf",
    "--expert-profile", "data/expert-profile.bin", "--expert-cache", "auto", "--prefill", "2048",
    "--spec", "2", "--spec-min-p", "0.5", "--adapt-every", "2", "--pcie-frac", "0.30",
    "--mtp", "/home/jianwei/Strata-data/models/IQ3_S/mtp/rt",
    "--kv", "int8",
]

TIERS = [("64K", 65536, 65536 - 64), ("128K", 131072, 131072 - 64), ("256K", 262144, 261000)]


def build_prompt():
    """~300 KB from each of four codebases -> one mixed corpus, then 261,000 tokens of it."""
    if not os.path.exists(CORP):
        parts = []
        for cmd in (
            "find /home/jianwei/Documents/deepseek-harness/packages -name '*.ts' -type f | head -120 | xargs cat",
            "find /home/jianwei/Documents/llama.cpp/ggml/src/ggml-cuda -name '*.cu' -o -name '*.cuh' | head -50 | xargs cat",
            "find /home/jianwei/Documents/vllm-serving -name '*.py' | head -80 | xargs cat",
            "cat src/core/*.cpp src/prefill/*.cpp src/kernels/cuda/*.cu",
        ):
            out = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout
            parts.append(out[:300000])
        os.makedirs("/tmp/corp", exist_ok=True)
        with open(CORP, "w") as f:
            f.write("\n".join(parts))
    if not os.path.exists(FULL):
        subprocess.run(["python3", "bench/tools/tok_ascii.py", "--pack", PACK, "--text", CORP,
                        "--max-tokens", "261000", "--out", FULL], check=True)
    ids = open(FULL).read().strip().split(",")
    return ids


def prefix(ids, n, path):
    with open(path, "w") as f:
        f.write(",".join(ids[:n]) + "\n")
    return path


if __name__ == "__main__":
    ids = build_prompt()
    print(f"prompt has {len(ids)} tokens", flush=True)
    for tag, ctx, n in TIERS:
        if n > len(ids):
            print(f"{tag}: prompt too short ({len(ids)} < {n}), skipping", flush=True)
            continue
        pf = prefix(ids, n, f"/tmp/kv256_prompt_{tag}.txt")
        log = f"/tmp/kv256_{tag}.log"
        print(f"\n== {tag}: --max-context {ctx} --kv int8, prompt {n} tokens, decode 64", flush=True)
        cmd = BASE + ["--max-context", str(ctx), "--max-new", "64", "--tokens-file", pf]
        with open(log, "w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=3600)
        t = open(log, errors="ignore").read()
        pp = re.search(r"prefill\s+(\d+) tokens in ([\d.]+) ms\s+->\s+([\d.]+) tok/s", t)
        tg = re.search(r"decode\s+(\d+) tokens in ([\d.]+) ms\s+->\s+([\d.]+) tok/s", t)
        sp = re.search(r"drafts accepted (\d+) of (\d+) \(([\d.]+)\), ([\d.]+) tokens per round", t)
        cache = re.search(r"expert cache (\d+) slots, ([\d.]+) GiB", t)
        print(f"   pp  {pp.group(3) if pp else '?':>8} tok/s ({pp.group(2) if pp else '?'} ms)", flush=True)
        print(f"   tg  {tg.group(3) if tg else '?':>8} tok/s ({tg.group(2) if tg else '?'} ms)", flush=True)
        if sp:
            print(f"   draft acceptance {sp.group(3)}, {sp.group(4)} tokens/round", flush=True)
        if cache:
            print(f"   expert cache {cache.group(1)} slots, {cache.group(2)} GiB", flush=True)
        if not tg:
            print("   (no decode line - see " + log + ")", flush=True)
    print("\nDONE", flush=True)
