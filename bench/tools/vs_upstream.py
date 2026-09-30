#!/usr/bin/env python3
"""Is our HIP port better than upstream's merged AMD backend?  Interleaved, same pack/model/prompts.

  ours     : /home/jianwei/Documents/Strata/build-hip/strata-hip   (branch hip-gfx1100)
  upstream : /tmp/strata-main/build-hip/strata                     (origin/main, built in a worktree)

Each arm runs its OWN documented/tuned configuration, because that is the question being asked.  The differences
are recorded rather than silently equalised:

  ours     --prefill 2048 --pcie-frac 0.30 --adapt-every 2            (pinned arena; WMMA GEMM + WMMA attention)
  upstream --prefill 8192 --pcie-frac 0    --adapt-every 0            (their measured config; HIP MMQ + a
             --kv-resident 32768 --vram-reserve-mib 1024 --pool-workers 8   calibrated hipBLASLt table)

Both use --max-context 32768 --kv int8 and the same prompt files, so capacity and KV mode match; upstream's
--kv-resident 32768 means "not streamed" at that size (their docs).  Their mmap path is unusable here: it
requires experts.bin, which our pack does not have, so both arms use the pinned arena.

Runs are interleaved (ours, theirs, ours, theirs) so host drift cannot favour either arm.  Each run is logged
separately under /tmp/vs-upstream/.
"""
import os
import re
import statistics
import subprocess

DATA = "/home/jianwei/Strata-data"
ST = "/home/jianwei/Documents/Strata"
OURS = f"{ST}/build-hip/strata-hip"
THEIRS = "/tmp/strata-main/build-hip/strata"
OUT = "/tmp/vs-upstream"
REPS = int(os.environ.get("REPS", "3"))
MAXNEW = os.environ.get("MAXNEW", "128")
CTX = os.environ.get("CTX", "32768")

COMMON = [
    "--pack", f"{DATA}/packs/iq3_s",
    "--native", f"{DATA}/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
    "--ple-gguf", f"{DATA}/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf",
    "--expert-profile", f"{ST}/data/expert-profile.bin", "--expert-cache", "auto",
    "--spec", "4", "--spec-min-p", "0.5",
    "--mtp", f"{DATA}/models/IQ3_S/mtp/rt",
    "--kv", "int8", "--max-context", CTX, "--max-new", MAXNEW,
]

ARMS = {
    "ours": (OURS, {"HIP_VISIBLE_DEVICES": "0"},
             ["--prefill", "2048", "--adapt-every", "2", "--pcie-frac", "0.30"]),
    "upstream": (THEIRS,
                 {"HIP_VISIBLE_DEVICES": "0", "STRATA_PREFILL_MMQ": "1", "STRATA_PREFILL_RING": "96",
                  "STRATA_IO_THREADS": "32",
                  "STRATA_HIPBLASLT_TUNING": "/tmp/strata-main/tools/hip/gfx1100-hipblaslt-100100.txt"},
                 ["--prefill", "8192", "--adapt-every", "0", "--pcie-frac", "0",
                  "--kv-resident", "32768", "--vram-reserve-mib", "1024", "--pool-workers", "8"]),
}

PROMPTS = [("1k", f"{ST}/bench/tools/prompts/1k.txt"),
           ("4k", f"{ST}/bench/tools/prompts/4k_diverse.txt")]


def run(arm, tag, rep, prompt):
    binary, env_extra, flags = ARMS[arm]
    env = dict(os.environ)
    env.update(env_extra)
    log = f"{OUT}/{arm}_{tag}_r{rep}.log"
    cmd = [binary] + COMMON + flags + ["--tokens-file", prompt]
    with open(log, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=1800, env=env)
    t = open(log, errors="ignore").read()
    pp = re.search(r"^prefill\s+\d+ tokens in [\d.]+ ms\s+->\s+([\d.]+)", t, re.M)
    tg = re.search(r"^decode\s+\d+ tokens in [\d.]+ ms\s+->\s+([\d.]+)", t, re.M)
    return (float(pp.group(1)) if pp else None, float(tg.group(1)) if tg else None)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    data = {(a, t): [] for a in ARMS for t, _ in PROMPTS}
    for rep in range(1, REPS + 1):
        for tag, prompt in PROMPTS:
            for arm in ("ours", "upstream"):
                subprocess.run(["pkill", "-x", "strata"], capture_output=True)
                subprocess.run(["pkill", "-x", "strata-hip"], capture_output=True)
                pp, tg = run(arm, tag, rep, prompt)
                data[(arm, tag)].append((pp, tg))
                print(f"  rep {rep} {tag:>2} {arm:>8}: pp {pp if pp else 'NA':>8}  tg {tg if tg else 'NA':>8}",
                      flush=True)

    print(f"\n=== interleaved x{REPS} medians (ctx={CTX}, kv=int8, max-new={MAXNEW}, spec=4)")
    print(f"{'prompt':>6} {'arm':>9} {'pp median':>10} {'pp range':>16} {'tg median':>10} {'tg range':>14}")
    for tag, _ in PROMPTS:
        for arm in ("ours", "upstream"):
            pps = [p for p, _ in data[(arm, tag)] if p]
            tgs = [g for _, g in data[(arm, tag)] if g]
            pr = f"{min(pps):.0f}-{max(pps):.0f}" if pps else "-"
            tr = f"{min(tgs):.1f}-{max(tgs):.1f}" if tgs else "-"
            print(f"{tag:>6} {arm:>9} {statistics.median(pps) if pps else float('nan'):>10.1f} {pr:>16} "
                  f"{statistics.median(tgs) if tgs else float('nan'):>10.1f} {tr:>14}", flush=True)
    print("\nDONE", flush=True)
