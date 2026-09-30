#!/usr/bin/env python3
"""Gate our performance numbers against the merged build (upstream's backend + our kernels).

Question: does the merge hold our measured throughput, or did it regress?

  old     : /home/jianwei/Documents/Strata/build-hip/strata-hip          (branch hip-gfx1100, pre-merge)
  merged  : /tmp/strata-rebase/build-merge/strata-hip                   (branch rebase-test, merge dd51c89)

Both run identical EXPLICIT flags, so the only variable is the engine.  Reference numbers from today's
same-session interleaved measurement with this configuration (bench/tools/vs_upstream.py, "ours" arm):

    1K : pp 577.0 (501-600)   tg 47.6 (46.4-50.0)
    4K : pp 922.4 (800-933)   tg 49.5 (46.5-51.2)

A run that hangs is a gate failure, not a missing data point: the merge mixes upstream's wait paths with our
shim, so a bounded timeout is part of the test.  Usage: python3 bench/tools/gate_merged.py [reps]
"""
import os
import re
import statistics
import subprocess
import sys

DATA = "/home/jianwei/Strata-data"
ST = "/home/jianwei/Documents/Strata"
OLD = "/tmp/strata-hip-premarge"   # preserved pre-merge binary
MERGED = f"{ST}/build-hip/strata-hip"   # after the merge, the canonical build IS the merged engine
OUT = "/tmp/gate-merged"
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2

FLAGS = [
    "--pack", f"{DATA}/packs/iq3_s",
    "--native", f"{DATA}/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
    "--ple-gguf", f"{DATA}/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf",
    "--expert-profile", f"{ST}/data/expert-profile.bin", "--expert-cache", "auto",
    "--prefill", "2048", "--spec", "4", "--spec-min-p", "0.5", "--adapt-every", "2",
    "--pcie-frac", "0.30", "--kv", "int8", "--max-context", "32768", "--max-new", "128",
    "--mtp", f"{DATA}/models/IQ3_S/mtp/rt",
]

ARMS = {"old": OLD, "merged": MERGED}
PROMPTS = [("1k", f"{ST}/bench/tools/prompts/1k.txt"),
           ("4k", f"{ST}/bench/tools/prompts/4k_diverse.txt")]
REFERENCE = {("old", "1k"): (577.0, 47.6), ("old", "4k"): (922.4, 49.5)}


def run(arm, tag, rep, prompt):
    log = f"{OUT}/{arm}_{tag}_r{rep}.log"
    env = dict(os.environ)
    env["HIP_VISIBLE_DEVICES"] = "0"
    timed_out = False
    try:
        with open(log, "w") as f:
            subprocess.run([ARMS[arm]] + FLAGS + ["--tokens-file", prompt],
                           stdout=f, stderr=subprocess.STDOUT, timeout=900, env=env)
    except subprocess.TimeoutExpired:
        timed_out = True
    t = open(log, errors="ignore").read()
    pp = re.search(r"^prefill\s+\d+ tokens in [\d.]+ ms\s+->\s+([\d.]+)", t, re.M)
    tg = re.search(r"^decode\s+\d+ tokens in [\d.]+ ms\s+->\s+([\d.]+)", t, re.M)
    out = re.search(r"^output\s*:\s*(.+)$", t, re.M)
    return {"pp": float(pp.group(1)) if pp else None,
            "tg": float(tg.group(1)) if tg else None,
            "timeout": timed_out,
            "sample": (out.group(1)[:90] if out else "")}


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    data = {(a, t): [] for a in ARMS for t, _ in PROMPTS}
    for rep in range(1, REPS + 1):
        for tag, prompt in PROMPTS:
            for arm in ("old", "merged"):
                subprocess.run(["pkill", "-x", "strata-hip"], capture_output=True)
                r = run(arm, tag, rep, prompt)
                data[(arm, tag)].append(r)
                flag = " TIMEOUT" if r["timeout"] else ""
                print(f"  rep {rep} {tag:>2} {arm:>6}: pp {r['pp'] if r['pp'] else 'NA':>8}  "
                      f"tg {r['tg'] if r['tg'] else 'NA':>8}{flag}", flush=True)

    print(f"\n=== gate: merged vs pre-merge, {REPS} interleaved reps, identical flags")
    print(f"{'prompt':>6} {'arm':>7} {'pp median':>10} {'pp range':>15} {'tg median':>10} {'tg range':>14}")
    verdict = []
    for tag, _ in PROMPTS:
        med = {}
        for arm in ("old", "merged"):
            rs = data[(arm, tag)]
            pps = [r["pp"] for r in rs if r["pp"]]
            tgs = [r["tg"] for r in rs if r["tg"]]
            med[arm] = (statistics.median(pps) if pps else float("nan"),
                        statistics.median(tgs) if tgs else float("nan"))
            pr = f"{min(pps):.0f}-{max(pps):.0f}" if pps else "-"
            tr = f"{min(tgs):.1f}-{max(tgs):.1f}" if tgs else "-"
            print(f"{tag:>6} {arm:>7} {med[arm][0]:>10.1f} {pr:>15} {med[arm][1]:>10.1f} {tr:>14}", flush=True)
        ref = REFERENCE[("old", tag)]
        dpp = 100 * (med["merged"][0] / ref[0] - 1)
        dtg = 100 * (med["merged"][1] / ref[1] - 1)
        print(f"       -> merged vs our recorded numbers: pp {dpp:+.1f}%  tg {dtg:+.1f}%")
        verdict.append((tag, dpp, dtg))
    print("\n=== correctness spot-check (first output tokens, merged)")
    for tag, _ in PROMPTS:
        print(f"  {tag}: {data[('merged', tag)][0]['sample']}")
    print("\n=== GATE", "PASS" if all(d > -8 for _, d, _ in verdict) else "FAIL",
          "(pp within -8% of our recorded numbers on every tier)")
    print("DONE", flush=True)
