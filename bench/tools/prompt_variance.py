#!/usr/bin/env python3
"""Does the prompt's topic range explain the run-to-run spread, or just make the numbers unrepresentative?

Three prompts of identical length (4,095 tokens) that differ only in diversity:
  * narrow  - one repository's C++ (what every previous number in this record used)
  * diverse - three projects in three languages (TypeScript, CUDA C++, Python)
  * prose   - markdown documentation rather than code

Configurations are interleaved within each repetition so host-side drift cannot be read as a prompt effect.
Records the engine's own expert counters (`experts streamed` / `resident`) as well as prefill throughput, so a
change in the expert working set is visible directly rather than inferred.

Usage: python3 bench/tools/prompt_variance.py
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
    "--spec", "4", "--spec-min-p", "0.5", "--adapt-every", "2", "--pcie-frac", "0.30",
    "--mtp", "/home/jianwei/Strata-data/models/IQ3_S/mtp/rt",
    "--max-context", "8192", "--max-new", "1",
]

PROMPTS = [("narrow", "bench/tools/prompts/4k_narrow.txt"),
           ("diverse", "bench/tools/prompts/4k_diverse.txt"),
           ("prose", "bench/tools/prompts/4k_prose.txt")]
REPS = 5


def run(prompt):
    with open("/tmp/pv_run.log", "w") as f:
        subprocess.run(BASE + ["--tokens-file", prompt], stdout=f, stderr=subprocess.STDOUT, timeout=900)
    t = open("/tmp/pv_run.log", errors="ignore").read()
    pp = re.search(r"prefill\s+\d+ tokens in ([\d.]+) ms\s+->\s+([\d.]+) tok/s", t)
    st = re.search(r"experts streamed (\d+).*?resident (\d+)", t)
    return (float(pp.group(2)) if pp else None,
            int(st.group(1)) if st else None,
            int(st.group(2)) if st else None)


if __name__ == "__main__":
    data = {k: [] for k, _ in PROMPTS}
    stream = {k: None for k, _ in PROMPTS}
    for i in range(REPS):
        for name, path in PROMPTS:
            pp, s, r = run(path)
            data[name].append(pp)
            if s:
                stream[name] = (s, r)
        print(f"  rep {i + 1}/{REPS} done", flush=True)

    print(f"\n{'prompt':>8} {'pp median':>10} {'pp range':>14} {'spread':>8} "
          f"{'streamed':>10} {'resident':>10}")
    for name, _ in PROMPTS:
        xs = [x for x in data[name] if x]
        if not xs:
            print(f"{name:>8}  no data")
            continue
        m = statistics.median(xs)
        spread = 100 * (max(xs) / m - 1)
        s, r = stream[name] or ("-", "-")
        print(f"{name:>8} {m:>10.1f} {f'{min(xs):.0f}-{max(xs):.0f}':>14} {spread:>7.0f}% "
              f"{s:>10} {r:>10}", flush=True)
    print("\nDONE", flush=True)
