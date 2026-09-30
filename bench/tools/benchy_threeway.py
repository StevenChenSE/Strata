#!/usr/bin/env python3
"""Three-way llama-benchy comparison: our branch, upstream master, and the merge.

Each arm gets its own server config, its own documented engine args, and (for master) its documented environment.
The server is started and stopped per arm by PID - never with a `pkill -f` pattern, which in this project has five
times killed the wrapper that contained the very command it was launching.

Run it with plain python3; it invokes `uv run ...` for the server (ephemeral deps) and `uvx llama-benchy` for the
benchmark.  Requires uv on PATH and the HF tokenizer produced by bench/tools/make_hf_tokenizer.py.

Usage: python3 bench/tools/benchy_threeway.py [--runs N] [--depths 0,4096,8192,32768] [--arms ours,master,merged]
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

DATA = "/home/jianwei/Strata-data"
ST = "/home/jianwei/Documents/Strata"
PORT = 8095
TOKENIZER = "/tmp/iq3_s-hf"          # from bench/tools/make_hf_tokenizer.py, exact against the engine
MODEL = "qwen3.8-flash-next-iq3_s"
GGUF = f"{DATA}/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"

COMMON = ["--serve",
          "--pack", f"{DATA}/packs/iq3_s",
          "--native", GGUF,
          "--ple-gguf", f"{DATA}/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf",
          "--expert-profile", f"{ST}/data/expert-profile.bin",
          "--expert-cache", "auto", "--spec", "2", "--spec-min-p", "0.5",
          "--kv", "int8", "--max-context", "65536",
          "--mtp", f"{DATA}/models/IQ3_S/mtp/rt"]

ARMS = {
    # our branch, pre-merge, on the tuned config from tools/run-tuned.sh
    "ours": dict(exe=f"{ST}/build-hip/strata-hip", env={},
                 args=COMMON + ["--prefill", "2048", "--adapt-every", "2", "--pcie-frac", "0.30"]),
    # upstream origin/main, on its own documented measured configuration
    "master": dict(exe="/tmp/strata-main/build-hip/strata",
                   env={"STRATA_PREFILL_MMQ": "1", "STRATA_PREFILL_RING": "96", "STRATA_IO_THREADS": "32",
                        "STRATA_HIPBLASLT_TUNING": "/tmp/strata-main/tools/hip/gfx1100-hipblaslt-100100.txt"},
                   args=COMMON + ["--prefill", "8192", "--adapt-every", "0", "--pcie-frac", "0",
                                  "--kv-resident", "32768", "--vram-reserve-mib", "1024", "--pool-workers", "8"]),
    # the merge of the two
    "merged": dict(exe="/tmp/strata-rebase/build-merge/strata-hip", env={},
                   args=COMMON + ["--prefill", "2048", "--adapt-every", "2", "--pcie-frac", "0.30"]),
}


def kill_engines():
    """By process NAME, so it can never match this script or a wrapper."""
    for name in ("strata", "strata-hip"):
        subprocess.run(["pkill", "-x", name], capture_output=True)


def start_server(arm, cfg, log_path):
    conf = dict(exe=cfg["exe"], cwd=ST, args=cfg["args"], tokenizer=f"{DATA}/packs/iq3_s/tokenizer",
                model_name=MODEL, host="127.0.0.1", port=PORT)
    conf_path = f"/tmp/bench-{arm}.json"
    with open(conf_path, "w") as f:
        json.dump(conf, f, indent=2)
    env = dict(os.environ)
    env.update(cfg["env"])
    log = open(log_path, "w")
    p = subprocess.Popen(["uv", "run", "--no-project", "--with", "regex", "--with", "jinja2",
                          "--with", "tokenizers", "--with", "numpy",
                          "python", "-m", "serve.server", "--engine", "strata",
                          "--config", conf_path, "--port", str(PORT)],
                         cwd=ST, env=env, stdout=log, stderr=subprocess.STDOUT)
    ready = False
    for _ in range(120):                      # up to ~4 min for the model load
        if p.poll() is not None:
            break
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=3) as r:
                if r.status == 200:
                    ready = True
                    break
        except Exception:
            time.sleep(2)
    return p, log, ready


def run_benchy(depths, runs, out_path):
    cmd = ["uvx", "llama-benchy", "--base-url", f"http://127.0.0.1:{PORT}/v1",
           "--model", "local/iq3_s", "--served-model-name", MODEL, "--tokenizer", TOKENIZER,
           "--pp", "2048", "--tg", "32", "--depth"] + [str(d) for d in depths] + \
          ["--exact-tg", "--runs", str(runs), "--format", "md"]
    with open(out_path, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=2400)
    t = open(out_path, errors="ignore").read()
    rows = [ln for ln in t.splitlines() if ln.startswith("| local/iq3_s")]
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--depths", default="0,4096,8192,32768")
    ap.add_argument("--arms", default="ours,master,merged")
    a = ap.parse_args()
    depths = [int(x) for x in a.depths.split(",")]
    results = {}
    for arm in a.arms.split(","):
        cfg = ARMS[arm]
        print(f"\n=== {arm}: {cfg['exe']}", flush=True)
        kill_engines()
        p, log, ready = start_server(arm, cfg, f"/tmp/bench-{arm}-server.log")
        if not ready:
            print(f"  SERVER FAILED to become ready - see /tmp/bench-{arm}-server.log", flush=True)
            tail = open(f"/tmp/bench-{arm}-server.log", errors="ignore").read()[-500:]
            print("  " + tail.replace("\n", "\n  "), flush=True)
            p.terminate()
            continue
        print("  server ready; running llama-benchy", flush=True)
        rows = run_benchy(depths, a.runs, f"/tmp/bench-{arm}-benchy.txt")
        for r in rows:
            print("  " + r, flush=True)
        results[arm] = rows
        p.terminate()
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()
        log.close()
        kill_engines()
        time.sleep(5)
    print("\n=== all arms done", flush=True)
