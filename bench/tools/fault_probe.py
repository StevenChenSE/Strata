#!/usr/bin/env python3
"""Where does the variance come from?  Sample the engine's own page-fault and I/O counters while it runs.

For each run this records, from the engine process:
  * /proc/<pid>/stat   - minflt (minor faults) and majflt (MAJOR faults = a real disk read)
  * /proc/<pid>/io     - read_bytes (bytes actually fetched from the block layer) and rchar
  * /proc/<pid>/status - VmHWM (peak RSS)
and from the system:
  * /proc/pressure/{io,memory,cpu} - PSI, per-run deltas of the "total" stall time, which separates OUR paging
    from OTHER processes' pressure (the hypothesis that explains the variance is external interference, so PSI
    is the discriminator).
Plus, sampled from /proc/<pid>/smaps, the resident size of each mapping whose path contains .gguf - that is
whether the mapped token-embedding / PLE tables are actually resident or being faulted in.

Two phases: the default (--ple-io direct, unbuffered SSD reads) and --ple-io mmap after warming the shard into
the page cache.  Usage: python3 bench/tools/fault_probe.py
"""
import re
import subprocess
import time
import statistics

SHARD = "/home/jianwei/Strata-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf"

BASE = [
    "env", "HIP_VISIBLE_DEVICES=0", "./build-hip/strata-hip",
    "--pack", "/home/jianwei/Strata-data/packs/iq3_s",
    "--native", "/home/jianwei/Strata-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
    "--ple-gguf", SHARD,
    "--expert-profile", "data/expert-profile.bin", "--expert-cache", "auto", "--prefill", "2048",
    "--spec", "4", "--spec-min-p", "0.5", "--adapt-every", "2",
    "--mtp", "/home/jianwei/Strata-data/models/IQ3_S/mtp/rt",
    "--max-context", "8192", "--max-new", "1", "--tokens-file", "bench/tools/prompts/4k.txt",
]


def psi():
    out = {}
    for k in ("io", "memory", "cpu"):
        try:
            txt = open(f"/proc/pressure/{k}").read()
            out[k] = int(re.search(r"total=(\d+)", txt).group(1))
        except Exception:
            out[k] = 0
    return out


def stat_of(pid):
    """(minflt, majflt) from /proc/<pid>/stat, tolerating a comm with spaces."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            s = f.read()
        rest = s[s.rfind(")") + 2:].split()
        return int(rest[7]), int(rest[9])   # minflt, majflt (fields 10 and 12, 0-based 7 and 9 after comm)
    except Exception:
        return None, None


def io_of(pid):
    try:
        d = dict(l.split(": ") for l in open(f"/proc/{pid}/io").read().strip().split("\n"))
        return int(d.get("read_bytes", 0)), int(d.get("rchar", 0))
    except Exception:
        return 0, 0


def rss_of(pid):
    try:
        for l in open(f"/proc/{pid}/status"):
            if l.startswith("VmHWM:"):
                return int(l.split()[1])          # kB
    except Exception:
        pass
    return 0


def gguf_rss(pid):
    """resident kB per .gguf mapping, from smaps (header line, then its Rss line)."""
    out, cur = {}, None
    try:
        for l in open(f"/proc/{pid}/smaps"):
            if ".gguf" in l and ":" in l:
                cur = l.split("/")[-1].strip()[:28]
                out.setdefault(cur, 0)
            elif cur and l.startswith("Rss:"):
                out[cur] += int(l.split()[1])
                cur = None
    except Exception:
        pass
    return out


def one_run(extra_env=(), extra_args=()):
    p0 = psi()
    logpath = "/tmp/fault_run.log"
    logf = open(logpath, "w")
    proc = subprocess.Popen(list(extra_env) + BASE + list(extra_args),
                            stdout=logf, stderr=subprocess.STDOUT, text=True)
    pid = proc.pid
    f0 = stat_of(pid) or (0, 0)
    i0 = io_of(pid)
    peak_rss, maps, samples = 0, {}, []
    deadline = time.time() + 300          # never let one run hang the sweep
    while proc.poll() is None and time.time() < deadline:
        st = stat_of(pid)
        if st[0] is not None:
            samples.append(st)
        peak_rss = max(peak_rss, rss_of(pid))
        if len(samples) % 4 == 0:
            g = gguf_rss(pid)
            for k, v in g.items():
                maps[k] = max(maps.get(k, 0), v)
        time.sleep(0.5)
    logf.close()
    out = open(logpath, errors="ignore").read()
    if proc.poll() is None:
        proc.kill()
        proc.wait()
    f1 = samples[-1] if samples else (0, 0)
    i1 = io_of(pid)
    p1 = psi()
    pp = re.search(r"prefill\s+\d+ tokens in ([\d.]+) ms", out)
    return {
        "pp": float(pp.group(1)) if pp else float("nan"),
        "majflt": f1[1] - f0[1],
        "minflt": f1[0] - f0[0],
        "read_mb": (i1[0] - i0[0]) / 1e6,
        "rchar_mb": (i1[1] - i0[1]) / 1e6,
        "peak_rss_mb": peak_rss / 1024,
        "psi_io_ms": (p1["io"] - p0["io"]) / 1000,
        "psi_mem_ms": (p1["memory"] - p0["memory"]) / 1000,
        "psi_cpu_ms": (p1["cpu"] - p0["cpu"]) / 1000,
        "maps": maps,
    }


def phase(tag, n, extra_env=(), extra_args=()):
    print(f"\n== {tag}")
    print(f"{'run':>3} {'pp ms':>8} {'majflt':>7} {'minflt':>8} {'disk MB':>8} {'rchar MB':>9} "
          f"{'RSS MB':>7} {'psi_io':>7} {'psi_mem':>8} {'psi_cpu':>8}")
    rows = []
    for i in range(n):
        r = one_run(extra_env, extra_args)
        rows.append(r)
        print(f"{i:>3} {r['pp']:>8.0f} {r['majflt']:>7} {r['minflt']:>8} {r['read_mb']:>8.0f} "
              f"{r['rchar_mb']:>9.0f} {r['peak_rss_mb']:>7.0f} {r['psi_io_ms']:>7.0f} "
              f"{r['psi_mem_ms']:>8.0f} {r['psi_cpu_ms']:>8.0f}", flush=True)
        if r["maps"]:
            print(f"      gguf mappings resident: " +
                  ", ".join(f"{k}={v / 1024:.0f}MB" for k, v in sorted(r["maps"].items())), flush=True)
    pps = [r["pp"] for r in rows]
    if pps:
        print(f"   median pp {statistics.median(pps):.0f} ms, slowest {max(pps):.0f} ms "
              f"({100 * (max(pps) / statistics.median(pps) - 1):.0f}% over median)", flush=True)
    return rows


if __name__ == "__main__":
    phase("phase A: --ple-io direct (the default)", 6)
    print("\nwarming the PLE shard into the page cache...", flush=True)
    subprocess.run("dd if=%s of=/dev/null bs=4M status=none" % SHARD, shell=True)
    print("warmed.", flush=True)
    phase("phase B: --ple-io mmap (shard warm)", 4, extra_args=["--ple-io", "mmap"])
    print("\nDONE", flush=True)
