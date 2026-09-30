# HIP/gfx1100 (RX 7900 XTX) vs the CUDA counterpart — summary

Distilled from `README.md` in this directory, which holds the full chronological record (~40 measurement
rounds) and every raw number.  Branch `hip-gfx1100`.  Everything here was measured on this box unless marked
"counterpart", which was not.

## Final numbers

Recommended configuration throughout:
`--expert-cache auto --prefill 2048 --spec 4 --spec-min-p 0.5 --adapt-every 2 --mtp`

| metric | this port (repeated median) | counterpart | ratio |
| --- | ---: | ---: | ---: |
| pp @1K | **408.5 tok/s** (5 runs, 396.6–410.5) | 419 | **97.5 %** |
| pp @4K | **676 tok/s** (tight cluster of 5) | 893 | **76 %** |
| pp @32K | **677.4 tok/s** (3 runs, ±0.7 %) | ~1,490 | 46 % |
| tg @1K | **45.0** (128 tokens) / **49.47** (256 tokens) | 50.5 | 89–98 % |
| tg @32K | **32.4** (32 tokens) / **37.87** (128 tokens) | 49.0 | 66–77 % |

Two caveats that matter more than the digits:

* **The counterpart numbers are external and unverified here.**  This box has no NVIDIA hardware (no
  `nvidia-smi`; only `build-hip`; `rocm-smi` reports AMD cards), so every *ratio* above inherits the
  counterpart's unknown configuration and protocol.
* **Decode throughput depends on the run length** — the first rounds of a short run pay the MTP prompt and draft
  warm-up — so tg must be compared at identical options *and* lengths.

## Cost models

**Prefill** (4K, 5.8–6.1 s): the expert stream is the spine.  The copy engine moves 54.6 GB of 2 MB expert blobs
at **13.6 GB/s when busy** — exactly the standalone probe's peak — so the link, not the kernel, sets the floor.
The measured phases are `wait copy` ~33 %, the expert MMQ GEMMs ~20 %, the gather ~7 %, the GDN family ~11 %,
`hc read` ~6 %, the WMMA attention ~3.5 %.  At 32K the stream is **537 GB at 11.3 GB/s = 80 % of the link**, and
the two streaming phases are 40 % of the prefill.

**Decode**: the round is linear in the verify window — **round ≈ 17.9 ms × window − 20.7 ms** (windows 4/5/6/8 →
52.0/68.0/85.7/123.2 ms) — i.e. volume-driven, with `--spec 3-4` the throughput optimum.  A 1K round decomposes
as **PCIe expert transfer 24.3 ms + doorbell/handshake waits 39.1 ms + arithmetic ~22 ms = ~85 ms**, matching the
measured 85.7 ms within 1 %.  The CPU expert pool is the pacing producer at ~27 GB/s, which is this machine's
memory bandwidth with all six physical cores in use.

## What was done, and what it was worth

| change | measured effect |
| --- | --- |
| RDNA3 WMMA GEMM for the dense projections (fp16), dispatched from `Gemm::f16` | 1.6–2.3× in isolation; **1K pp −13.4 %**, 4K −7.2 % |
| …the same for bf16 (`Gemm::bf16`); bf16 GEMMs were 204 of 731 ms of rocBLAS time | 1K pp **−18 % combined**, 4K −11 to −14 %; the `hc read` phase fell from 892–1,921 ms to **342 ms** |
| real WMMA prompt attention replacing the `mma16816` emulation | `qsa attn` **1,115 → 201 ms**; 1K pp wins all 3 interleaved pairs (408–420 tok/s); 32K pp **+28 %** |
| batched MMQ gather (16 experts per launch) | 246k fewer launches per 32K prefill; time neutral — the idle simply moved to `wait copy` |
| GDN recurrence load software-pipelining | `gdn rec` 1,694 → 1,580 ms at 32K (−6.7 % of the phase, 0.25 % overall) |
| MMQ gather batching + sub-phase timing + GDN sub-phases | instrumentation that made the above attributable |

Both WMMA GEMMs and the WMMA attention were verified independently: exact against a host fp64 reference on
exactly-representable inputs (`maxdev = 0.000e+00`, 0 NaN, all shapes), the gfx1100 code objects contain real
`v_wmma_*` instructions, and the 8-token seed reproduces the reference token for token under the emulated
attention path.

## What was refuted, and why it matters

* **`auto`-mode WMMA stubs.**  The fp16 WMMA path was silently compiled to its `return false` stub for a whole
  round because its guard tested `__gfx1100__`, which the *host* pass of this toolchain does not define (while
  `hipcc`, which the probes used, does).  Fixed by defining the arch macro from the build and verifying the
  linked symbol's size.  All other arch guards in the port were audited; this was the only one.
* **Dequant+WMMA for streamed experts** (`STRATA_PREFILL_MMQ=0`): **41 % slower** — a dense weight is
  dequantised once and reused across all tokens, but an expert block is streamed and used ~once, so expanding it
  to fp16 multiplies its bytes ~3.5×.  The quantised dp4a path is correct for experts.
* **A deeper copy ring** (`STRATA_PREFILL_RING=384`): worse, and the gap metric is itself noisy (±20 %), so the
  test is inconclusive rather than a clean refutation.
* **`--pcie-mode dma`** is genuinely ~0–6 % faster but **must not be the default**: the call site records
  issue #31, a host hang inside `cudaMemcpyAsync` on a driver lock, which the shader path exists to avoid.
* **A parallel GDN scan** would break the kernel's deliberate bit-parity with its predecessor, so it is off the
  table for this port.

## Measurement rules learned here

* **4K is the unstable tier** (±10 % and occasional +26–71 % outliers), while 1K and 32K are stable within ±2 %.
  The cause is **host-side interference** — another process taking CPU or host-memory bandwidth — not the GPU,
  not the sibling GPU on the shared ×8 slot, and not the tier: the slow runs have *identical* GPU-side counters
  and inflate only the phases that wait on the stream.  It is the same mechanism AGENTS.md recorded from the
  "download during a benchmark" case.  **4K comparisons need medians of ≥5 runs with outliers identified.**
* `pgrep -af <name>` matches its own wrapper when the wrapper's command line contains `<name>`; the bracket form
  `pgrep -af "[n]ame"` does not.  This produced false build alarms and can self-kill.
* Check binary mtimes *before* a comparison, not after: one set of A/B runs here straddled a rebuild.
* Waits are inflated by tracing, so kernel-trace *shares* of wait kernels are upper bounds; the engine's own
  phase table is the better instrument for attribution.

## The one remaining item

The expert MMQ kernels (~20 % of the 32K prefill) are llama.cpp's tuned dp4a code.  They are the only
substantive lever never attempted, and they are a project rather than a bounded change — with uncertain
headroom, since the quantised path is already established as correct and these formats are unpacking-bound
rather than int8-peak-bound.  Everything else is either taken, refuted, or attributed to the PCIe 4.0 ×8 link
and this machine's RAM bandwidth.
