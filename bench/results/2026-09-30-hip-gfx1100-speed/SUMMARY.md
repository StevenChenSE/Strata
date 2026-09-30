# HIP/gfx1100 (RX 7900 XTX) vs the CUDA counterpart — summary

Distilled from `README.md` in this directory, which holds the full chronological record (~40 measurement
rounds) and every raw number.  Branch `hip-gfx1100`.  Everything here was measured on this box unless marked
"counterpart", which was not.

## Final numbers

Recommended configuration throughout:
`--expert-cache auto --prefill 2048 --spec 4 --spec-min-p 0.5 --adapt-every 2 --mtp`

| metric | this port (repeated median) | counterpart | ratio |
| --- | ---: | ---: | ---: |
| pp @1K | **602.5 tok/s** | 419 | **144 %** |
| pp @4K | **1,016 tok/s** | 893 | **114 %** |
| pp @32K | **1,079.2 tok/s** | ~1,490 | 72 % |
| tg @1K | **46.5** (128 tokens) | 50.5 | 92 % |
| tg @32K | **46.3** (32 tokens) | 49.0 | **94 %** |

These are the **PCIe 4.0 ×16** numbers, measured after the second card was removed from the shared slot (2 MB
H2D blobs measure 25.8 GB/s against 13.5 at ×8).  On the old ×8 link the same configuration gave pp 408.5 /
676–707 / 677.4 and tg 45.0 / 32.4, so the link was the long-context ceiling: lifting it moved prefill by
+44–59 % and 32K decode by +43 %.  The 4K prompt had to be regenerated and routes slightly differently, so that
tier's gain is the least like-for-like of the three.

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

The GDN recurrence prefetch is worth 0.25 % overall (6.7 % of its phase), and the batched MMQ gather is time
neutral - the idle simply moved to `wait copy`.  Both are kept because they are real and reversible, not because
they are large.

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
* **The copy ring is a lever, not a dead end** (corrected): an earlier A/B compared 384 with 384 - the default
  already *is* 384, because the expert arena is fully pinned - so it was a no-op.  Measured properly, 96 -> 384 is
  worth **+6.2 %** and 384 -> 512 is neutral, so the default is the optimum.
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

## Nothing is left: every item is taken, refuted, measured to its ceiling, or hardware

The expert MMQ was the last open item and it is now measured rather than guessed.  Its rate is **16.6 TFLOPS,
flat in T** (from the real routing: k = 10, so 4095 × 10 × 48 = 1,965,600 pairs at 4.92M MACs each), and a probe
using the intrinsic llama.cpp uses here (`__builtin_amdgcn_sudot4`) puts the practical ceiling at **21.7 TOPS**
for a realistic 2-bit block dot (26.3 TOPS for the bare dot).  The engine is therefore at **76 % of its
ceiling**, leaving at most 1.31x on the MoE phases - 20 % of the 4K prefill - and that bound assumes a redesign
pays nothing for the int8->fp16 conversion it would need.  (This also corrects the denominator: the dot peak is
26.3 TOPS, not the card-quoted ~123 TOPS int8, so the engine was 63 % of the right ceiling all along.)

The copy ring is at its measured optimum: **96 -> 384 is worth +6.2 %** (14 ms of buffer against ~26 ms stalls),
and 384 -> 512 is neutral while costing 2 % of expert-cache residency.  The residual `wait copy` idle is
**host-driven**, not a GPU pipeline problem: the issuer keeps the copy stream exactly `ring` entries ahead with
waits that are already satisfied at enqueue time, so only stalls longer than 58 ms show up - and those are the
host's (per-chunk PLE/embedding work, and interference from whatever else this box is doing, which is also what
makes the 4K tier the unstable one).

So the remaining gaps were the PCIe link and this machine's host side - and that prediction was then tested by
removing the second card from the shared slot.  On the full ×16 link the same configuration gives the numbers at
the top of this file: **+47 % prefill at 1K, +44–50 % at 4K, +59 % at 32K and +43 % on 32K decode**, with the 4K
`wait copy` phase collapsing from ~2,000 ms to 378 ms.  The link was indeed the long-context ceiling; what
remains now is the host side and the ×16 link itself.  Everything else was taken, refuted, or measured.

