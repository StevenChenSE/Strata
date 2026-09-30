# HIP / gfx1100 baseline — first working decode (engine 0.1.23, branch `hip-gfx1100`)

RX 7900 XTX 24 GB (gfx1100, ROCm 7.14), single GPU, native IQ3_S pack
(`/home/jianwei/Strata-data/packs/iq3_s`, 46.84 GiB expert arena in pinned host memory, 5341-slot
VRAM expert cache), shards in `/home/jianwei/Strata-data/models/IQ3_S/`.  Greedy.  This is the first
run in which decode completes at all (see `.rocm-eval/REVIEW-decode-hang.md`: the blocker was a
one-line unsigned underflow in `compat/hip/cuda_runtime.h`).

## Measured

| run | prompt | new | prefill (pp) | decode (tg) | speculation |
| --- | ---: | ---: | ---: | ---: | --- |
| 1K tier | 1,023 tok (`--tokens-file`, see caveat) | 256 | **344.6 tok/s** (2968.7 ms) | **40.02 tok/s** (6397.5 ms) | 65 rounds of 4, 3.95 tok/round, drafts 192/192 |
| smoke | 12 tok | 8 | ~16-17 tok/s\* | 16.4-19.3 tok/s | 8 rounds of 4, 1.00 tok/round |

\* the 12-token "prefill" number is dominated by fixed per-run costs (lent-slot refill, first-chunk
setup); it is not a throughput figure.  It is recorded only because it is the reproducibility test
(identical token sequence across 4 runs).

## Compared with the CUDA counterpart (IQ3_S, RTX 5070, `bench/results/2026-09-29-speed-0122`)

| metric | CUDA @1K (MTP `--spec 4 --spec-min-p 0.5`) | HIP @1K | ratio |
| --- | ---: | ---: | --- |
| prompt tok/s | 419 | 344.6 | 1.22x slower |
| output tok/s | 50.5 | 40.0 | 1.26x slower |

Different GPUs (RTX 5070 12 GB / PCIe 5.0 vs 7900 XTX 24 GB / 14.3 GB/s measured H2D), so treat this as
"same order, not the same machine" rather than a like-for-like benchmark.  Worth noting the CUDA box has
*less* VRAM and its engine still reaches 419/50.5, while the HIP run's PCIe probe reports only
**14.3 GB/s host->device**, which is low for this card and is a first-order suspect for anything that
streams experts.

## Where the decode time goes (engine's own per-round breakdown, 98.4 ms/round wall)

```
verify window   wait for rings 29.473   pool 66.144   host 0.022   commit 0.888  ms/round
pool multi      gate/up 27.768  quantize 0.072  down 16.676 ms/round; 26.4 GB/s over the rows phases; CPU pool call 46.636 ms/round
pcie experts    4.51 distinct experts per layer read over PCIe (share 77/256 of the misses)
adaptive tier   192 experts swapped into the VRAM tier (every 4 rounds, 0.053 ms/round)
```

The wall time per round (98.4 ms) is almost exactly `wait 29.5 + pool 66.1 + commit 0.9` — i.e. **the CPU
expert pool and the GPU layer work are serialised, not overlapped**: the dependency chain per layer is
`pre(l) -> pool(l) -> post(l) -> pre(l+1)`, so the only overlap available is the pool of layer `l`
against `pre(l)`'s tail.  The pool (66 ms of 98) is therefore the first target for tg, and the GPU's
29.5 ms of 48 layers (0.61 ms/layer) the second.

## Caveats

* The 1K prompt is 80 repeats of the 13-token smoke prompt (`tools/strata_tokenizer.py` needs the
  `regex` module, which is not installed here), so the drafter predicts it almost perfectly: 3.95
  tokens/round at 100 % acceptance.  **Real text will accept fewer drafts**, so the 40.02 tok/s figure
  is an upper bound on this prompt shape, and the CUDA row uses real code-agent prompts.  A
  non-repetitive prompt, or `--spec-min-p 1.0` to suppress drafting, is the next measurement.
* One run per cell; decode speed moves a few percent with the text (as the CUDA README says).
* MTP weights were never fetched (`--mtp` would pull ~5 GB from HF, not authorised), so HIP speculation
  here comes from the engine's built-in drafter, not MTP.

## Clean measurement: a non-repetitive prompt (drafting does not help)

`/tmp/prompt_rand1k.txt` is 1,024 pseudo-random ids (seed 7, range 1000..150000), so the drafter cannot
predict anything.  Two runs, 256 generated:

| run | prefill (pp) | decode (tg) | speculation |
| --- | ---: | ---: | --- |
| default (`--spec 2`) | 344.37 tok/s (2970.6 ms) | **26.01 tok/s** (9841.3 ms) | 252 rounds of 4, 1.02 tok/round, drafts 5/254 |
| `--spec-min-p 1.0` | 344.65 tok/s (2968.2 ms) | **26.72 tok/s** (9580.1 ms) | 250 rounds of 4, 1.02 tok/round, drafts 6/252 |

So on text the drafter cannot predict, HIP does **26.0-26.7 tok/s** and pp is **344.4-344.7 tok/s**
(three prompt shapes now agree to within 0.3 %, so pp is a solid number).  The 40.02 tok/s figure above
is what a *perfectly predictable* prompt buys (3.95 tokens/round).

## Attribution (my own, from the engine's per-round lines)

Per round, wall ≈ `wait for rings + pool + commit` (the three terms sum to the wall within 2 %):

| | random prompt | repetitive prompt |
| --- | ---: | ---: |
| wall per round | 39.1 ms | 98.4 ms |
| wait for rings | 22.4 ms | 29.5 ms |
| pool | 14.2 ms | 66.1 ms |
| commit | 0.7 ms | 0.9 ms |
| CPU experts / layer (distinct) | 2.82 | 12.04 |
| PCIe experts / layer | 0.58 | 4.51 |

1. **The two terms are serialised, not overlapped.**  The layer dependency chain is
   `pre(l) -> pool(l) -> post(l) -> pre(l+1)`, and `post(l)` consumes the plan and the CPU rows the pool
   produces, so the pool sits on the critical path; the only GPU work that can overlap it is `pre(l)`'s
   tail (shared expert + quantize).  That is why wall ≈ wait + pool.
2. **The pool scales with the number of CPU-resident experts** (14.2 ms at 2.8/layer, 66.1 ms at
   12/layer ≈ 4.9 ms per round per expert-per-layer) and is **host-RAM-bandwidth-bound**:
   `26.4-27.7 GB/s over the rows phases` against a machine that measures **45.4 GB/s** for a strided
   read (my `mbr.c`) — so the pool is at ~60 % of the achievable read bandwidth.
3. **Host facts that bound this** (measured): Ryzen 5 9600X, **6 physical cores / 12 threads**, 96 GB
   RAM, `--pool-workers 0` = every physical core minus the host's = **5 workers**; THP is `madvise`
   (not `always`) and `HugePages_Total = 0`, so the 46.84 GiB expert arena is on **4 KB pages**.
4. **The tg gap against CUDA is mostly speculation, not round latency.**  CUDA's 50.5 tok/s at 2.77
   tokens/round implies ~54.9 ms/round; HIP's random-prompt round is 39.1 ms.  HIP is *faster per round*
   and loses on tokens/round (1.02 vs 2.77, i.e. MTP): the same round rate with CUDA's acceptance would
   put HIP near 70 tok/s.  (MTP weights were never fetched — not authorised.)
5. **pp is within 1.2x of CUDA** (344.5 vs 419) on a smaller, different GPU, with the prompt path
   spending 161 ms of 2,921 ms in PLE and 25 ms DMA'ing 9,229 expert blobs.

## Ranked levers (next experiments, cheapest first)

1. `--expert-cache` size (4096 -> 8192/auto): more resident experts directly shrinks the pool term.
2. `--pool-workers` (5 -> more) and/or huge pages for the arena (`madvise(MADV_HUGEPAGE)` on the
   registered arena, or a configured hugetlb pool): the pool is at 60 % of measured read bandwidth.
3. Overlapping pool(l) with GPU work: needs the plan/CPU-row hand-off redesigned so `post(l)` can start
   its VRAM groups before the CPU rows are ready (the A/B/CPU doorbells already separate those waits in
   principle — worth checking whether the GPU actually runs the VRAM groups before the CPU flag rises).
4. pp: `--prefill` chunk size sweep, and the prompt path's per-stage attribution (no prompt-path
   profiler was found; the verify window's `STRATA_VERIFY_PROFILE` does not cover it).
5. Speculation: real-text acceptance (needs a tokenizer; `tools/strata_tokenizer.py` needs the `regex`
   module, which is not installed and `pip` is unavailable) or MTP weights (not authorised).

## Round 2 of the perf investigation: pp phase attribution, the prefill ring, and the PLE

### pp phase breakdown (`STRATA_PREFILL_TIMING=1`, 1,023 tokens, one chunk, best config)

```
GPU timeline 2428 ms, wall 2429 ms, host staging 21 ms:
  gather         632 (17.5%)   <- staging the expert blobs
  wait copy      574 (15.9%)   <- the GPU *idle*, waiting on an expert DMA event
  embed+steps    502 (13.9%)
  gdn            554 (15.4%)
  hc read        284 ( 7.9%)
  dequant+gemm   623 (17.3%)   <- dequant 250, gemm gate/up 229, gemm down 144
  qsa proj+attn  345 ( 9.6%)   <- the emulated mma16816 prompt attention is NOT dominant
  router+shared+combine+grouping  87 ( 2.4%)
```

So a third of the prompt is expert *staging*, and **574 ms of it is the GPU idling on copy events**.
The emulated prompt attention, which PORT-STATUS flagged as the prefill suspect, is only ~10 %.

### The staging ring is already tuned (`STRATA_PREFILL_RING`)

The ring depth is chosen by the pinned-memory share (384 slots when ~all experts are DMA'd from pinned
RAM, else 96) and can be overridden by env.  Measured sweep on this box (all with a warm PLE, 6,947
experts streamed):

| ring | 96 | 192 | 384 | 512 |
| --- | ---: | ---: | ---: | ---: |
| pp tok/s | 390.0 | 379.3 | **421.2** | 394.4 |

384 is both the default and the optimum, so there is nothing to win there; the ~574 ms of exposed copy
wait would need deeper *issue* lead (or per-layer prefetch) rather than a bigger ring.

### The PLE table: does it need to be in RAM? **No — measured.**

The table is ~25 GiB inside the 27.5 GiB second shard.  The default `--ple-io direct` is documented as
*"unbuffered SSD reads, the table never enters RAM or the file cache"*; `--ple-io mmap` is the A/B arm.

| PLE configuration | PLE gather (1,023 tokens) | pp |
| --- | ---: | ---: |
| `direct` (default) | **157 ms** | 402 tok/s |
| `--ple-io mmap`, table fully in page cache (`buff/cache` 38 -> 64 GB after `cat`) | **2,300 ms** | 217 tok/s |
| `direct` + `--ple-inflight 256` | 164 ms | 391 tok/s |
| `direct` + `--ple-row-cache 8388608` (90 MB -> 720 MB) | 163 ms | 400 tok/s |

* Warming the table into RAM changes **nothing** for the default mode (157 vs 161 ms) — it is unbuffered
  by design, so the page cache is never consulted.
* The RAM-resident `mmap` arm is **15x slower** than the unbuffered reader (its own comment in
  `ngram.cpp` says "SIXTEEN SERIAL PAGE FAULTS, MEASURED" per token).
* The two direct-mode tuning knobs do not move it either, so ~157 ms is that reader's floor here.
* Practical conclusion: **do not load the PLE into RAM.**  It would also compete with the 46.84 GiB
  *pinned, unevictable* expert arena on a 92 GiB box.  If the 6 % ever matters, the levers are a smaller
  table or batching the row reads by address, not residency.

### Measurement hygiene: pp has ~10 % run-to-run spread

Two runs with the *same* configuration (cache 8192, `--prefill 2048`, `--max-new 256`, ring 384) gave
2,633 ms and 2,950 ms of prefill.  One contributor is measurable — the PLE chunk-setup phase swung
157 -> 491 ms between runs — the rest is unexplained.  **Compare pp only across runs that repeat the
configuration, and report a spread rather than a single number.**  Best observed pp at 1K is
**~400-420 tok/s** (ring 384, warm PLE, `--max-new 8`), i.e. at parity with the CUDA counterpart's 419
on real prompts; the conservative same-protocol number (256 generated) is ~347-390.

## Follow-up: "is RAM really slower than NVMe?" — no, it is latency-hiding

The PLE table rows are 90 B (`PLE_ROW_BYTES = (160/32)*18`), 16 heads per token, so a 1,023-token chunk
fetches **16,368 rows**.  The two arms' per-row costs differ 15x (9.6 us vs 140 us), which looks like
"RAM slower than NVMe" — it is not.  They differ in pipelining:

* `direct`: an I/O thread submits the whole batch with up to `--ple-inflight` outstanding reads, a
  90 MB row cache absorbs repeats, and the dequant runs over one contiguous buffer.
* `mmap`: `for (i < n) read_row(rows[i], ...)` — serial random access into a 27.5 GiB mapping (the code's
  own comment: "SIXTEEN SERIAL PAGE FAULTS, MEASURED").  140 us/row is a *major* fault, so those pages
  were not resident either: the same run commits a 46.84 GiB pinned arena and reads 52 GiB of shard 1,
  which cannot leave 27.5 GiB of table in the page cache.

Proof that depth, not medium, is the variable — cripple the *same* unbuffered reader:

| direct-reader setting | prefill (1,023 tokens) |
| --- | ---: |
| `--ple-inflight 64` (default) | 2,542 ms |
| `--ple-inflight 1` | 3,999 ms |
| `--ple-inflight 1 --ple-row-cache 0` | 3,983 ms |
| `--ple-row-cache 0` (depth 64) | 2,587 ms |

Dropping the queue depth from 64 to 1 costs **+1.46 s**, the same order as the mmap arm's penalty;
the row cache is worth ~2 %.  So "load the PLE into RAM" is not the question — *serial vs pipelined
row access* is, and the default already pipelines.

## `--pcie-frac`: measured worse, leave it automatic

Forcing the PCIe miss share up (the engine auto-scales it to 0.30 from the 14.3 GB/s probe):

| `--pcie-frac` | CPU experts/layer | pool ms/round | wait for rings | decode |
| --- | ---: | ---: | ---: | ---: |
| auto (0.30) | 1.35 | 7.70 | 23.1 | **30.30 tok/s** |
| 0.55 | 0.88 | 7.33 | 25.2 | 28.33 tok/s |
| 0.80 | 0.73 | 7.25 | 28.3 | 26.38 tok/s |

Moving misses to the 14.2 GB/s link shrinks the CPU side but makes the GPU wait longer for staged
experts — net negative.  Keep the probe-driven default.

## Round 3: kernel profile, the per-layer sync, huge pages, and a determinism trap

### rocprofv3 kernel profile (2 decode rounds, 48 layers each)

```
__amd_rocclr_streamOpsWait   x96   33.61 ms   (0.35 ms each - one per layer per window)
wait_flag_ge_kernel          x288  21.03 ms   (0.073 ms each - three per layer)
fetch_blobs_kernel           x96   10.31 ms
__amd_rocclr_copyBuffer      x486   4.34 ms
__amd_rocclr_streamOpsWrite  x288   1.89 ms   (three per layer: flags A, B, flag_)
```

**Half of the decode's GPU kernel time is waiting**, and the single per-layer `streamOpsWait` is the
`hipStreamSynchronize(dma)` inside `hip_raise` (verify.cpp:75): under the engine's saturated CP queue the
write packet takes ~0.35 ms to be *serviced*, where a standalone probe with an idle CP sees only
30 us.  Removing that sync (change 2 below) does **not** hand back 48 x 0.35 = 16.8 ms/round, because the
sync was hiding behind GPU work: measured, "pool" fell 7.70 -> 4.81 ms/round while "wait for rings" rose
23.1 -> 25.7 ms/round, for a net +1-5 % (30.30 -> 30.49/30.71 tok/s).  Lesson repeated: kernel-duration
attribution is GPU-timeline time, not critical-path time.

### `madvise(MADV_HUGEPAGE)` on the arena: measured INERT here

The call succeeds, but nothing materialises:

* with `Rss = 47.9 GiB` (the full arena + weights), `/proc/<pid>/smaps_rollup` reports
  **`AnonHugePages: 0`**;
* every `thp_*` counter in `/proc/vmstat` (`thp_fault_alloc`, `thp_collapse_alloc`,
  `thp_fault_fallback`, `thp_collapse_alloc_failed`) is **unchanged** across whole runs.

Why: the arena is `hipHostRegister`ed, so its pages are pinned (`VM_LOCKED`) and are already faulted in
by the time the advice is given; khugepaged skips locked VMAs.  The call is kept (it is the right thing
to ask for, and would take effect where the arena is huge-mapped before it is locked), but its log now
says "advised", not "ok", and the note records that pinned pages are not collapsed.  Consequence: the
pool's ~30 GB/s against this box's measured ~45 GB/s line-read ceiling is **not** a page-size problem.

### Determinism trap: acceptance tests must compare matching adaptive states

The engine's output is **not bit-reproducible at a fixed command line**: the adaptive VRAM tier makes
timing-dependent swap decisions, different residency sends an expert down a different (CPU vs GPU)
arithmetic path, and the tokens diverge.  Evidence from the `adaptive tier N experts swapped` line:

| run | config | swapped | tokens match |
| --- | --- | ---: | --- |
| pre-change A | cache 8192 | 3156 | reference |
| pre-change B | cache 8192 (different ring env only) | 3156 | **yes** |
| post-change run 2 | cache 8192 | 3156 | **yes** |
| post-change run 1 | cache 8192 | 3182 | no |
| pre-change C/D | cache 4096 | 5270 / 5176 | no (they differ from each other) |

So "identical token sequence" is only a valid acceptance test between runs whose swap count agrees (and
only then does it prove the change is numerically neutral, which it is here).  The short 8-token seed run
remains bit-reproducible and is the cheap regression check.

## Round 4: real-text throughput — the tg gap is DRAFTING, not kernels

`tools/strata_tokenizer.py` cannot run here (it imports the third-party `regex`, and there is no pip),
and neither of my constructed prompts measures realistic acceptance (80x repeat -> 100 %, pseudo-random
ids -> 2 %).  So `bench/tools/tok_ascii.py` (mine, stdlib only) tokenizes ASCII text with the pack's own
`vocab.json`/`merges.txt` and the pack's shipped Qwen3.5 pre-tokenizer pattern, with
`decode(encode(text)) == text` as a self-check (3.36 chars/token, round-trip exact, coherent tokens).

**1,024 tokens of real C++ (`src/core/expert_source.cpp`), 256 generated, cache 8192:**

| | tokens/round | ms/round | decode |
| --- | ---: | ---: | ---: |
| **HIP (this box, real code text)** | **1.28** | **38.6** | **33.15 tok/s** |
| CUDA counterpart (`--spec 4 --spec-min-p 0.5`, MTP weights, real code-agent prompts) | 2.77 | ~54.9 | 50.5 tok/s |
| HIP, constructed repeat prompt | 3.95 | 98.4 | 40.0 tok/s |
| HIP, pseudo-random ids | 1.02 | 39.1 | 26.0 tok/s |

The CUDA row's round time is *inferred* from its published tokens/round.  On that basis **HIP is already
1.42x faster per round (38.6 vs 54.9 ms) and is behind only on tokens/round (1.28 vs 2.77)** - and the
CUDA figure is with MTP, which this port has never had (the ~5 GB MTP weights were not fetched).  So the
remaining tg gap is a drafting/weights gap, not a kernel gap.

Knob sweeps on the same real-text prompt confirm the defaults are already the optimum:

| knob | tokens/round | decode |
| --- | ---: | ---: |
| `--spec 2` (window 4) | 1.28 | **33.15 tok/s** |
| `--spec 4` (window 6) | 1.45 | 26.98 tok/s |
| `--spec 6` (window 8) | 1.47 | 18.88 tok/s |
| `--suffix-draft 3` (default) | 1.28 | **33.15 tok/s** |
| `--suffix-draft 4` | 1.24 | 31.58 tok/s |
| `--suffix-draft 24` | 1.00 | 27.10 tok/s |

Deeper windows buy a little acceptance (drafts accepted 56/202 at spec 2 vs 80/524 at spec 4 - the hit
rate *falls*) and cost more verification work per round, so they lose.  The prompt-lookup suffix drafter
is the accurate one (56 of 80 accepted at the default) but only fires on ~25 % of windows, and requiring
a longer match only reduces its coverage.

**Recommendation:** the only large tg lever left is the MTP draft weights (`--mtp`, a ~5 GB fetch);
everything reachable without them is measured at or near its optimum.

## Round 5: pp scaling — 1K is at CUDA parity, the 4K gap is the PCIe link

4,096 tokens of real C++ (2 chunks of 2048), 256-token cache tier:

| tier | HIP pp | CUDA counterpart pp | ratio |
| --- | ---: | ---: | --- |
| 1K | 415.7 tok/s (best) / 313-390 (spread) | 419 | ~1.0x |
| 4K | **494.8 tok/s** (8,275 ms) | 893 | 1.80x behind |

HIP improves only 1.18x from 1K to 4K where the CUDA counterpart improves 2.13x, and the reason is
visible in the engine's own line: `experts streamed 26864 (26864 by DMA)` for 4,095 tokens, i.e.
**~6.6 expert blobs per token, the same rate as at 1K (6.8)** - the cache does not capture a larger
share as the prompt grows, because the prompt's working set grows too (48 layers x up to 512 experts =
24,576 distinct against 9,854 resident slots).  At ~1.7 MB per blob that is ~45.7 GB over a link
measured at 14.2 GB/s = **~3.2 s of the 8.3 s prefill, 39 %**.

This box's slot is PCIe 4.0 **x8** (the user's x16 is split with another GPU; measured 14.2 GB/s, and
2 MB-blob transfers saturate it at 13.5 GB/s), while the CUDA counterpart sits on PCIe 5.0 x16.  So the
pp gap at 4K is a hardware bandwidth gap over *expert-streaming bytes*, and the remaining software lever
there is fewer streamed bytes (residency/profile), not kernel work: the non-streaming part of the
prefill is comparable between the two boxes.

### Correction: the expert blob is 1.95 MiB, not ~1.7 MB - the streaming share is larger

The header of the pack's `native_experts.txt` states the exact total: **50,292,326,400 B over 24,576
experts (48 layers x 512)** = **2,046,400 B = 1.95 MiB per expert blob** (per-layer values in the file
run 1,510,400 - 2,176,000 B).  My earlier "~1.7 MB" estimate was low, so the streaming shares above are
understated.  Recomputed at the measured 14.2 GB/s link:

| run | blobs streamed | bytes | DMA time | prefill | share |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1K, best (2,429 ms) | 6,947 | 14.2 GB | 1,001 ms | 2,429 ms | **41 %** |
| 1K, typical (2,968 ms) | 6,947 | 14.2 GB | 1,001 ms | 2,968 ms | 34 % |
| 4K (8,275 ms) | 26,864 | 55.0 GB | 3,871 ms | 8,275 ms | **47 %** |

Nearly half of the 4K prefill is expert bytes crossing a PCIe 4.0 x8 link.  That is the pp gap, and it
also bounds what any kernel work can recover on this box.

## Round 6: 4K prefill attribution, and a falsified ring hypothesis

Full phase table at 4,095 tokens / 2 chunks (`STRATA_PREFILL_TIMING=1 STRATA_PREFILL_RING=384`):

| phase | ms | % | | phase | ms | % |
| --- | ---: | ---: | --- | ---: | ---: |
| **gdn** | 1,990 | **23.9** | | **wait copy** | 1,006 | 12.1 |
| **qsa attn** | 1,104 | **13.2** | | **hc read** | 990 | 11.9 |
| dequant | 930 | 11.2 | | qsa proj | 467 | 5.6 |
| gemm gate/up | 820 | 9.8 | | gemm down | 386 | 4.6 |
| router+shared | 277 | 3.3 | | embed+steps | 214 | 2.6 |
| combine/host grouping/ple/gather/select | 156 | 1.9 | | | |

Two corrections to earlier reasoning:

* **The scaling culprit is the prompt attention, not staging.** `qsa attn` grows 112 ms -> 1,104 ms for 4x
  the tokens (superlinear), so on gfx1100 the emulated `mma16816` prompt attention is ~3 % of the prefill
  at 1K but **13 % at 4K** and would keep growing.  At 1K the phase table told me it was not the lever;
  that was true only for short prompts.
* **The staging ring is NOT the 4K problem, and the overlap is better there.**  I predicted that
  `RING_MAX = 512`, being exactly a layer's distinct expert count, would force poor overlap at 4K and move
  the ring optimum.  Measured, it does not:

  | ring | 4K pp | streamed | exposed wait copy |
  | --- | ---: | ---: | ---: |
  | 96 | 442.0 tok/s | 26,171 | 1,659 ms (18.1 %) |
  | **384** | **480.8 tok/s** | 26,864 | **1,006 ms (12.1 %)** |
  | 512 | 440.3 tok/s | 27,178 | 1,067 ms (11.7 %) |

  384 is optimal at 4K as well, and the *exposed* copy wait is proportionally **smaller** at 4K (12.1 %)
  than at 1K (23.6 %).  So the streaming is overlapped better on longer prompts, and the pp gap at 4K is
  GDN (24 %) + the expert compute (26 %) + attention (19 %) + the streamed bytes themselves (47 % of the
  prefill's bytes must cross the link), not a staging-lead problem.

## Round 7: the GDN recurrence A/B, and a benchmarking confound to respect

`src/prefill/kernels.cu` already carries an A/B switch for the GDN recurrence: unset (default) uses the
column-parallel `gdn_rec_cols_kernel` + `gdn_out_norm_kernel`, while `STRATA_GDN_REC_HEADS=1` uses the
one-block-per-head serial `gdn_rec_kernel`.  GDN is the largest single 4K phase (21-24 %), so it is worth
knowing which is faster on gfx1100:

| run | 4K prefill | gdn phase |
| --- | ---: | ---: |
| default (cols) | 10,159 ms (403 tok/s) | 2,107 ms |
| `STRATA_GDN_REC_HEADS=1` (serial heads) | 8,386 ms (488 tok/s) | 1,880 ms |

**This A/B is inconclusive, and the reason matters:** the GDN phase moved by only 227 ms while the totals
differed by 1,772 ms, so the two runs differ in *other* phases as well - and the "slow" arm is the one
that ran while the 5 GB MTP download was writing to the same disk/CPU.  Four same-config 4K runs cluster
at **8,275 / 8,386 / 8,517 ms** with that single 10,159 ms outlier.  So:

* do not benchmark while a download, build or other CPU/disk-heavy job is running - it produced a 20 %
  error here, larger than the effect being measured;
* treat the serial-heads result as unproven rather than as a win, and keep the default.

A related hygiene note: the CPU governor is `powersave` (the CPU was observed at 5.37 GHz, so it does
boost), and the prefill's host-side work is CPU-heavy enough that frequency state is a plausible
contributor to the residual few-percent spread.
