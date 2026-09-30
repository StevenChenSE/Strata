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

## Round 8: the dense projections are launch-bound; the GDN recurrence is not the cost

`bench/tools/prof_prefill.py` (new) aggregates a rocprofv3 kernel trace into prefill vs decode, using the
verify window's first `wait_flag_ge_kernel` as the separator.  On the 12-token prefill trace:

| group | ms | % | launches |
| --- | ---: | ---: | ---: |
| **rocBLAS GEMM** | 144.3 | **42.3** | **1,420** |
| dequant (weight -> f16) | 81.4 | 23.9 | 300 |
| runtime copy/fill | 74.4 | 21.8 | 10,707 |
| MMQ (quantized GEMM) | 35.7 | 10.5 | 3,458 |
| **gdn recurrence** | **0.65** | **0.2** | 36 |

Two conclusions:

* **1,420 rocBLAS launches at ~0.1 ms each** - for a 12-token prompt the dense projections are almost
  purely launch latency, which is why the phase table showed `qsa proj` barely growing from T=12 to T=1023.
  Graph-capturing the prompt path (the machinery already exists for the verify window) or batching the
  projections is the lever for short prompts; at 4K the same launches are ~6 % of the prefill, so this is
  a 1K-shaped win, not the 4K one.
* **The GDN recurrence is 0.65 ms** even in a phase table that charges 24 % to "gdn" - so that phase is its
  five *projections* (`Gemm::native` = `dequant_f16` + rocBLAS, prefill.cpp:683 -> gemm.cu:96), not the
  recurrence.  `use_mmq` (prefill.cpp:1320) covers only the expert GEMMs; the 12 dense `native_proj` sites
  always dequantize the weight into scratch first, per chunk.

## Round 9: the shallow staging path costs 20 % of the prefill below T = 2048 (confirmed)

`stream_all` (prefill.cpp:891) is `m.ring > STAGE && T >= STREAM_ALL_MIN && m.src != nullptr` with
`STREAM_ALL_MIN = 2048`, so any prompt whose chunk is smaller takes the shallow `lookahead = STAGE - 1 = 7`
staging walk instead of the deep ring.  The engine's `T` is `min(--prefill, n_tokens - 1)`, so the boundary
is testable with two adjacent prompt lengths - same work, one predicate flip:

| engine T | path | prefill | pp | exposed `wait copy` |
| ---: | --- | ---: | ---: | ---: |
| 2046 | shallow | 4,764.7 ms | 429.4 tok/s | 930 ms (20.3 %) |
| 2047 | shallow | 4,632.4 ms | 441.9 tok/s | 930 ms (20.8 %) |
| **2048** | **deep ring** | **3,798.8 ms** | **539.1 tok/s** | **142 ms (3.9 %)** |

**+22 % pp and the exposed copy wait falls 20.8 % -> 3.9 %** for one token of extra prompt.  The clean 1K
runs sit at 577-588 ms of exposed copy wait (22.9-24.2 % of a 2,476-2,572 ms prefill), so 1K prompts pay
this today: lowering the threshold (or making it overridable) should recover roughly a fifth of the 1K
prefill.

Two caveats recorded honestly:

* an earlier single run showed `gather` at 632 ms (26 %) for a 1K prompt and I read it as a staging stall;
  five clean runs show `gather` at 10-16 ms, so that was an outlier run (it was also 40 % slower overall),
  not a phase.  The exposed `wait copy` figure is the reproducible one.
* the first attempt at this experiment used a 2047-token prompt, which gives engine T = 2046 - below the
  threshold, so both arms took the shallow path.  The pairing has to be 2048/2049 *tokens* for T = 2047/2048.

## Round 10: decode kernel anatomy

Same trace, decode window (2 rounds = 96 layer-visits, 110.5 ms of kernel time):

| kernel | ms | per layer | note |
| --- | ---: | ---: | --- |
| `__amd_rocclr_streamOpsWait` | 33.61 | 1.0 | the `hipStreamSynchronize` in `hip_raise` - removed since |
| `wait_flag_ge_kernel` | 21.03 | 3.0 | 0.073 ms each: three *genuine* dependency waits per layer |
| `fetch_blobs_kernel` | 10.31 | 1.0 | 0.107 ms, grid 98,304: gathers the staged PCIe blobs |
| `native_down_kernel<20>` | 4.12 | 1.6 | expert down projection |
| `gr_down_multi_kernel` | 3.25 | 2.0 | gated-residual down |
| `native_iq4_xs_mmvq_kernel<true>` | 3.06 | 0.4 | dense projection (IQ4_XS) |
| `gr_up_multi_kernel` | 2.85 | 2.0 | gated-residual up |
| `native_q6_k_mmvq_kernel<false>` | 2.74 | 1.3 | dense projection (Q6_K) |
| `__amd_rocclr_copyBuffer` | 4.34 | 5.1 | runtime memcpys |

So **half of the decode's GPU kernel time was the stream-ops wait plus the polling waits**, and the actual
compute is small and fragmented: ~8 kernels per layer across dense projections, expert projections and the
gated-residual pair, none above 0.2 ms.  Removing the sync did not return its 16.8 ms/round because a
separate stream's shader duration overlaps the compute stream (the measured effect was 1-5 %), which is the
same overlap lesson as before.  The three `wait_flag_ge_kernel` waits (10.5 ms/round = ~32 % of a 33 ms
round) are *genuine* producer latency - the host's pool and the expert DMAs - so they are not removable by
tuning the kernel; the backoff is only 100 ns (`__nanosleep(100)`), not the poll interval I first suspected.

## Round 11: MTP lands (the tg gap is essentially closed), and a withdrawn claim

### MTP: 33.15 -> 40.1-47.4 tok/s on the same real-text prompt

The draft head was fetched (`tools/mtp_fetch.py`), packed (`mtp_pack.py --experts q2_0`, 0.889 GB) and
turned into the runtime dir (`mtp_rt.py` + `data/draft_vocab.bin`), then run with the documented
`--mtp <rt>`:

| run | rounds | tokens/round | draft accept | tg | mtp cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| no MTP (suffix/lookahead drafter) | 200 | 1.28 | 0.277 | 33.15 tok/s | - |
| MTP, `--spec 4 --spec-min-p 0.5`, `--prefill 512` | 87 | **2.98** | **0.815** | **47.37 tok/s** | 3.03 ms/round |
| MTP, same, `--prefill 2048` | 79 | 3.24 | 0.859 | 40.08 tok/s | 3.43 ms/round |
| **CUDA counterpart (MTP, real code prompts)** | - | 2.77 | - | **50.5 tok/s** | - |

So gfx1100 now reaches **79-94 % of the CUDA counterpart's decode** against 66 % before, with *better*
acceptance than the CUDA bench recorded (2.98-3.24 vs 2.77 tokens/round); the spread between the two runs
is the round latency (62 vs 81 ms/round), i.e. which experts the prompt's prefill left resident.

Configuration note: `--mtp` takes 799 MiB of VRAM (675 experts + 111 dense), so with a large expert cache
the 2048-token prefill buffers no longer fit - `strata generate: prefill: device buffers for a chunk of
2048 tokens do not fit` and exit 1.  `--prefill 512` (as `docs/ORCA.md`'s config uses) works.

### Withdrawn: "the shallow staging walk costs 20 % below T = 2048"

The boundary experiment looked convincing (T=2047: 441.9 tok/s with 930 ms of exposed copy wait; T=2048:
539.1 tok/s with 142 ms), and the delegated `STRATA_PREFILL_STREAM_ALL_MIN` override was built to test it.
Three **interleaved** repetitions with the env override on an otherwise idle machine:

| rep | default (shallow) | forced deep |
| --- | ---: | ---: |
| 1 | 340.93 tok/s (wait 568 ms) | 352.25 (585 ms) |
| 2 | 333.66 (601) | 326.92 (583) |
| 3 | 339.78 (553) | 340.09 (627) |
| median | **339.78** | **340.09** |

No effect.  The earlier T=2046/2047 arms (429/442 tok/s, 930 ms) were taken while the 5 GB MTP download was
running, and re-running T=2047 alone on an idle machine gave 545.5 tok/s with 113 ms - the same as the
"deep" arm I had credited.  **The threshold claim is withdrawn**; the env override stays as a harmless
diagnostic (default unchanged).

### What actually moves 1K pp: the expert path

Comparing the phase tables of a 397.7 tok/s run and a 343.7 tok/s run, the difference is entirely in the
expert phases:

| phase | fast run | slow run | delta |
| --- | ---: | ---: | ---: |
| dequant | 249 ms | 490 ms | +241 |
| gemm gate/up | 237 | 348 | +111 |
| gemm down | 110 | 155 | +45 |
| gdn | 558 | 455 | -103 |
| wait copy | 577 | 619 | +42 |

So 1K pp should be quoted as a **range (340-420 tok/s)** with the phase table as the explanatory artifact,
and any A/B below ~20 % needs interleaved repetitions on an idle machine - the single biggest methodology
lesson of this session.

## Round 12: the MTP configuration that keeps pp - and 98 % of the CUDA counterpart's tg

`--mtp` needs 799 MiB, which stopped a 2048-token prefill chunk from fitting at `--expert-cache 8192`.  The
trade-off curve (same real-text prompt, 256 generated, `--spec 4 --spec-min-p 0.5`, `--prefill 2048`):

| `--expert-cache` | slots | VRAM | decode | prefill | tokens/round |
| --- | ---: | ---: | ---: | ---: | ---: |
| **auto** | **9,492** | 18.01 GiB | **49.60 tok/s** | **328.5 tok/s** | **3.34** |
| 6144 | 8,034 | 15.23 GiB | 42.69 tok/s | 286.1 tok/s | 3.24 |
| 4096 | 5,341 | 10.16 GiB | 41.83 tok/s | 275.0 tok/s | 3.28 |

`auto` is the answer: the engine sizes the cache to the VRAM that is actually free once the draft layer is
resident, so the 2048-token chunk fits *and* the expert pool stays small.  Result:

* **tg 49.60 tok/s against the CUDA counterpart's 50.5 = 98 % of it** (from 66 % before MTP), with better
  acceptance (3.34 vs 2.77 tokens/round);
* **pp 328.5 tok/s**, back inside the no-MTP 1K range (340-420), instead of the 240 tok/s that
  `--prefill 512` cost.

Recommended configuration: `--expert-cache auto --prefill 2048 --spec 4 --spec-min-p 0.5 --mtp <rt>`.

## Round 13: why pp is behind - rocBLAS on gfx1100 runs at ~14 TFLOPS (no WMMA path)

1K prefill kernel attribution (my `bench/tools/prof_prefill.py` on a rocprofv3 trace):

| group | ms | % | launches |
| --- | ---: | ---: | ---: |
| **rocBLAS GEMM** (the dense projections) | 731.4 | **42.9** | 998 |
| **MMQ** (the expert GEMMs) | 526.8 | 30.9 | 24,191 |
| dequant (weight -> f16) | 153.3 | 9.0 | 300 |
| attention | 106.7 | 6.3 | 24 |
| runtime copy/fill | 72.5 | 4.2 | 12,637 |
| gdn recurrence | 62.8 | 3.7 | 36 |

`Gemm::native` (gemm.cu:96) dequantizes each dense weight to FP16 in scratch and then calls rocBLAS
(`Cijk_..._HSS_...`).  `bench/tools/rocblas_f16_probe.cu` measures what rocBLAS itself can do on this
GPU, with the engine's own shape convention:

| shape | ms | TFLOPS |
| --- | ---: | ---: |
| 4096^3 | 10.19 | 13.5 |
| q_proj-like M=12288 N=1023 K=2560 | 4.50 | 14.3 |
| o_proj-like M=1023 N=2560 K=12288 | 4.58 | 13.9 |
| ssm/gate-like M=5120 N=1023 K=2560 | 1.86 | 14.2 |

**~14 TFLOPS regardless of shape.**  The 7900 XTX's FP16 WMMA peak is ~123 TFLOPS, so rocBLAS is on a
SIMT/VectorALU path - RDNA3 has no MFMA, and the Tensile kernels selected for gfx1100 are not WMMA ones.
`ROCBLAS_USE_HIPBLASLT=1` does not change it (13.2 TFLOPS), so this is not an env-var fix either.  The CUDA
counterpart's cuBLAS would run these on tensor cores at several times that rate, which is a large part of
why its 1K prompt is 419 tok/s against this port's 328-420 and its 4K prompt is 893 against 495.

Consequences for the objective: the port's remaining pp gap is **not a port bug**.  It is (a) the PCIe 4.0
x8 link, which carries 47 % of the 4K prefill's bytes, (b) rocBLAS's missing WMMA path for the dense
projections (43 % of the 1K prefill), and (c) the emulated `mma16816` prompt attention (13 % at 4K).
Only (c) and a hand-written WMMA GEMM would be addressable in software, and both are projects rather than
patches.  ROCm issue worth noting for the record: `rocblas` fp16 GEMM on gfx1100, any shape, ~14 TFLOPS.

## Round 14: the window is capped at 6 tokens by a shared-memory limit (latent bug), and --spec-min-p is fine

Sweeping the draft depth and the acceptance threshold on the recommended MTP configuration:

| config | tokens/round | draft accept | decode |
| --- | ---: | ---: | ---: |
| `--spec 4 --spec-min-p 0.5` (default) | 3.20 | 0.848 | 47.74 tok/s |
| `--spec 4 --spec-min-p 0.3` | 3.37 | 0.804 | 46.50 tok/s |
| `--spec 6` | **segfault (exit 139)** | - | - |
| `--spec 8` | **segfault (exit 139)** | - | - |

`--spec-min-p 0.3` buys more accepted tokens per round (3.37 vs 3.20) but costs more verification, so the
default 0.5 stays.  The segfaults are a real defect, not a tuning result:

`src/kernels/cuda/fused_gr.cu:299` (`fused_gr_read_multi`) launches `gr_down_multi_kernel` with
`n_tok * TILE * sizeof(float)` of dynamic shared memory, `TILE = 2560` -> **10,240 B per token**, while the
opt-in it requests is `min(kFusedGrMaxT * TILE * 4, cudaDevAttrMaxSharedMemoryPerBlockOptin)`.  On gfx1100
that attribute is 65,536 B, so:

| window | smem needed | launch |
| ---: | ---: | --- |
| 6 (i.e. `--spec 4`) | 61,440 B | ok |
| 7 (`--spec 5`) | 71,680 B | `hipErrorInvalidArgument` |
| 8 (`--spec 6`) | 81,920 B | fails, then the process dumps core during teardown |

The verify window is `--spec + 2` tokens, so **every `--spec >= 5` crashes on any 64 KB-shared-memory
device** - including the CUDA product on Turing, which the code's own comment acknowledges ("Turing: 64 KB -
enough for windows of up to 6 tokens").  This also blocks measuring whether deeper draft windows help
decode.  Delegated fix: fall back to the existing per-token `fused_gr_read` for the tokens that do not fit,
log it once, and keep the <= 6-token fast path untouched.

### What the window fix unlocks (and that nothing else blocks it)

`src/program/generate.cpp:1200-1202`: with the default `--suffix-draft > 0` and no explicit `--mtp-max-t`,
the engine sets `mtp_max_t = spec` and then `o.spec = min(spec + 2, 8)` - kVerifyMaxT is 8, so:

| flag | verify window | MTP drafts | today |
| --- | ---: | ---: | --- |
| `--spec 4` | 6 | 4 | works |
| `--spec 6` | 8 | 6 | blocked by the fused_gr shared-memory bug |
| `--spec 8` | 8 | 8 | blocked by the same bug |

An audit of every dynamic-shared-memory opt-in in `src/kernels/cuda/` shows **`fused_gr.cu:336` is the only
launch whose shared memory scales with the token count** (the other three opt-ins - `qsa.cu:674`,
`qsa_prompt_attn.cu:630,661` - are chunk-scaled, which the prompt path already sub-batches).  So the fused_gr
fallback is sufficient to make windows 7-8 work, and no second limit should bite.

## Round 15: the window fix lands - `--spec 6/8` no longer crash, and deeper windows are a wash

With the fused_gr fallback in place (`fit` tokens through the fused path, the rest per-token, warning logged
once), the first-ever window-8 measurements (MTP, `--expert-cache auto`, `--prefill 2048`, real-text prompt):

| `--spec` | verify window | rounds | tokens/round | draft accept | ms/round | decode |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **4** | **6** | 76 | 3.38 | 0.854 | **68.9** | **48.88 tok/s** |
| 6 | 8 | 62 | **4.15** | 0.783 | 96.0 | 43.00 tok/s |
| 8 | 8 | 66 | 3.95 | 0.720 | 102.4 | 37.89 tok/s |

A window of 8 accepts **23 % more tokens per round** (4.15 vs 3.38) and still loses, because a round costs
39 % more.  Two reasons, and they matter for anyone tempted to "fix" this further:

* the extra window work itself (8 tokens verified instead of 6), and
* the fallback's cost: 2 of 8 tokens per layer go through the per-token `fused_gr_read`, i.e. 96 extra
  kernel pairs per round, worth roughly 10 ms of the 27 ms increase.

So even a chunked-smem GR kernel (which would remove the fallback cost) would leave window 8 at best at
parity (43.00 -> ~48 tok/s at 96 - 10 = 86 ms/round = 20.7 ms/token vs 20.4 for window 6).  **`--spec 4`
stays optimal**, and the value of the fix is that `--spec >= 5` no longer segfaults: the window is now
bounded by the engine's own `kVerifyMaxT = 8` with a diagnostic instead of a core dump.

## Round 16: `--adapt-every 2` is a replicated ~3 % tg win

The adaptive VRAM tier swaps experts every 4 rounds by default.  Halving that was tested with pairs
interleaved in time, and after the first (order-confounded) batch the order was reversed to balance the
design.  All five pairs favour the shorter cadence:

| pair | order | base | `--adapt-every 2` | delta |
| --- | --- | ---: | ---: | ---: |
| 1 | base first | 48.48 | 48.40 | -0.08 |
| 2 | base first | 47.45 | 48.76 | +1.31 |
| 3 | base first | 46.59 | 49.53 | +2.94 |
| 4 | **adapt2 first** | 47.77 | 49.36 | +1.59 |
| 5 | **adapt2 first** | 48.31 | 50.15 | +1.84 |
| **mean** | | **47.72** | **49.24** | **+1.52 (+3.2 %)** |

The base arm's apparent downward drift (48.48 -> 46.59) is why the first batch alone was not trustworthy;
reversing the order in pairs 4-5 removes the sequencing explanation and the effect survives.

Recommended configuration is therefore
`--expert-cache auto --prefill 2048 --spec 4 --spec-min-p 0.5 --adapt-every 2 --mtp <rt>`, at
**49.2-50.2 tok/s decode** against the CUDA counterpart's 50.5 (97-99 % of it), with pp unchanged
(334 tok/s at 1K).

## Round 17: the full tier matrix against the CUDA counterpart

Same model (IQ3_S), same recommended HIP configuration
(`--expert-cache auto --prefill 2048 --spec 4 --spec-min-p 0.5 --adapt-every 2 --mtp <rt>`, `--kv int8`
above 4K), real C++ prompts via `bench/tools/tok_ascii.py`:

| tier | HIP pp | CUDA pp | ratio | HIP tg | CUDA tg | ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 340-420 | 419 | 0.8-1.0 | **49.2-50.2** | 50.5 | **0.97-0.99** |
| 4K | 495 | 893 | 0.55 | - | 44.6 | - |
| 32K | 521-572 | 1,499 | 0.35-0.38 | **40.5** | 49.0 | **0.83** |

**tg is 83-99 % of the counterpart across the range; pp falls off with prompt length, and the cause is
measured rather than assumed.**  At 32K the engine streamed **258,880 expert blobs = 516 GB** in 16 chunks
(the same experts re-streamed once per chunk, since 9,260 resident slots cannot hold a 24,576-expert
working set), which is **36.3 s of the 62.9 s prefill = 58 %** on the 14.2 GB/s link.  The CUDA counterpart
spends the *same share* of its time (32767/1499 = 21.9 s total, 516 GB at ~40 GB/s = 12.9 s = 59 %) - it is
2.9x faster at 32K because its link is ~2.8x faster.  So the long-prompt pp gap is PCIe bandwidth over
*re-streamed expert bytes*, not a port defect; the software lever would be residency/streaming strategy
(prompt-aware caching), and the hardware lever is the x8 slot.

Also worth recording at 32K: MTP drafting costs 4.61 ms/round (925 MiB of VRAM), admission acceptance stays
high (0.860, 3.79 tokens/round), and `--adapt-every 2` costs only 0.154 ms/round of host swapping.

## Reference: the sibling project already solved this on the same GPU

`../vllm-serving` (same box, gfx1100, same model family) has hand-written RDNA3 kernels worth copying
from, and its notes reach the same conclusions this benchmark did:

* `vllm/csrc/rocm/q_gemm_rdna3_wmma.cu` (2,106 lines) is a fused **W4A16 GPTQ WMMA prefill GEMM** with tile
  variants `gemm_q4_wmma_kernel_{16x16_1w, 32x16_2w, 64x16_4w, 64x32_4w}`: the weight tile is dequantized
  into `__shared__ T b_lds[...]`, accumulated with
  `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` (16x16x16 per instruction on wave32, FP32 accumulate), with a
  K-split epilogue for large K.  Small M (decode) deliberately stays in the sibling scalar/dot kernel
  `q_gemm_rdna3.cu`.  **WMMA is therefore viable for this GPU's prefill shapes** - which is exactly what
  rocBLAS is not doing for the engine's dense projections (measured 14 TFLOPS, see Round 13).
* Documented traps to respect: wave32 WMMA input fragments are "doubled" (lanes 16..31 mirror 0..15) and the
  C fragment mapping differs (lane t holds output column n = lane_lo, 8 elements alternating M rows by
  lane_hi); and putting WMMA in the same translation unit as the scalar kernel **silently miscompiled the
  M=1 path** - they isolate the TUs on purpose.
* Their own bottleneck note matches this benchmark's: "`hipMemcpyAsync` 13.8 < Triton gather 14.4 GB/s ...
  only two paths remain: reduce bytes (hot-cache hit rate; VRAM is full) or overlap the copies with compute
  (the copy engine is idle during the GEMM)".  That is the same 58%-of-the-32K-prefill finding here, and the
  same two levers: fewer re-streamed bytes, or a busier copy engine during compute.

Delegated next: replace the HIP side of `Gemm::f16` (gemm.cu, currently `cublasGemmEx`/rocBLAS at 14 TFLOPS)
with a WMMA GEMM modelled on that reference, keeping the existing `dequant_f16` staging and verifying both
numerics against rocBLAS and throughput at the engine's own shapes.

## Round 18: the long-prompt lever is OVERLAP, not bytes (correcting my own framing)

I had described the 32K prefill as "10.5x re-streaming redundancy", implying waste to recover.  It is not
waste - it is forced by the VRAM cache size:

* the cache held 9,260 slots = **193 of the 512 experts per layer**, so *every* chunk must stream at least
  512 - 193 = 319 experts per layer;
* over 16 chunks that floor is 245,056 blobs, and the measured 258,880 is just **1.06x the floor**;
* the floor is 501 GB = **35.3 s at 14.2 GB/s**, inside a 62.9 s prefill.

So the link's duty cycle is at most **58 %**, i.e. the link is **idle ~42 % of the prefill** while the GPU
computes, and the prefill is roughly 36 s of unavoidable DMA plus ~26 s of compute with poor overlap.
**Perfect overlap would give ~max(36, 26) = 36 s -> pp ~910 tok/s instead of 521-572 (+65 %)** at 32K.

That is the same conclusion the sibling project recorded for this GPU ("the copy engine is idle during the
GEMM; only two paths remain: reduce bytes or overlap the copies with compute"), and here the byte side is
already at 1.06x of its floor - so for long prompts the software lever is **pipelining the expert streaming
under the compute** (the routing for layer l+1 is only known after layer l's MoE output, so a useful
prefetch has to be predictive - e.g. stream what the expert profile says layer l+1 is likely to need while
layer l computes, and correct after the router runs).

### Correction (Round 19): the prefill is KERNEL-bound; only 13 % of it is idle - "overlap, not bytes" was wrong

Round 18 argued from arithmetic (35 s of floor DMA inside a 62.9 s prefill) that the long-prompt lever was
overlap, implying ~65 % headroom.  That arithmetic conflated "ideal DMA time" with "serial time".  Traced
directly at 4K with rocprofv3 (`--kernel-trace --memory-copy-trace`, window = first prompt-path GEMM to the
first decode marker, 9,118 ms):

| | busy | share |
| --- | ---: | ---: |
| kernels | 6,262 ms | **69 %** |
| copy engine | 4,133 ms | 45 % |
| **union (either busy)** | **7,906 ms** | **87 %** |
| true idle | 1,212 ms | **13 %** |

The 13 % idle matches the phase table's `wait copy` almost exactly (12.4 % at 4K), and kernels and copies
overlap by 27 points of the window.  So the prefill is **kernel-bound with the streaming already largely
hidden**: the recoverable overlap is ~13 %, not 65 %.

Two consequences:

* **`STRATA_KV_STAGE_OWN` and a deeper ring do not help** - verified that the knob takes effect (the borrow
  grew 1020 -> 1187 slots / 1.94 -> 2.26 GiB) and pp got ~6 % *worse*, so ring depth was never the binding
  constraint.  The ~42-58 % "link duty" I quoted earlier is simply the consumer not needing more bytes, not
  an idle copy engine waiting to be fed.
* **The lever is faster kernels** - 69 % of the prefill is kernel time, which is what the WMMA GEMM works on
  (dense projections are 27-43 % of it depending on tier) and what the emulated attention and the expert MMQ
  kernels are.  The reference project's "copy engine is idle during the GEMM" note is true of its own setup;
  measured here, the copy engine and kernels overlap for 27 % of the window and the union leaves only 13 %.

## Round 20: WMMA GEMM for the dense projections - correct and 2x faster in isolation, but NEUTRAL end-to-end

Implemented (delegated, `workflow-13`, modelled on the `../vllm-serving` reference): a HIP-only RDNA3 WMMA
FP16 GEMM in `src/prefill/wmma_gemm.{h,cu}`, dispatched from `Gemm::f16` for `T >= 16` and `beta in {0,1}`
with `hipblasGemmEx` as the fallback (`STRATA_WMMA_GEMM=0` forces the old path for A/B).  Two variants:
`gemm_wmma_16x16_1w` (single wave, no LDS) and `gemm_wmma_64x64_4w` (4 waves, double-buffered W in LDS).
Wave32 "doubled" input fragments and the C mapping follow the reference's documented layouts.

**Correctness (verified independently, not taken on trust):** the delegated probe reported 0.00e+00 error
for every shape, which is not credible for a different accumulation order - and it turned out to be a
NaN-blind comparison (`std::max` never updates on NaN, and a broken host `__ushort_as_half` made both sides
NaN).  `bench/tools/wmma_gemm_min_check.cu` re-tests with conversion-free data (all-ones fp16, so the answer
must be exactly K):

| shape | Y[0] | expected | wrong | NaN |
| --- | ---: | ---: | ---: | --- |
| 16x16x16 | 16.000 | 16 | 0 | 0/256 |
| 16x16x32 | 32.000 | 32 | 0 | 0/256 |
| 33x48x64 (ragged T,N) | 64.000 | 64 | 0 | 0/1584 |
| 64x64x96 (4w path) | 96.000 | 96 | 0 | 0/4096 |
| 7x16x32 (small T) | 32.000 | 32 | 0 | 0/112 |

and the engine's 8-token seed run reproduces the pre-change output exactly
(`271 7734 264 13280 9834 421 15339 279`), so the change is numerically faithful in practice.

**Isolated throughput** (`bench/tools/wmma_gemm_test.cu`, my rocBLAS column reproduces my own probe's
12.3-13.6 TFLOPS exactly):

| shape | WMMA | rocBLAS |
| --- | ---: | ---: |
| 12288x1023x2560 | 22.4 TFLOPS | 13.4 |
| 1023x2560x12288 | 22.1 | 12.3 |
| 2560x1023x12288 | 25.1 | 12.2 |
| 5120x1023x2560 | 15.4 | 13.1 |
| 4096^3 | 29.7 | 13.0 |

**End-to-end: no measurable change.**  Five interleaved 1K pairs: medians 3,062 ms (WMMA) vs 3,126 ms (off),
~2 % - inside the run-to-run spread, and the phase table shows no systematic drop in `qsa proj`/`gdn`.  Two
clean 4K pairs are identical to 0.02 % (7,980.7 vs 7,969.6 ms; 7,907.7 vs 7,906.2) - one earlier 10,902 ms
WMMA run was an outlier, not a regression.

**Lesson that corrects Round 13:** the "dense projections are 42.9 % of the 1K prefill" figure came from
aggregating rocprof kernel durations over ~1000 *small* launches, where per-launch profiling overhead
inflates the sum.  A 2x faster kernel for that component moves the wall clock by ~0, so the true share is
small and the dense GEMMs are not on the prefill's critical path.  Kept enabled because it is correct, never
slower, HIP-only with a fallback, and a foundation for a fused dequant+WMMA variant; the probe is preserved
at `bench/tools/`.

## Round 21: the WMMA GEMM was silently compiled out - after fixing that it is worth 7-13% pp

Round 20 concluded "correct and 2x faster in isolation, but NEUTRAL end-to-end".  That was a false negative:
a rocprofv3 kernel trace showed **zero `gemm_wmma` kernels** in a 1K prefill, and the object's symbol was
**3 bytes with no undefined references** - the `return false` stub.  A temporary probe print in the dispatch
showed `eligible=1 used=0` for every call: the arguments were fine, the *implementation* was missing.

Root cause, measured with the exact build flags (host vs device pass of `clang++ -x hip`):

| pass | `__gfx1100__` | `__HIP_DEVICE_COMPILE__` | `STRATA_WMMA_GFX11` |
| --- | --- | --- | --- |
| **host** (emits the symbol the engine links) | **absent** | defined | **undefined** |
| device | defined | defined | defined |

`wmma_gemm.cu`'s guard was `#if defined(__HIPCC__) && (defined(__gfx1100__) || ...)` and the outer guard was
`#if defined(STRATA_WMMA_GFX11) || !defined(__HIP_DEVICE_COMPILE__)`, so on this toolchain the host pass took
the `#else` stub.  The probes worked because **hipcc** (not the build's direct `clang++` invocation) defines
the arch macro in both passes - i.e. the probes were testing a different compilation of the same file.

Fix: define the arch macro from the build (`CMakeLists.txt`, `target_compile_definitions(strata_prefill_hip
PRIVATE STRATA_WMMA_GFX11=1)` when `CMAKE_HIP_ARCHITECTURES` matches `gfx11`), and reduce the file to a single
`#if defined(STRATA_WMMA_GFX11)` with the stub as the only fallback, so a non-gfx11 build returns false and
can never silently launch empty kernels.  Verified: the symbol is now 413 bytes, the binary contains the four
`gemm_wmma` symbols, and the A/B finally measures the real thing:

| tier | WMMA | `STRATA_WMMA_GEMM=0` | gain |
| --- | ---: | ---: | ---: |
| 1K, 3 interleaved pairs (medians) | **2,832.9 ms (361.1 tok/s)** | 3,271.8 ms (312.7 tok/s) | **-13.4 %** |
| 4K | **7,069.2 ms (579.3 tok/s)** | 7,621.1 ms (537.3 tok/s) | **-7.2 %** |

The WMMA arm is also far steadier (2,828.8 / 2,870.6 / 2,832.9 ms, +-0.7 %) than the rocBLAS arm
(3,029 / 3,272 / 4,312 ms).  The 8-token seed run still reproduces the pre-change tokens exactly, so the
change is numerically faithful in practice.

**Lesson (third time this session): a "no effect" result must be validated before it is believed.** Round 18
over-trusted arithmetic, Round 20 over-trusted a kernel-level speedup, and both times the missing step was
confirming that the code under test actually ran.  Here the check is cheap: trace for the kernel symbol, or
assert the linked function's size/references.

### Audit: is any OTHER arch- or compiler-macro guard silently stubbing code?

After the FP16 WMMA stub was found, every arch/compiler-macro guard in the port was audited:

| site | guard | verdict |
| --- | --- | --- |
| `src/prefill/wmma_gemm.cu` | was `defined(__HIPCC__) && defined(__gfx1100__)`, plus an `!defined(__HIP_DEVICE_COMPILE__)` fallback | **the trap** - fixed to the build-provided `STRATA_WMMA_GFX11` with a false-returning fallback |
| `include/strata/kernels/{rope,f16_bits,bf16_bits,mrope}.hpp` | `defined(__CUDACC__) || defined(__HIPCC__)` | safe: `__HIPCC__` **is** defined in both passes of this build (measured), unlike `__gfx1100__` |
| `compat/hip/cuda_runtime.h:165,189` | `defined(__HIP_DEVICE_COMPILE__)` | safe by intent: these define device-only shim intrinsics |
| dispatch sites (`Gemm::f16/bf16`, `use_mmq`, `STRATA_KV_STAGE_OWN`, `--adapt-every`, `--spec`) | env/CLI gates | all verified live by their observable side effects (borrow counts, log lines, phase changes) |

So `wmma_gemm.cu` was the only site where a compiler macro could disagree with the build and silently select a
stub.  The general rule this yields: **gate architecture-specific code on a macro the BUILD defines (or on
`__HIPCC__`, which this toolchain sets in both passes), never on `__gfx1100__`, and always keep a
false-returning fallback rather than empty kernels.**

### Scoping note: the "hc read" phase IS the bf16 GEMM target

The phase table's `hc read` is 10.5 % of the 1K prefill and 11.9 % at 4K - stable across tiers, which is why it
looked like host work.  It is not: prefill.cpp:1073-1079 marks it and then runs, per *half* and so twice per
layer:

    gr_norm(R, hc_*_norm, xn, xn16)          -> bf16_proj(hc_*_down,   xn16 -> lo)
    gr_silu(lo, lo16)                        -> bf16_proj(hc_*_up,     lo16 -> gated)
                                             -> bf16_proj(hc_*_inject, xn16 -> inj)
    gr_mix(xn, gated, mixed, mixed_bf)

i.e. **six bf16 GEMMs per layer** (three per half) plus three small elementwise kernels.  That is exactly the
`Gemm::bf16` path the BF16 WMMA variant targets, so the expected gain from that work is the GEMM share of this
phase - with the bf16 GEMMs measured at 204 ms of the 731 ms of rocBLAS time in a 1K prefill (28 %), a 2x
faster kernel should be worth roughly 4-6 % of prompt throughput.

### Scoping note 2: the "dequant" phase is a per-expert relayout, and it is launch-bound

`pt.mark(kPfDequant)` (prefill.cpp:1402) does **not** dequantize on the MMQ path - it calls
`mmq::gather_native` (moe_mmq.cu:151), which is just `copy16_kernel` (or `copy1_kernel`) moving the expert's
gate/up halves interleaved plus its down matrix into a 16-expert group slot.  Two measured facts:

* it is called **once per expert**: 26,864 times in the 4K prefill (one per streamed blob), and `MMQ_GROUP = 16`
  with the author's own comment "experts per MMQ launch (the gather is per expert, as blobs arrive)" - the
  per-expert granularity is deliberate, so the GPU can start an expert as soon as its blob lands;
* the phase costs 933 ms at 4K = **35 us per expert**, while the copy itself is ~2 MB and should take ~3 us at
  VRAM bandwidth.  So the phase is launch/host-bound, not bandwidth-bound (~26,864 small launches).

So the next candidate after the bf16 work is to remove the relayout rather than speed it up: the MMQ kernels
consume the group slot only because their operands are contiguous, and each expert's GGUF blocks are already
contiguous in its own blob - passing a per-expert source pointer/offset (the same indirection idea noted for the
decode path's `fetch_blobs_kernel`) would delete the gather and its extra VRAM traffic outright.  Alternatives
are batching the gather per `MMQ_GROUP` (which sacrifices the arrival-order pipelining the comment describes) or
capturing the walk in a graph (hard: the walk's length and order are data-dependent).

#### Correction to the note above: the phase's 933 ms is not all launch cost

I wrote that the `dequant` phase is "launch/host-bound".  That overstates it.  The phase timer measures the
*stream* timeline from the `kPfDequant` mark to the next mark, so it includes idle.  The gathers themselves are
~2 MB copies: 26,864 of them at ~3 us of VRAM traffic each is only ~80 ms of GPU work, so ~850 ms of the
933 ms is *idle inside that phase*.  The author's comment - "the gather is per expert, as blobs arrive" - says
the gathers are scheduled to the arrivals, so the most likely explanation for that idle is **waiting for the
PCIe stream to deliver the next blob**, not the host enqueueing 26,864 launches (which the host-side staging
figure, ~667 ms for 258,880 blobs at 32K, would not explain either way).

What survives: the relayout is avoidable work (its ~80 ms plus a second pass of ~55 GB through VRAM), and
removing it is still the right idea.  What does not survive: the suggestion that it is worth the whole 12 % of
the phase.  Expected gain from deleting the gather is on the order of its own cost, and the phase's dominant
idle needs a finer instrument (a per-launch or copy-arrival timeline) to attribute before anyone spends effort
on it.

## Round 22: GDN split into sub-phases - the recurrence is NOT a lever, but the bf16 GEMMs are the biggest and most variable phase

Added four sub-phase marks inside the GDN branch (`gdn gates`, `gdn conv`, `gdn rec`, `gdn out`) so the largest
table entry stops being a black box.  A 4K run:

| sub-phase | ms | share |
| --- | ---: | ---: |
| gdn (qkv/gate/alpha/beta projections) | 580 | 6.3 % |
| gdn out (ssm_out projection) | 202 | 2.2 % |
| **gdn rec** (the sequential scan) | **228** | **2.5 %** |
| gdn conv | 20 | 0.2 % |
| gdn gates | 2 | 0.0 % |
| **GDN family total** | **1,032** | **11.3 %** |

Two conclusions:

* **The GDN recurrence is not a lever** - 228 ms at 4K, and its kernel is a modest 192-block x 128-thread loop
  (Round 6).  That closes the question opened there.  The GDN family is 11.3 %, not the 21.6 % the single
  `gdn` entry suggested, because the mark previously absorbed everything up to the next phase's mark.
* **The projections are the GDN cost** (782 ms = 8.5 %), and they are exactly the `native_proj`/`bf16_proj`
  calls the WMMA work covers.

And the run exposed something more useful: **`hc read` - the six bf16 GEMMs per layer - is now the largest and
by far the most variable phase**, 1,921 ms (20.8 %) here against 892-990 ms (11.6 %) in earlier 4K runs, with
the same configuration.  That variance is the rocBLAS bf16 path behaving like the rocBLAS fp16 path did (the
1K A/B showed the rocBLAS arm swinging 3,029-4,312 ms while the WMMA arm held within 0.7 %), so the delegated
BF16 WMMA variant should both speed this phase up and stabilise it - and stabilising it may matter as much as
its average cost, since it is what moves whole-run prompt throughput between 7.6 s and 9.2 s.

## Round 23: BF16 WMMA lands - 1K pp reaches 385-409 tok/s (92-98 % of the CUDA counterpart)

`workflow-16` (the tight re-delegation after killing a stalled one) added a BF16 variant by templating the
existing kernels on the element type (`WmmaTraits<_Float16>` / `WmmaTraits<__bf16>`, the latter calling
`v_wmma_f32_16x16x16_bf16_w32` with the same fragment layout) and dispatching it from `Gemm::bf16` behind the
same env gate.  Its own stub check was the one I demanded:

    strata_wmma_gemm_f16 : 0x19d (413 bytes)
    strata_wmma_gemm_bf16: 0x19d (413 bytes)     <- not the 3-byte stub
    nm -C build-hip/strata-hip | grep -c gemm_wmma -> 8

**Verified independently before trusting it.** The probe's new NaN counter (which I required) immediately
reported 5.8M NaNs on the *fp16* suite - with 0.00e+00 error, the same NaN-blind artifact as before.  That
turned out to be the probe's own input generation: `(uint16_t)__float2half_rn(x)` converts `__half` to float
and truncates instead of taking its bits, so both sides computed on near-zero data.  Fixed in the tracked
probe.  `bench/tools/wmma_gemm_varied_check.cu` then checked both entry points against a host **fp64**
reference with varied exactly-representable inputs: `maxdev = 0.000e+00`, 0 wrong, 0 NaN on every shape,
including the engine's 12288x1023x2560 (12.6M outputs) for **both** fp16 and bf16.

**Measured** (interleaved pairs; `STRATA_WMMA_BF16=0` isolates the bf16 increment, `STRATA_WMMA_GEMM=0` forces
rocBLAS for both):

| tier | both WMMA | all rocBLAS | gain |
| --- | ---: | ---: | ---: |
| 1K | **2,502 / 2,661 ms (408.9 / 384.4 tok/s)** | 3,116 / 3,127 ms (328 / 327) | **-18 %** |
| 4K | **6,913 / 7,118 ms (592.4 / 575.3 tok/s)** | 8,040.5 / 8,039.6 ms (509.3) | **-11 to -14 %** |

Two incidental findings:

* **The mixed configuration is worse than either extreme**: fp16-on-WMMA with bf16-on-rocBLAS measured
  4,603 / 3,203 ms at 1K - slower than all-rocBLAS (3,116 / 3,127) and far more erratic.  So it is worth
  having both paths on WMMA, and the bf16 work was worth doing even though its own share is modest.
* The 8-token seed run still reproduces the reference tokens exactly
  (`271 7734 264 13280 9834 421 15339 279`) with the bf16 path live, so the accumulation-order change did not
  flip a single argmax there.

Current standing: pp@1K ~385-409 vs the counterpart's 419 (92-98 %), pp@4K 575-592 vs 893 (64-66 %).

## Round 24: the "tensor-core" batched attention is SLOWER than the old path on gfx1100 (and why the default stays)

The prompt attention has two implementations and a built-in harness (`build-hip/qsa_prompt_attn_parity_hip`,
which FP64-checks both and times them).  Running it:

| case | old (per-token) | batched (emulated mma16816) | ratio |
| --- | ---: | ---: | ---: |
| int8 ctx 4096, 1024 q | 19.36 ms | 22.90 ms | 0.85x |
| fp16 ctx 4096, 1024 q | 19.27 ms | 31.46 ms | **0.61x** |
| int8 ctx 32768, 2048 q | 36.73 ms | 52.13 ms | **0.70x** |
| fp16 ctx 32768, 2048 q | 40.16 ms | 59.52 ms | **0.67x** |

Both PASS the FP64 check (errors ~2e-06; they agree with each other to ~7e-06), so the batched path is not
*wrong* - it is a **pessimisation on this GPU**: its `mma16816` emulation (24 shuffles + ~64 FMAs per 16x8x16
tile) costs more than the batching saves, whereas the identical code on CUDA uses real tensor cores.

Confirmed in the engine with the existing switch (`STRATA_PROMPT_ATTN_OLD=1`), 4K prefill:

| rep | batched (default) | old path |
| --- | ---: | ---: |
| 1 | 6,709 ms total, `qsa attn` 1,072 ms (16.4 %) | 6,258 ms, **614 ms (10.1 %)** |
| 2 | 7,533 ms, `qsa attn` 1,129 ms (15.3 %) | 6,253 ms, **604 ms (9.9 %)** |

and at 1K the effect disappears (total 2,522/2,535 vs 2,507/2,519 ms; `qsa attn` 99-125 vs 57 ms), because the
attention is only 4-5 % of a short prefill.

**The default deliberately stays on the batched path.**  The old path is *not bitwise* with the CUDA engine -
the 8-token seed changes from `271 7734 264 13280 9834 421 15339 279` to
`271 7734 264 13280 9834 421 3817 4603` - and the batched path was written for exactly that bitwise parity
(its header says "accuracy but not bitwise" of the alternative).  So this is a real **perf-vs-parity choice the
owner should make**, not something to flip silently: `STRATA_PROMPT_ATTN_OLD=1` buys 7-17 % at 4K and ~0 at 1K
at the cost of that parity.  The proper fix is a WMMA prompt attention: it should beat *both* paths (the
emulated one at 1,115 ms and the old one at 604 ms per 4K prefill) while keeping fp32 accumulation, which is
also the numerically closest thing this GPU has to CUDA's real `mma.sync`.

## Round 25: is the attention-path accuracy change purely numeric? Yes - and two of my earlier claims were invalid

Prompted by a direct question, this was measured properly.  Separating the two changes:

**The WMMA GEMM work (fp16 and bf16) did not change accuracy.**  The 8-token seed output is bit-identical
across the change (`271 7734 264 13280 9834 421 15339 279`, repeated), and both kernels are exact against a
host fp64 reference on exactly-representable inputs (`maxdev = 0.000e+00`, 0 NaN, all shapes).

**The attention path swap is purely a summation-order difference.**  The repo's own harness
(`qsa_prompt_attn_parity_hip`) FP64-checks both implementations on prompt-shaped pools (2,051 selections):

| case | old vs FP64 | batched vs FP64 | batched vs old |
| --- | ---: | ---: | ---: |
| fp16 ctx 4096, 1024 q (output scale 3.21) | 1.54e-06 | 2.68e-06 | 5.19e-06 (1.4e-06 of scale) |
| int8 ctx 4096, 1024 q (scale 3.27) | 2.13e-06 | 3.69e-06 | 6.78e-06 (1.8e-06 of scale) |
| int8 ctx 32768, 2048 q (scale 3.62) | 2.13e-06 | 3.76e-06 | 6.56e-06 (1.7e-06 of scale) |

At token level, with a **fixed** expert cache (which makes the engine deterministic):

| prompt | batched vs batched (repeat) | batched vs old (path swap) |
| --- | --- | --- |
| 1K real text, 128 new tokens | identical | **identical** (no divergence) |
| 13-token seed, 128 new tokens | identical | first divergence at index **5** |

So the difference is normally invisible and only flips a token where the top-2 logits are within ~1e-6 of a
tie - the 1K prompt had none in 128 tokens, the seed prompt has one at index 5.

**Two corrections to earlier rounds in this file:**

* Round 20's "85 of 256 tokens agree" was **not valid evidence**: that run used `--expert-cache auto`, and the
  adaptive VRAM tier makes the engine non-deterministic - *batched vs batched* diverged at the same index 85.
  Only a fixed-cache comparison (where same-path repeats are bit-identical) means anything.
* The 8-token seed is a **boundary** check, not an equivalence check: the same path produces `421` as its 6th
  token at `--max-new 8` but `1608` at `--max-new 128`, because the verify window depends on the remaining
  budget.

**Unverified:** the logit vectors themselves could not be compared - `--dump-logits` writes only its 8-byte
header under the captured-graph path (its write site is in the non-graph sampling loop) and `--dump-final-r`
writes 0 bytes, while `--no-capture` requires `--no-pool` and changes the model.  "A near-tie flip" is
therefore inferred from the harness's FP64 accuracy and the divergence pattern, not measured at the logits.

### Correction + the WMMA attention result (same round)

**Correction to the note above.**  The claim that `--max-new 8` and `--max-new 128` give different tokens *for the
same path* was wrong: the engine binary was rebuilt at **12:43:08** by the WMMA-attention job *while those runs
were in flight* (`lock_*` finished 12:43:07 on the old binary and are valid; `seed_*` at 12:44 used the new one).
That is my own experiment-hygiene failure - I checked binary mtimes afterwards, not before.

**The WMMA prompt attention is verified and fast.**  `qsa_prompt_attn_parity_hip` (its "new" path is now the
WMMA kernel), 3 reps:

| case | old per-token | emulated batched | **WMMA** |
| --- | ---: | ---: | ---: |
| int8 ctx 4096, 1024 q | 19.20 ms | 22.90 ms (0.84x) | **5.17 ms (3.71x)** |
| fp16 ctx 4096, 1024 q | 19.00 | 31.16 (0.62x) | **6.44 ms (2.95x)** |
| int8 ctx 1500, 1024 q | 11.47 | 11.70 (0.96x) | **2.75 ms (4.17x)** |
| int8 ctx 2100, 256 q | 6.63 | 6.04 (1.09x) | **1.63 ms (4.07x)** |

`FAILURES: 0` in both configurations, and the WMMA path's own error against FP64 is 3.65-4.96e-06 against the
emulated path's 2.68-5.19e-06 - the same order, so it is not a precision regression.  If the 4K `qsa attn`
phase (1,115 ms) scales like the harness suggests (~3.7x), this is worth roughly 11 % of the 4K prefill.

**But it changes the port's 8-token regression artifact**, which must be an explicit decision: with the three
paths on one stable binary and a fixed cache,

| path | seed output |
| --- | --- |
| emulated (previous default) | `271 7734 264 13280 9834 421 15339 279`  <- the reference |
| old per-token | `271 7734 264 13280 9834 421 3817 4603` |
| WMMA (new default) | `271 7734 264 13280 9834 1608 83871 8708` |

All three agree for five tokens and then each diverges at a different place.  `STRATA_PA_WMMA=0` restores the
exact reference in one env var, so parity remains available; the default is now the fast path, which is the
stated objective, but the artifact should be re-baselined rather than treated as a failure.

**Also flagged from the diff review:** the new code self-defines `STRATA_WMMA_GFX11` for *any* HIP build
(`#if defined(STRATA_BACKEND_HIP) && !defined(STRATA_WMMA_GFX11)`) instead of taking it from the build's
architecture setting.  On this repo (gfx1100 only) that is harmless, and the device code does emit real
`v_wmma_f32_16x16x16_f16` instructions, but on a non-gfx11 HIP target it would try to compile gfx11 intrinsics
rather than fall back.  It should be moved behind the same build-provided define the prefill targets use.

## Round 26: the WMMA prompt attention lands - pp@1K reaches the CUDA counterpart (408-420 vs 419 tok/s)

`workflow-17` replaced the emulated `mma16816` inner loop of the batched prompt attention with real
`v_wmma_f32_16x16x16_f16_w32` (scores GEMM as 16x32 across two waves, output GEMM as 4 warps x 64 dims), keeping
the emulation as the non-gfx11 fallback and the A/B arm (`STRATA_PA_WMMA=0`; `STRATA_PROMPT_ATTN_OLD=1` still
selects the per-token path).  Verified, not taken on trust:

* the unbundled gfx1100 code object contains `v_wmma_f32_16x16x16_f16` instructions;
* `qsa_prompt_attn_parity_hip`: **FAILURES 0**, and the phase-level speedups are 3-5x per case (int8 ctx 4096
  5.17 ms vs 19.20 old / 22.90 emulated; fp16 6.44 vs 19.00 / 31.16; ctx 1500 2.54-2.75 vs 11.5-12.6 / 11.7);
* its own FP64 error (3.65-4.96e-06) is the same order as the emulated path's (2.68-5.19e-06), so it is not a
  precision regression.

**In the engine (4K):**

| | before | after |
| --- | ---: | ---: |
| GPU timeline | 6,913-7,118 ms | **5,929 ms** |
| `qsa attn` phase | 1,115 ms (15.7 %) | **201 ms (3.4 %)** |

**End-to-end A/B:**

| tier | WMMA attention | emulated | note |
| --- | ---: | ---: | --- |
| 4K | **6,045.0 ms (677.4 tok/s)** | 6,610.8 ms (619.4) | +9.4 %, one pair |
| 1K, 3 interleaved pairs | **2,507 / 2,438 / 2,451 ms (408.0 / 419.6 / 417.3)** | 2,595 / 3,057 / 2,784 ms (394.2 / 334.7 / 367.5) | WMMA wins all three |

The first single 1K pair showed the opposite sign (2,842 vs 2,557 ms), which is why the repeat mattered: the
emulated arm swings +-10 % run to run while the WMMA arm holds +-1.4 %, the same stability difference the GEMM
work showed.  **pp@1K is now 408-420 tok/s against the CUDA counterpart's 419 (97-100 %), and pp@4K 677 against
893 (76 %).**

**Guard fix in the same commit.**  The new code self-defined `STRATA_WMMA_GFX11` for *any* HIP build
(`#if defined(STRATA_BACKEND_HIP) && !defined(...)`), which cannot tell gfx11 from any other HIP target.  Moved
to the build: `target_compile_definitions(strata_kernels_hip PRIVATE STRATA_WMMA_GFX11=1)` under the same
`CMAKE_HIP_ARCHITECTURES MATCHES gfx11` condition the prefill target uses, and removed the self-definition.
Verified after rebuilding: the define is in `strata_kernels_hip`'s flags, the code is *still* compiled in (4
`launch_wmma` symbols in the object - the check that matters, since a guard mistake here compiles the fallback
silently), and the harness still reports 4-5x with FAILURES 0.

## Round 27: the 32K tier gains most from the WMMA attention (pp 690.6 tok/s, +28 %), and decode is untouched

The prompt attention's share grows with context, so the 32K tier benefits most.  Measured with the recommended
configuration (`--expert-cache auto --prefill 2048 --spec 4 --spec-min-p 0.5 --adapt-every 2 --mtp ...`):

| metric | value | counterpart | ratio |
| --- | ---: | ---: | ---: |
| pp @32K | **690.6 tok/s** (47,448 ms / 32,767 tokens, 16 chunks) | ~1,490 | **46 %** (was 35-38 %) |
| pp @1K (same run) | 410.2 tok/s | 419 | 98 % |
| tg @1K, 256 tokens | **49.47 tok/s** | 50.5 | 98 % |
| tg @32K, 128 tokens | 37.87 tok/s (acceptance 0.913, 3.79 tokens/round) | 49.0 | 77 % |

The decode path is unchanged by construction (the prompt attention is prefill-only), and the 1K decode confirms
it at 49.47 tok/s.  The 32K decode number is not directly comparable to the earlier 40.5 tok/s: it moved with
the acceptance rate, which the *prefill's* numerics influence (the new attention shifts the trajectory at
~1e-6, exactly as the path comparison in Round 25 showed).

The 32K phase table (47,375 ms) now reads:

| phase | ms | share |
| --- | ---: | ---: |
| wait copy | 9,772 | 20.6 % |
| dequant (the per-expert gather) | 9,202 | 19.4 % |
| gemm gate/up | 6,306 | 13.3 % |
| gdn (qkv/gate/alpha/beta) | 4,360 | 9.2 % |
| gemm down | 3,247 | 6.9 % |
| hc read | 2,728 | 5.8 % |
| **qsa attn** | **2,420** | **5.1 %** |
| qsa proj | 1,879 | 4.0 % |
| gdn rec | 1,732 | 3.7 % |
| gdn out | 1,643 | 3.5 % |
| the rest (embed, router, ple, combine, select, host grouping, gather) | ~2,000 | 4.4 % |

So at 32K the two streaming-related phases (`wait copy` + the gather that waits on the same arrivals) are
**40 %** of the prefill and the expert MMQ GEMMs another 20 % - i.e. the long-context tier is dominated by
moving experts across this box's PCIe 4.0 x8 link (14.2 GB/s measured; 262,555 blobs streamed, 115,481
resident of 512/layer at 193 of 512 for a 4K run).  That is the remaining structural gap against a counterpart
on PCIe 5.0 x16, not kernel efficiency.

### Quantifying the 32K streaming floor: the gap is the link, but ~20 % of it is unoverlapped

From the same run's own counters: 262,555 expert fetches of 2,046,400 B = **537 GB in 47.4 s = 11.3 GB/s**,
against the **14.2 GB/s** this box's PCIe 4.0 x8 link measured in a standalone H2D probe - so the stream already
runs at **80 % of the link**.

| | value |
| --- | ---: |
| link time if perfectly overlapped | 37.8 s of the 47.4 s wall |
| unhidden streaming (= `wait copy` phase) | **9.6-9.8 s = 21 %** |
| a PCIe 5.0 x16 counterpart at ~52 GB/s | 10.3 s of DMA = 47 % of its 22.0 s prefill |

Two conclusions, and they point in the same direction:

1. The 4K/32K prompt-throughput gap against the counterpart is **structural**: the same byte volume crosses a
   link with roughly a quarter of the bandwidth, so no amount of kernel work removes it.  (The two streaming
   phases being 40 % of the 32K prefill is the same statement in phase-table form.)
2. It is not *entirely* structural.  The link is busy 80 % of the wall, so **21 % of the 32K prefill is expert
   streaming that is not hidden behind compute** - and the two independent measurements (the phase table's
   `wait copy` at 9,772 ms and the arithmetic's 9.6 s) agree to within 2 %.  That is the remaining software
   lever at long context: prefetching deeper/earlier so the copy engine never idles while the SMs wait, which is
   what `STRATA_PREFILL_RING` and the chunked stream plan govern.

### Round 28: the 21% "recoverable" streaming was wrong - the honest state of the link question

The previous entry claimed the unoverlapped 21% was a software lever (`STRATA_PREFILL_RING`, prefetch depth).
That claim does not survive testing:

| hypothesis | test | result |
| --- | --- | --- |
| prefetch window too shallow (the prefetch window is measured in *order entries*, and 31 % of placements are already resident, so runs of resident experts issue no fetches) | `STAGE` 8 -> 16 in `prefill.cpp` | `wait copy` 9,772 -> 9,728 ms (unchanged) and the total got *worse* (47,375 -> 49,431 ms).  Reverted. |
| VRAM contention between the DMA writes and the compute kernels | `bench/tools/h2d_contention_probes.cu` #1: 1024 x 2 MB H2D, idle vs under a saturating SM VRAM-copy kernel | 13.2 / 13.5 / 13.6 GB/s - **no effect** |
| scattered host reads (the same 2 MB region is L3-friendly; the arena is 47 GB) | probe #2: scattered 2 MB regions in a 6 GiB pinned arena | 13.5 GB/s vs 13.1 for the same region - **no effect** |
| 8 rotating VRAM destination slots | probe #3 | 13.5 GB/s - **no effect** |
| other traffic sharing the copy engine | probe #3 with a concurrent D2D stream | 12.7 GB/s - **-6 %**, a partial contributor |

So the engine's stream runs at 11.3 GB/s against a 13.5 GB/s probe peak (84 %), and none of the mechanical
explanations account for it.  What is left, in order of plausibility:

1. **host dispatch coupling**: at 32K the prefill issues ~262,555 DMA copies *plus* roughly a million kernel
   launches (a gather and two MMQ launches per expert per layer) from one host thread, while the ring buffers
   only ~1-2 ms of work - so any host hiccup stalls the DMA pipeline.  The engine reports only the *issue* cost
   (`host staging 701 ms`), not total dispatch time, so this is unmeasured rather than disproved.
2. accumulated small copy-engine overheads (each transfer is 2 MB, so ~148 us of data plus per-transfer setup,
   and probe #3 shows a -6 % tax the moment anything else shares the engine).

The next instrument is a rocprofv3 trace of a 32K prefill: it would show the copy engine's busy fraction and,
crucially, *where* its gaps sit (during resident runs? at layer boundaries? at chunk boundaries?).  That is a
direct measurement, unlike the phase table's aggregate.

## Round 29: batching the gather removes 6 s of launch overhead from one phase and it reappears in the next - the launches were not on the critical path

`workflow-18` added `mmq::gather_native_batch` (2D-grid copy kernels, one launch per 16-expert group, pointers
passed by value in the kernarg) and a call site that defers the gather to the group's last expert, with
`STRATA_MMQ_GATHER=per_expert` as the A/B arm.  It also added a fallback I had not asked for but which is
correct: batching is disabled when the staging ring is smaller than `MMQ_GROUP`, which would otherwise
deadlock waiting for slots the group is holding.

32K A/B (`--max-context 36864 --max-new 8`, `STRATA_PREFILL_TIMING=1`):

| | batched (default) | per_expert |
| --- | ---: | ---: |
| total | 48,442.6 ms (676.4 tok/s) | 49,241.1 ms (665.4) |
| `dequant` | **3,975 ms (8.2 %)** | 9,921 ms (20.2 %) |
| `wait copy` | **14,955 ms (30.9 %)** | 9,829 ms (20.0 %) |
| the two together | **18,930 ms** | 19,750 ms |

So the batching **halved the gather phase (-5,946 ms)** - i.e. ~6 s of that phase really was launch overhead,
and the Round 16 explanation ("mostly arrival waits") was wrong about the mechanism - but **almost the same
amount reappeared as `wait copy` (+5,126 ms)**: the idle *moved* instead of disappearing, and the total moved
only 1.6 %, inside the run-to-run spread.  The combined pair of phases (-820 ms) accounts for the total
difference (-798 ms), which is a useful cross-check that the phase marks are consistent.

The conclusion is that the per-expert gather launches were **filling GPU time that was already idle**, waiting
on the expert stream - they were never on the critical path.  The change is kept (it removes 246,145 launches
per 32K prefill, so less host work and less risk of host-side hiccups, and it is reversible through the env
gate), but it is not a performance win and the report says so.

This also sharpens the standing question.  The 32K run uses the `stream_all` walk (`m.ring` 384 > `STAGE`), in
which the chunk's non-resident experts are issued in layer order through a 384-slot ring - so the copy stream
is *not* short of queued work, yet it still runs at 11.3 GB/s against the 13.5 GB/s probe peak.  The remaining
explanation is that the copy engine is shared with the engine's other DMA traffic (pool rows, KV staging, PLE
rows, doorbell payloads), which the concurrent-D2D probe reproduced as a -6 % tax.  If so, the link is busy
close to 100 % with a *mix* of useful traffic, and the "16 % idle" is not idle at all.  `rocprofv3
--memory-copy-trace` on a 4K prefill can settle it by summing the copied bytes by category.

## Round 30: the copy engine is idle in 54 discrete ~26 ms gaps, and the 4K variance is host-side, not GPU-side

`rocprofv3 --memory-copy-trace` on a 4K prefill (38,834 transfers, 4 MB of CSV) answers where the copy engine's
"idle" goes, and corrects two of my own stories.

**When the copy engine is busy it runs at full speed.**  In the prefill window (5,412 ms, the span of the expert
stream) there are **26,714 copies of 2 MB-class duration = 54.6 GB in 4,001 ms of copy-busy time = 13.6 GB/s** -
exactly the standalone probe's peak.  (The 11,780 transfers I first saw on another stream are in the *model
load* phase, not the prefill; inside the window the only other traffic is 5 ms.)  So the link is not slow and
not contention-limited: the loss is entirely **gaps**, and they are not diffuse.

**The gaps are 54 discrete events of ~26 ms each** (25.4-27.0 ms, total 1,413 ms; the remaining 24,097
inter-copy gaps are 10-50 us of normal spacing).  Everything mechanical was tested and excluded:

| hypothesis | test | result |
| --- | --- | --- |
| ring too small to bridge a layer's non-MoE work | `STAGE` 8 -> 16 | no change (Round 28) |
| ring depth at 4K | `STRATA_PREFILL_RING` 96 / 192 / 384, 2 reps each | inconclusive: the same configuration spans 5,976-7,462 ms |
| the adaptive cache tier | captured `experts streamed` / `resident` on all six runs | **identical** (26,696-26,738 / 16,232-16,264) - not the cause |
| GPU shared with the user's work | `rocm-smi --showuse` + `fuser /dev/kfd` | GPU 0 % busy, nothing holds `/dev/kfd` |
| CPU throttling | governor `powersave`, but 5.4 GHz measured | not throttled |

**The variance is host-side.**  Capturing the engine's own host breakdown over three identical 4K runs:

| rep | total | chunk setup (PLE rows, the expert stream plan) |
| --- | ---: | ---: |
| 1 | 6,982 ms | **720 ms** |
| 2 | 6,015 ms | 204 ms |
| 3 | 5,996 ms | 202 ms |

so the slow run is a **3.5x larger host chunk-setup**, i.e. the PLE I/O path - and that also makes the 26 ms
copy-engine gaps plausible as the *same* host stall propagating into the DMA issue loop.

**The default I/O mode is the right one.**  `--ple-io` has two modes: `direct` (default, unbuffered SSD) and
`mmap`:

| | `direct` | `mmap` |
| --- | ---: | ---: |
| setup, warm | 201-207 ms (stable) | **45-47 ms** |
| setup, cold | up to 720 ms | **3,265 ms** (10.7 s prefill) |

`mmap` is 4x faster warm but catastrophic when the pages are cold, so **the default stays**, and the follow-up
worth doing is making the PLE rows cheap *and* predictable (preload, read-ahead or a larger row cache), since
that is worth up to ~0.5 s of a 6 s prefill and removes the ±11 % tail that has been confusing 4K A/Bs all
session.

**How to read earlier 4K numbers:** the *within-run phase attributions* are sound (a 5.5x change in `qsa attn`,
a 2-5x change in `hc read`), but 4K *totals* carry a +/-11 % host-side spread, so single-pair 4K comparisons in
this file are weaker than the 1K ones (whose WMMA arms held within 1.4 %).

## Round 31: the host setup is the embeddings (stable), and my Round-27 variance attribution was wrong

Split the "chunk setup" with two extra counters (kept; they are cheap and only print under
`STRATA_PREFILL_TIMING`):

| rep | total | embeddings | stream plan + first issue |
| --- | ---: | ---: | ---: |
| 1 | 5,971.6 ms | 239 ms | **0 ms** |
| 2 | **8,712.6 ms** | 202 ms | **0 ms** |
| 3 | 5,937.4 ms | 206 ms | **0 ms** |

Two conclusions:

* The setup is the **embedding gather** (202-239 ms, stable) - *not* the PLE rows (the print's label is stale) and
  not the stream plan, which is 0 ms at this chunk size.
* **The Round-27 attribution was wrong.**  That entry blamed the slow runs on a 3.5x larger host chunk setup
  (720 vs 202 ms) from a single pair.  Here rep2 ran **8,713 ms - a 47 % outlier - with a completely normal
  setup (202 ms)**, so the correlation did not hold.  The host setup is not the variance source.

Also tested and rejected: `--ple-row-cache` 1M/4M/16M rows (no systematic effect; one 4M run spiked to a 966 ms
PLE), and sampling `pp_dpm_sclk` during runs (inconclusive - it reported level 1 as 0-185 MHz while the same
runs took 6 s, so it was not reading the active card's clock).

**Consequence for reading this file.**  The 4K prefill's run-to-run spread is much larger than the ~3 % assumed
early on: over the last twelve 4K runs the totals were 5,937-8,713 ms, i.e. up to +47 %.  Single-pair 4K
comparisons are therefore weak evidence, and this applies retroactively to a few entries above - notably the
Round 21 WMMA-attention numbers (6,045 vs 6,611 ms, +9.4 %), which are *inside* the spread.  What remains solid
is the *within-run phase evidence*, where the same change moved `qsa attn` from 1,115 ms to 201 ms (a 5.5x
change, and the phase shares are computed against the run's own total).  1K comparisons are unaffected: the
WMMA arms there held within 1.4 %.

## Round 32: a quantitative decode cost model - tg is bound by CPU expert bytes, not attention

The engine's own `STRATA_VERIFY_PROFILE=1` gives a per-round breakdown, and it fits a simple model.  At 1K:

    verify window   wait for rings 34.517  pool 33.580  host 1.075  commit 2.683 ms/round
                    CPU experts 8.99 distinct / 12.43 routed per layer
    pool multi      gate/up 20.352  quantize 0.078  down 12.436 ms/round; 27.0 GB/s; CPU pool call 33.511
    dispatch        plan 0.201  activation quantize 0.331  jobs 0.101  run 32.872 ms/round
    pcie experts    3.22 distinct experts per layer read over PCIe (share 77/256 of the misses)
    mtp             4.099 ms/round drafting

and at 32K the same lines read 61.367 / 60.579 / 59.621 / 60.482 ms with **16.25 distinct of 24.15 routed
per layer**.  The model is just the bytes:

| tier | distinct CPU experts/layer | bytes/round (x 48 layers x 2,046,400 B) | at the pool's 27 GB/s | profile's pool |
| --- | ---: | ---: | ---: | ---: |
| 1K | 8.99 | 883 MB | 32.7 ms | 33.6 ms |
| 32K | 16.25 | 1,596 MB | 59.1 ms | 60.6 ms |

agreement within 3 %.  So **decode throughput is bound by the CPU expert path's memory bandwidth**, and the
1K -> 32K slowdown is *routing breadth* (a longer context routes each token to more distinct experts: 8.99 ->
16.25 distinct per layer, +81 %), not attention and not KV size.  The pool itself is 5 workers + the host thread
on this 6-core/12-thread box - every physical core - and its header documents 36.32 GB/s standalone against the
27 GB/s achieved in-engine, so it is already near the machine's memory bandwidth.

That reframes the tg@32K gap (37.87 vs 49.0 tok/s): the extra ~27 ms/round is expert *bytes*, so the lever is
tier policy (keep the hot set resident), not kernel work on the attention.  Note also that the profiled items
sum to ~41 ms of an ~81 ms round at 1K, i.e. **~39 ms/round of the decode is in the draft/sampling path**, which
the profile does not label - a second, unexamined target.

## Round 33: the decode round is linear in the verify window (~13-15 ms per verified token), and spec 3-4 is the measured optimum

Varying the window (which is `min(spec + 2, kVerifyMaxT)`), 64 fresh tokens, 1K prompt, recommended options
otherwise:

| config | window | ms/round | tokens/round | **ms per window slot** | tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| `--spec 2`, no MTP | 4 | 52.0 | 1.38 | 13.0 | 26.5 |
| `--spec 3` + MTP | 5 | 68.0 | 2.78 | 13.6 | 40.9 |
| `--spec 4` + MTP | 6 | 85.7 | 3.56 | 14.3 | **41.5** |
| `--spec 6` + MTP | 8 | 123.2 | 4.00 | 15.4 | 32.5 |

So the round costs **13-15 ms per verified token**, independent of what the profile labels, and the optimum is
where tokens/round stops outgrowing the per-round cost: **spec 3-4 (40.9/41.5 tok/s)**, with spec 6 losing
badly (32.5).  That is an independent confirmation of the configuration this file has recommended since the MTP
work.

This also corrects the previous entry's framing.  There, I subtracted the profiled items from the round and
called the remainder "the draft/sampling path, ~39 ms/round".  The window-scaling here shows the scaling
quantity is **per verified token, not per round**, and that the profiled items are *nested* (at 32K they sum to
129.5 ms against a 100 ms round, which is only possible if the window line contains the pool line).  So the
honest statement is: the round grows ~13-15 ms per window slot and the profile accounts for roughly a third of
that growth; **the identity of the remaining per-token cost is not yet attributed.**

Ruled out as the explanation: the LM head on the CPU (`src/core/native_head.cpp` uploads `output.weight` to the
device, and its FP64-accurate path is a GPU kernel), the QSA attention (the selection reads a bounded cell set,
~0.05 ms of MACs per token), and the MTP drafting (the profile measures 1.5-4.2 ms per *round*).  What would
settle it is a timer inside the window itself, per verified token, rather than the current per-round labels.

## Round 34: the decode is ~65 % expert-staging handshake and ~25 % arithmetic (kernel trace)

`rocprofv3 --kernel-trace` on a 12-token decode (`--spec 4` + MTP, 1K prompt), split at the first
`wait_flag_ge_kernel` and aggregated over the 12,638 decode-side kernels (the run took 384.9 ms under tracing,
so read shares, not absolutes; and note tracing *inflates waits* specifically):

| kernel | total | share | launches |
| --- | ---: | ---: | ---: |
| `fetch_blobs_kernel` | 96.6 ms | 24.7 % | 192 |
| `__amd_rocclr_streamOpsWait` | 92.2 ms | 23.6 % | 192 |
| `wait_flag_ge_kernel` | 64.3 ms | 16.5 % | 576 |
| `native_mmvq_m...` (all variants) | ~40 ms | ~10 % | 654 |
| `native_down_k...` | 13.1 ms | 3.4 % | 312 |
| `gr_down_multi` / `gr_up_multi` / `gr_norm_multi` | 17.0 ms | 4.4 % | 1,209 |
| `__amd_rocclr_copyBuffer` | 13.6 ms | 3.5 % | 982 |

Two facts stand out:

* **192 = 48 layers x 4 rounds**, so the staging handshake happens once per layer per round - 503 us of
  `fetch_blobs` and 480 us of stream-ops wait per layer.  **~65 % of the decode is that plumbing and only ~25 %
  is arithmetic.**  The math kernels sum to a quarter *even with waits inflated by tracing*, so the direction is
  robust even though the exact share is not.
* The untraced round (85.7 ms at this window) is close to the profiled window (34.5 ms, which *contains* the
  pool's 33.6 ms) **plus** the staging time, which is what a *serialised* producer/consumer would look like: the
  GPU appears to stall until the CPU pool delivers each layer, rather than computing layer L while the pool
  produces L+1.  That is a hypothesis, not yet a measurement - the test is a per-verified-token timer inside the
  window, since the current labels are per round and nested.

If it holds, the lever is the same one the prefill's `wait copy` pointed at, one level down: **overlap the
expert staging with the compute instead of lock-stepping per layer.**  The payload is large - the decode is
41.5 tok/s against the counterpart's 50.5 at 1K and 37.9 against 49.0 at 32K.

## Round 35: the decode cost model closes - PCIe transfer + exposed host-pool latency + math = the round

Following `fetch_blobs` to its caller (verify.cpp:788) resolves the previous entry's puzzle and refutes the
"50x kernel inefficiency" I briefly inferred:

* `fetch_blobs` gathers the layer's **PCIe share**, which the profile reports as **3.22 distinct experts per
  layer** (share 77/256 of the misses).  3.22 x 2.05 MB = 6.6 MB; at the link's 13 GB/s that is **507 us/layer**
  against the trace's 503 us, i.e. **the kernel is the PCIe transfer and is already at the link rate** - a shader
  reading host-mapped expert blobs, not a slow gather.
* Independent probe (`/tmp/gather_probe.cu`, kept as evidence): the same kernel over *VRAM* sources reaches
  **220-367 GB/s** and beats the copy engine's D2D at 166-258 GB/s, so the kernel itself is fine.
* The doorbell waits (`streamOpsWait` 92.2 + `wait_flag_ge_kernel` 64.3 = 156.5 ms over 4 rounds) are
  **39.1 ms/round**, and the host's pool work is 700 us/layer (33.6 ms / 48 layers).  **The GPU waits on the host
  pool per layer instead of computing layer L while the pool produces L+1** - the serialisation hypothesis of the
  previous entry, confirmed after I had talked myself out of it.

The round then closes to within 1 %:

| component | ms/round | evidence |
| --- | ---: | --- |
| PCIe expert transfer | 24.3 | trace 96.6 ms / 4 rounds = 24.2 |
| doorbell / handshake waits | 39.1 | trace 156.5 ms / 4 rounds |
| arithmetic (mmvq, down, gr_*) | ~22 | trace |
| **sum** | **~85** | **measured round 85.7 ms** |

So the tg lever is concrete and large: **overlap the host expert pool with the device compute** rather than
lock-stepping per layer through the doorbells.  If the 33-39 ms of exposed pool/handshake time were hidden, the
round would be bounded by the PCIe transfer plus arithmetic (~46 ms), which is why the payload looks like
40 %+ - though the two transfers and the compute overlap imperfectly in practice, so treat that as an upper
bound rather than a prediction.

## Round 36: the decode's shader fetch is a deliberate robustness choice, not an oversight - and a delegation was wasted

Following the previous entry's lever ("move the PCIe expert transfer off the blocking shader onto the copy
engine"), I delegated that implementation - then found, by reading the call site, that **the copy-engine path
already exists** as an option and the shader is the *deliberate* default:

    src/program/generate.cpp:3163
    // issue #31's thread dumps show the host stuck in that cudaMemcpyAsync on a driver lock for good.  The copy
    // kernel needs no host CUDA call there, and costs ~1-3% decode on IQ3_S (45.3 -> 44.8 tok/s, 8 requests).
    ver.set_pcie_mode(o.pcie_mode == "dma" ? 0 : o.pcie_mode == "direct" ? 1 : 2);

`--pcie-mode` takes `auto | dma | kernel | direct`; `auto` and `kernel` select the shader `fetch_blobs` (mode 2),
`dma` selects mode 0, `direct` mode 1.  Measured (64-token decode, 1K): **dma 44.65 tok/s, kernel 42.22,
auto 41.89, direct 41.03**, with the window's `wait for rings` falling 36.5 -> 31.8 ms - so the DMA path really
is faster, as the mechanism predicted.  At 32K (32 tokens): dma 34.30 vs auto 32.91 tok/s (+4.2 %); at 1K on
that shorter sample they were level (41.08 vs 41.32).

**But it must not become the default.**  Issue #31 is a host hang inside `cudaMemcpyAsync` on a driver lock, and
the shader path exists precisely to avoid a host CUDA call there.  The trade the comment records (~1-3 % decode)
matches what I measured (~0-6 %), so there is nothing to fix by flipping the default - it would reintroduce a
hang to chase a few percent.

Two process notes, both mine:

* The delegation (`workflow-19`, "move the fetch to the copy engine") was **redundant** - it was asked to build
  what `--pcie-mode dma` already provides.  I killed it.  AGENTS.md's "prefer an A/B switch that already exists"
  applies to options the codebase already has, and I searched the wrong place (the kernel) instead of the
  command-line surface.
* The default flip I was about to make was **wrong**, and the call-site comment is what stopped it.  Reading the
  comment above a line before changing that line is cheap; shipping a hang is not.

## Round 37: the decode is volume-bound, not lockstep-bound - a hypothesis refuted by a fit, which closes the decode thread

The previous entry left "overlap the host pool with the device compute" as the lever, on the observation that
both sides wait per layer (the host's `ms_wait` is 36.5 ms/round = 760 us/layer; the GPU's doorbell waits are
39.1 ms/round = 815 us/layer - a ping-pong).  A fixed per-layer handshake would show up as a large positive
constant in the window scaling.  Fitting the four measured points:

    round = 17.9 ms x window - 20.7 ms        (windows 4, 5, 6, 8 -> 52.0, 68.0, 85.7, 123.2 ms)

**The intercept is negative**, so there is no fixed per-layer cost to remove: the round is *volume-driven*, and
the per-slot cost rises with the window (13.0 -> 15.4 ms) because a wider window routes more distinct experts
and therefore moves more bytes.  The lockstep is real but it is not the bottleneck, and its cost scales with the
same volume.

So the decode is at a principled local optimum: cost = volume, the window optimum (`--spec 3-4`) is the balance
point between tokens/round and bytes/round, and the levers that remain are the same memory-hierarchy ones the
prefill hit - the PCIe 4.0 x8 link (the DMA alternative is faster but hits issue #31's host hang) and the CPU
expert pool, which runs at the machine's memory bandwidth with every physical core in use.

**Caveat on the counterpart numbers.**  The CUDA-side figures in this file (50.5 tok/s at 1K, 49.0 at 32K) come
from earlier rounds and I have not re-verified their exact configuration in this session.  The 1K comparison is
the one I have re-measured repeatedly on my side (49.47 tok/s, 256 tokens); the 32K ratio (37.87 vs 49.0) should
be treated as indicative until both sides are re-run with identical options.

**What is still unattributed and still worth attacking:** the prefill's copy-engine *gaps* - 54 discrete ~26 ms
stalls at 4K (1,413 ms of 5,412 ms, 26 %), which the mechanical explanations (ring depth, VRAM contention,
scattered reads, rotating slots, adaptive tier, GPU sharing, CPU frequency, host chunk-setup) have all failed to
explain.  That is ~24 % of the 4K prefill and the last unexplained item in the model.

## Round 38: the copy-engine gaps are an ordering effect, not bandwidth - the last unexplained item now has a mechanism

Correlating the two traces **from the same 4K run** (`--memory-copy-trace --kernel-trace`) settles what the
copy engine stalls on.  The expert stream has 66 gaps > 1 ms totalling **1,507 ms**, and **84 % of that time has
a kernel running**:

| kernel overlapping the gaps | gap time | launches |
| --- | ---: | ---: |
| `gemm_wmma_64x64_4w<half>` | 394 ms | 301 |
| `gemm_wmma_64x64_4w<bf16>` | 212 ms | 339 |
| `prompt_attn_wmma_kernel` | 167 ms | 12 |
| `gdn_rec_cols_kernel` | 97 ms | 37 |
| `dequant_kernel<14,...>` | 80 ms | 128 |
| `mul_mat_q<...>` (MMQ) | 68 ms + 28 + 26 | 53 |

So over half the stall time coincides with the dense WMMA projections and attention - the kernels this session
added.

**It is not bandwidth.**  My earlier probe used a plain streaming-copy kernel as the load, which is not what the
engine runs, so I re-ran it with the actual kernel (`bench/tools/dma_vs_wmma_probe.cu`, the engine's
`strata_wmma_gemm_f16` on its `12288x1023x2560` shape, launched continuously on another stream):

    idle 12.9 GB/s   during gemm_wmma 13.3 GB/s   idle again 13.5 GB/s

The copy engine and a saturating WMMA GEMM coexist fine.  Since the copies are enqueued on a separate stream and
the issuing is driven by the walk's progress (`give_back` -> `issue_until(consumed + ring)`), the mechanism that
fits the evidence is **ordering, not contention**: while the walk is inside a long non-MoE kernel it consumes no
experts, so the copy stream runs out of supply it is allowed to start (each copy's destination slot is released
by an event recorded on the compute stream) and idles.  That is ~26 % of the 4K prefill.

**Fix direction** (delicate, same class as the decode's): decouple the expert-copy supply from the compute
stream's progress - deeper issuing that does not wait on consumption, or slot-free signalling that does not ride
on the compute stream - with the ring's capacity as the bound.  I am recording it as a mechanism with a
direction, not as a verified fix, because the pipeline change needs its own careful measurement.

## Round 39: the copy-engine idle is attributed but has no found lever - and the ring A/B is inconclusive

Two hypotheses tested and closed:

* **The idle is real idle.**  In the same 4K run, other streams' copies overlap the 53 gaps by **3 ms out of
  1,032 ms (0 %)**.  The engine is not busy with someone else's work.
* **The ring depth is not the lever.**  `STRATA_PREFILL_RING=384` (against the default 96, since
  `g_pinned_share` is below 0.9 here: 96 x ~150 us is 14 ms of buffer against ~26 ms stalls) gave gaps of
  1,242 ms versus 1,032, with one 224 ms stall the default run did not have, and a slower prefill
  (7,031 vs 5,821 ms).  The cache trade-off I expected does **not** explain it either: `experts streamed` and
  `resident` were identical (26,753/16,238 vs 26,764/16,241).

But the honest caveat is that **the gap metric is itself noisy**: across three 4K traces the total gap was 1,413,
1,507 and 1,032 ms - a +-20 % spread on the very quantity I was trying to use for a low-noise comparison.  So
the ring result is *inconclusive* rather than a clean refutation, and with wall-clock spreads of +-11-47 % on
this box, micro-tuning this last item is not a productive use of measurement time.

**Status of the item:** attributed (the copy supply is ordered behind the compute stream's consumption, which is
by design - "a slot is refilled only once the compute stream has recorded that it is done with it"), coincident
with the dense WMMA projections and attention (84 % of the gap time has a kernel running, over half of it those
kernels), and *without* a found lever: not bandwidth, not other traffic, not the ring.

**Remaining candidates, with their measured sizes** (so the next round can pick on evidence rather than
appetite):

| target | size | note |
| --- | ---: | --- |
| expert MMQ GEMMs | 9,553 ms of 47,375 = **20 %** at 32K | llama.cpp's kernels; ~11 TFLOPS effective against the card's ~123 INT8 TOPS, so there is apparent headroom, but it is the largest and longest-shot project left |
| copy-engine idle | 1.0-1.5 s = **18-26 %** at 4K | attributed, no lever found |
| GDN recurrence | 1,732 ms = **3.7 %** at 32K | 192 blocks x 128 threads = 24k lanes on a GPU with ~196k: a serial scan using ~12 % of the machine, so a bounded kernel project with a clean target |
| PCIe expert transfer (decode) | 24 ms/round = 28 % | link-bound; the DMA alternative hits issue #31 |
| CPU expert pool | 33.6-60.6 ms/round | at the machine's memory bandwidth with every physical core in use |

## GDN recurrence prefetch: works, but my prediction was 10x too optimistic - and dequant+WMMA for the experts is refuted

**GDN recurrence prefetch (`workflow-21`, kept).**  `gdn_rec_cols_kernel` is now a template with a pipelined
schedule that issues the next token's loads (the `h` rows, `gate` and `beta`) into registers before the current
token's barriers, with the baseline preserved verbatim as `<false>` behind `STRATA_GDN_REC_PREFETCH=0`.  The
arithmetic, shared-write order and barrier placement are unchanged, and the 8-token seed confirms it: with the
default (WMMA) attention the output is bit-identical to what it was before this change, and with
`STRATA_PA_WMMA=0` it still reproduces the reference token for token.

Measured `gdn rec` phase:

| tier | prefetch on | prefetch off | gain |
| --- | ---: | ---: | ---: |
| 4K | 198 ms | 202 ms | -2 % |
| 32K | **1,580 ms** | 1,694 ms | **-6.7 %** |

That is **0.25 % of the prefill**, not the 2-3x I predicted from the chain analysis.  The derivation was wrong:
those three loads were not what the serial chain spends its ~1.19 us/token on.  A GDN phase breakdown at 4K
shows where the phase actually sits - `gdn_rec_cols_kernel` 175.1 ms of 234.9 ms (72 launches, 2.43 ms each),
`gdn_out_norm_kernel` 38.5 ms, `gdn_conv_tiled` 13.6 ms and the rest under 5 ms - so the rec kernel is 88 % of
the phase and the prefetch moved a small part of it.  Kept because it is a real, numerics-preserving
improvement with a clean A/B, but recorded at its true size.

**Refuted: dequant-to-fp16 + WMMA for the *experts*.**  `STRATA_PREFILL_MMQ=0` routes the experts through the
dequantise-then-`gemm.f16` path, which now uses the WMMA GEMM this session added - i.e. the same trick that won
for the dense projections:

| config | 4K prefill | gemm gate/up | gemm down |
| --- | ---: | ---: | ---: |
| MMQ (default) | **5,989-6,019 ms (680-684 tok/s)** | 784-803 ms | 383-410 ms |
| `STRATA_PREFILL_MMQ=0` | 8,379-9,157 ms (447-489) | 3,231-3,714 ms | 1,699-2,064 ms |

**~41 % slower**, with the GEMM phases 4-5x larger - a structural signature, not contention.  The reason is the
one difference between the dense projections and the experts: the dense weights are dequantised once per chunk
and reused across all T tokens (a 1:T reuse ratio, which is why WMMA wins there), whereas each expert block is
streamed and used ~once, so expanding it to fp16 multiplies the bytes it must move by ~3.5x and the GEMM becomes
bandwidth-bound.  The quantised dp4a path is the right one for streamed experts.

**A measurement trap worth recording:** my in-loop "build started" guard used `pgrep -af clang`, which matches
its own wrapper because the wrapper's command line contains that string - the AGENTS.md section 3 trap, this time
producing false alarms rather than a self-kill.  The bracket form `pgrep -af "[c]lang"` does not, and the
confirmation run above uses it.
