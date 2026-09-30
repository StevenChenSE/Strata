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
