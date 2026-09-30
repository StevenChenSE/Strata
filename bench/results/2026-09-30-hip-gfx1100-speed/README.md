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
