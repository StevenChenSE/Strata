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
