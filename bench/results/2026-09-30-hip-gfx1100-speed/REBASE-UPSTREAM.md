# Can our branch be rebased onto upstream's merged gfx1100 backend?

**Answer: yes, and the work is bounded and now enumerated.**  Measured on 2026-09-30 against `origin/main`
(`bbaaabb`, engine 0.1.28, latest commit that evening).  Our `hip-gfx1100` is 83 commits ahead / 46 behind, on an
older base: the merge-base is `cd97dafa` ("Speed tables: prompts with engine 0.1.22"), so upstream is 0.1.28 and
we are on 0.1.22 - a six-release gap that a rebase would pick up.

## The shape of the divergence

Most of our 83 commits are the bench record, not code:

| | commits | files | lines |
| --- | ---: | ---: | ---: |
| ours since the base | 83 (22 code) | 58 | +6,590 / -88 |
| theirs since the base | 46 | 76 | +7,638 / -405 |
| **ours, code only** | **22** | **26** | **+1,964 / -88** |

Our code footprint is small and concentrated: `wmma_gemm.{cu,h}` (new, 302 lines), `qsa_prompt_attn.cu` (+377),
`verify.cpp` (+236), `prefill/kernels.cu` (+103), `moe_mmq.cu` (+80), `prefill.cpp` (+82), `fused_gr.cu` (+82),
`expert_cache.cpp` (+31), `pinned.cu` (+31), `gemm.cu` (+26), `verify_kernels.cu` (+39), `elementwise.cu` (+16),
`ggml_cuda_host.cu` (+16), `compat/hip/*` (our CUDA-to-HIP shim, 257 lines), `tools/run-tuned.sh` (new).

## Textual merge cost: 11 files, 18 hunks

A merge in a throwaway worktree (`/tmp/strata-rebase`, branch `rebase-test`) resolves in one shot what a rebase
would hit repeatedly:

* **35 files auto-merge**, including our new files (`wmma_gemm.*`, `run-tuned.sh`) and - notably - `CMakeLists.txt`,
  `src/prefill/gemm.cu` and `src/prefill/moe_mmq.cu`.
* **11 files conflict, 18 hunks total**: `prefill.cpp` (4), `bf16_bits.hpp` (3), `f16_bits.hpp` (3), then one each
  in `verify.cpp`, `verify_kernels.cu`, `qsa_prompt_attn.cu`, `fused_gr.cu`, `expert_cache.cpp`, `rope.hpp`,
  `mrope.hpp`, `ggml_cuda_host.cu`.
* **14 of our commits touch those 11 files**, so a plain `git rebase origin/main` would stop about fourteen times,
  re-resolving the same regions.  **Squash our code work into a few logical patches and rebase those, or merge
  once.**  A merge resolves 18 hunks once.

Their edits to the contested files are mostly *small* (`verify.cpp` +18, `verify_kernels.cu` +4, `moe_mmq.cu` +2,
`elementwise.cu` +0), which is why the hunks are few: the bulk of their backend is in **new** files
(`include/strata/hip_compat/*`, `cmake/hip_backend.cmake`, `tests/hip/*`, `tools/hip/*`).

## The real obstacle: two parallel HIP compat layers

Both ports map the CUDA API onto HIP, at different paths, which is why git merged them without complaint:

| | ours | theirs |
| --- | --- | --- |
| path | `compat/hip/{cuda_runtime.h,cublas_v2.h,cuda_fp16.h,math_constants.h}` | `include/strata/hip_compat/{cuda_runtime.h,cuda_runtime_api.h,cublas_v2.h,cuda_fp16.h,intrinsics.hpp}` |
| `__nanosleep` | `strata_hip_nanosleep` counting down from `(ns+31)/32` (our deadlock fix) | **not defined at all** - they pace waits differently |

The merged tree **configures cleanly** (`CFG_EXIT=0`, hipcc found, both backends' targets generated) and then
**fails to build with 17 errors, all from the two layers being in force at once** in `src/core/device.cu:4`:

```
compat/hip/cuda_runtime.h:193: warning: '__nanosleep' macro redefined
hip_compat/intrinsics.hpp:123: note: previous definition is here
compat/hip/cuda_runtime.h:199: error: a type specifier is required for all declarations
hip_compat/intrinsics.hpp:114: note: expanded from macro '__vsubss4'
```
plus `redefinition of cudaGraphInstantiate` (x2) and `cudaFuncSetAttribute`.  Their `intrinsics.hpp` defines
`__vsubss4`/`__vsub4` as **macros**; our shim defines them as **inline functions** - so our declaration is
macro-expanded into nonsense.  The same primitives, two incompatible forms.

**One decision resolves this**: drop one shim and point the build at the other.  Keep **theirs** - it is the base
we would be rebasing onto, and their docs record it passing 29/29 CTests including asynchronous handoff, QSA, MMQ,
Lt GEMM, KV streaming and PLE reading.

## What our shim would lose, and why it is probably nothing

Worth checking before deleting it, and the checks came back favourable:

* `__nanosleep` - their layer does not define it, so our underflow fix has no counterpart to protect.  (The bug
  was ours, in our shim; they never had it.)
* `__vsubss4` / `__vsub4` - theirs has both, as macros over `__builtin_elementwise_sub_sat` and bit tricks.
* `__fmul_rn` / `__fadd_rn` contraction - their performance doc records the same finding independently
  ("HIP's `__fmul_rn` can become ordinary multiplication; disabling contraction in `quantize_act.cu` preserves the
  separately rounded product").

So the shim is very likely fully superseded, which is what makes this a bounded port rather than a ground-up merge.

## Recommended path

1. Base on **their** backend (0.1.28 + their HIP layer), not ours.
2. Re-apply our **value-add** as a handful of patches, most of which already apply cleanly:
   * the WMMA GEMM (`wmma_gemm.{cu,h}`, new - clean) plus its dispatch in `gemm.cu` (auto-merged, but their
     `gemm.cu` gained +279 lines of hipBLASLt path, so the **dispatch logic needs a semantic review**, not just a
     text merge);
   * the WMMA prompt attention in `qsa_prompt_attn.cu` (their version is an ordered FP32 fallback, +112 lines;
     ours is +377 - keep ours, and this is the change our pp/decode advantage is built on);
   * the prefill instrumentation and GDN sub-phases (`prefill.cpp`, `kernels.cu`);
   * the MMQ gather batching (`moe_mmq.cu`, auto-merged, theirs +2);
   * `tools/run-tuned.sh` and the bench record (clean).
3. Expect to **re-run the upstream-vs-ours comparison afterwards** (`bench/tools/vs_upstream.py`), because the
   point of rebasing is to combine their engine work with our kernels, and the interaction is not predictable from
   the diff.

## Caveats

The merge test resolved all 11 conflicts by taking **ours** purely to get a build; that is a diagnostic choice,
not the recommended resolution.  The build failure it produced is therefore *expected* and is evidence about the
compat layers, not about the 18 hunks.  Nothing here was validated on a working merged binary - the next round's
job is to produce one and measure it.

## Merge attempt: it builds, but it does not hold our performance yet

Following the recommendation above, the merge was carried out in the worktree `/tmp/strata-rebase` on branch
`rebase-test` (our `hip-gfx1100` was never touched), with the per-file strategy from the section above, and then
built and gated.  Commit **`dd51c89`** ("merge origin/main (upstream gfx1100 backend) with our kernels"), 0 compile
errors, binary `build-merge/strata-hip` (25.9 MB).

Verified independently rather than taken from the implementation summary:

| claim | evidence |
| --- | --- |
| single compat layer | `compat/hip` appears only in a stale comment in `CMakeLists.txt`; the build links upstream's `strata_hip_runtime` |
| our WMMA dispatch survives | `src/prefill/gemm.cu` calls `strata_wmma_gemm_bf16` / `_f16` *before* `try_hipblaslt` (lines ~356/384 vs ~362/390) |
| WMMA not silently stubbed | `STRATA_WMMA_GFX11=1` present 5x in `build-merge/CMakeFiles/{strata_kernels_hip,strata_prefill_hip}.dir/flags.make` |
| the nanosleep fix survived | countdown loop ported into `include/strata/hip_compat/intrinsics.hpp:110`; `verify_kernels.cu` and `elementwise.cu` still call `__nanosleep` |

One behavioural difference worth flagging: the ported helper sleeps `__builtin_amdgcn_s_sleep(1)` per iteration
where our shim slept `s_sleep(8)` - so the same number of iterations now pace 8x tighter, which changes how hard
the doorbell waits spin.

**The gate fails.**  `bench/tools/gate_merged.py` runs the pre-merge and merged binaries on identical explicit
flags in the same session, interleaved (2 reps; a hang counts as failure and there were none):

| prompt | arm | pp median (range) | tg median (range) |
| --- | --- | ---: | ---: |
| 1K | old | **582.2** (581-583) | 48.4 (48.3-48.5) |
| 1K | merged | 530.9 (456-606) | 45.9 (45.1-46.6) |
| 4K | old | **935.8** (933-938) | 47.6 (46.6-48.6) |
| 4K | merged | **825.0** (797-853) | 51.2 (48.7-53.7) |

Against our recorded same-session numbers the merged build is -8.0 % pp / -3.6 % tg at 1K and -10.6 % pp /
+3.4 % tg at 4K.  The 4K prefill deficit is the solid finding: both merged runs (797, 853) sit below both old runs
(933, 938) with no overlap.  The 1K merged arm is bimodal (606 then 456), so its median is weak evidence.

**So the merge is mechanically successful and performance-incomplete.**  A merged build that carries our WMMA GEMM
and attention *and* upstream's 46 commits should be at or above parity; being ~10 % under it on prefill means
something in the combination is worse than either side, and that has to be attributed before adopting it.
Candidates, in the order they are cheapest to test:

1. **The hipBLASLt fallback.**  Our tree had no hipBLASLt path at all; the merged `gemm.cu` falls through to
   `try_hipblaslt` whenever our WMMA declines a shape.  Their calibration table does *not* load here
   (`file=100100 runtime=100401`), so those fallbacks run on an uncalibrated path that our pre-merge binary never
   touched.  Test: `STRATA_WMMA_GEMM`/shape probes, or build with the hipBLASLt path disabled.
2. **The nanosleep pacing** above (`s_sleep(1)` vs `s_sleep(8)`).
3. **Upstream's prefill changes** (+267 lines in `prefill.cpp`, plus their staging/chunking), against which our
   instrumentation was re-applied - our `--prefill 2048` may no longer mean what it did.
