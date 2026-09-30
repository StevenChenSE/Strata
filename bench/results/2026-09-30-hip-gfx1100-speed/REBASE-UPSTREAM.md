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
