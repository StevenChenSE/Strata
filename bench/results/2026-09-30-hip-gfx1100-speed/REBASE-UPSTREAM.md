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

## The gap, addressed: prefill was drift, and a real tg difference replaces it

Chasing the merged build's -8.8 %/-11.8 % prefill deficit found the opposite.  A phase-table pair (same flags, 4K)
had the merged build *faster*: 4,185.7 ms (978.1 tok/s) against 5,000.5 ms (818.7), and the composition showed
where it came from - the merged build carries upstream's 0.1.25 prompt work:

| phase (4K) | old | merged |
| --- | ---: | ---: |
| gdn (fused hyper-connections) | 859 ms | **591 ms** |
| gemm gate/up | 783 ms | **613 ms** |
| wait copy | 758 | 626 |
| dequant | 334 | 292 |
| qsa attn | 181 | 159 |

A proper interleaved gate (`bench/tools/gate_merged.py 3`) then **passes**:

| prompt | arm | pp median (range) | tg median (range) |
| --- | --- | ---: | ---: |
| 1K | old | 582.3 (570-595) | 48.4 (46.0-48.8) |
| 1K | **merged** | **602.6** (464-614) | 46.0 (44.8-46.6) |
| 4K | old | 823.4 (704-918) | 48.1 (44.5-48.8) |
| 4K | **merged** | **971.6** (952-975) | 46.1 (42.1-47.2) |

So **the identified gap does not exist**: it was host drift, and prefill is at parity or better (+4 % to +18 %
depending on the run; a later 3-arm confirmation put it at +4.4 %, 971.2 against 929.9).  The 2-rep gate that
reported -11.8 % was simply under-sampled on a box with this much drift.

**What is real is the other axis: tg is 5-8 % lower**, reproducibly - the 3-rep gate (-6.9 %), the 3-arm
confirmation (-6.5 %), and the flag isolation (-7 %) all agree.  The mechanism is in the draft accounting:

| | old | merged |
| --- | ---: | ---: |
| rounds | 5 of 6 | 7 of 6 |
| drafts accepted | 4 of 4 (1.000) | 2 of 2 (1.000) |
| **tokens per round** | **1.80** | **1.29** |

Both accept every draft they propose, but the merged engine **proposes far fewer per round** - it stops drafting
earlier.  That is worth ~28 % of tokens/round, partly offset by its cheaper CPU pool (14.3 vs 21.5 ms/round,
because it routes more experts over PCIe and is ring-limited instead), for a net -7 % on tg.

**Two flag hypotheses tested and rejected**, both recorded so they are not re-tried:

* `--pcie-frac` does not close it.  A 2-rep sweep looked like it did (0.15 giving tg 50.3), but an isolated
  3-rep test at fixed `--adapt-every 2` put 0.05/0.15/0.30 all within noise (43.6/44.7/43.9) - the sweep was
  under-sampled, which is exactly the trap this box sets.
* `--adapt-every 0` (upstream's documented setting) makes tg much *worse* here: 34.1 tok/s, because without the
  adaptive cache swaps the CPU pool grows to 36.3 ms/round and becomes the bottleneck.  Their setting suits their
  mmap-based configuration, not ours.

**One hypothesis of mine was wrong and is corrected here**: the merge replaced `data/draft_vocab.bin`
(162,100 -> 425,196 bytes, with `draft_vocab_en.bin` added at exactly our old 162,100), which looked like the
cause.  It is not: `src/core/mtp.cpp` reads the vocab from the **MTP runtime directory**
(`<rt_dir>/draft_vocab.bin`), not from `data/`, so both binaries read the same file and the repo copy is
irrelevant to a run.

**Next lead for the tg difference**: since acceptance is 1.000 in both and only the proposal count differs, the
draft-stop threshold is the place to look - `--spec-min-p` (0.5 here) and any changed stopping rule in their
`mtp.cpp` (the merge added 21 lines there).  A one-flag test at a lower `--spec-min-p` should say whether the
drafting recovers; if it does, the deficit is a configuration difference rather than a regression.

## Pursuing the tg difference: no fixable deficit - it is parity with wider variance

The -5 to -8 % tg was pursued through every configuration lever that could plausibly matter.  None of them changes
it, and the best case turns out to be parity.

**Corrected decode accounting.**  The earlier "1.80 vs 1.29 tokens/round" came from `--max-new 8` runs, i.e. 5-7
rounds - far too small a sample, and it is withdrawn.  Re-derived from the 128-token gate logs, which are the same
session and interleaved (medians of 3):

| 4K, 128 tokens | rounds | acceptance | tokens/round | wait for rings | CPU pool | per-round | ms/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old | 55 | 0.745 | 2.38 | 26.2 | 16.8 | 43.0 | 18.1 |
| merged | 60 | **0.773** | 2.13 | 27.3 | 15.0 | 42.3 | 19.9 |

Acceptance is equal (merged marginally higher), the per-round cost is equal, and merged's drafting is *cheaper* -
0.875 against 1.758 ms/round, with the MTP prompt cost down from 155.1 ms to 11.8 ms.  Only tokens/round differs,
by ~10 %, with fully overlapping ranges.

**Levers tested and rejected**, all on the merged build at 4K/128 tokens, interleaved, order rotated per repetition
(the rotation matters: the first sweep of this session ran a fixed order and showed a spurious pp anomaly):

| lever | result |
| --- | --- |
| `--spec-min-p` 0.0 / 0.2 / 0.5 / 0.8 | lower thresholds *do* raise tokens/round (2.71 at 0.2, 2.67 at 0.0, against 2.11 at 0.5) but tg gets **worse** (42.5 / 42.0 against 43.7) - the extra drafts are rejected, so the per-round cost wins.  0.5 stays. |
| `--pcie-frac` 0.05 / 0.15 / 0.30 | all within noise (43.6 / 44.7 / 43.9).  A 2-rep sweep had suggested 0.15 at 50.3; that was under-sampling. |
| `--adapt-every 0` (upstream's documented setting) | much worse: 34.1 tok/s, the CPU pool grows to 36.3 ms/round and becomes the bottleneck. |
| cache shrink from their 0.1.28 draft-head reservation | 46 slots (9,293 -> 9,247; 17.63 -> 17.54 GiB).  Too small to matter. |
| `--spec` 2 / 3 / 4 / 6 | **4 remains optimal at 48.0 tok/s**, equal to the old binary's 48.1 - and 953 pp against 823. |

**Conclusion.**  There is no systematic tg deficit to fix.  Across ~6 independent measurements the old binary is
consistently ~47.4-48.4 while the merged build ranges 43.6-48.0, so the merged engine's *median* is a few per cent
lower but its spread is much wider; in its best case it is exactly at parity, and its optimum `--spec` is
unchanged.  What the merge certainly brings is a large prefill win (+4 to +18 %, and 953 against 823 pp in the
`--spec 4` run), a 13x cheaper MTP prompt (155.1 -> 11.8 ms), and upstream's six releases of engine work.

**Recommendation.**  Adopt the merge: it is at least at par on decode and substantially better on prefill.  The
one reservation is *consistency*, not speed - the pre-merge binary's decode is tighter (47.4-48.4 against
43.6-48.0), which matters for latency-sensitive serving and is the thread to pull if the merged build becomes the
default.  The worktree branch `rebase-test` (`dd51c89`) is where it lives.

## Does the merged build consistently beat upstream master?  No - it is a lean, not an advantage

Adoption question asked directly: if the merge is taken, is it reliably better than simply using upstream's own
master?  Measured interleaved with the order **alternating** per repetition (merged, master / master, merged /
merged, master), each arm on its own documented configuration, 128 decoded tokens:

| prompt | arm | pp median (range) | tg median (range) |
| --- | --- | ---: | ---: |
| 1K | merged | 487.7 (353-555) | 44.0 (41.5-46.6) |
| 1K | master | 472.5 (468-474) | 44.3 (44.1-45.5) |
| 4K | merged | 783.5 (768-1087) | 46.7 (41.2-57.5) |
| 4K | master | 715.9 (669-751) | 42.4 (41.6-44.0) |

Per-pair winners: merged takes prefill **2/3 pairs at 1K and 3/3 at 4K** (+3.2 % and +9.4 % on the medians) and
decode **1/3 and 2/3** (-0.7 % and +10.2 %).  So prefill leans merged, decode is a coin flip, and neither is a
sweep.

Two caveats decide how much to read into it:

* **These absolute numbers are far below the earlier session** (merged 4K 783.5 here against 971.6 then; master
  715.9 against 758).  The box was in a slow phase for this run, which compresses the deltas and inflates the
  noise - so the practical statement is that the *difference* is small and partly inside the measurement floor.
* **The noise attaches to different arms in different sessions**, which means it is environmental rather than
  build-specific: in the earlier gate it was the pre-merge binary that had the wide 4K range (704-918 against
  merged's tight 952-975), and here it is merged that is wide (768-1087 against master's tight 669-751).  No
  build is intrinsically the noisy one by this evidence.

**This revises the earlier recommendation.**  The "+22-27 % prefill / +10-18 % decode" against master was measured
without alternating the order and is likely optimistic; with alternating order the merged build's edge over master
is +3 to +9 % on prefill and parity on decode, both partly inside the noise.  A three-way comparison
(ours / master / merged) with rotated order and at least five reps per tier is what an adoption decision needs,
and it has not been run.

What *is* consistent, because it is deterministic and not a timing measurement: the merged build's MTP prompt cost
is 11.8 ms against 155.1 ms, its drafting is 0.875 against 1.758 ms/round, its expert cache is 46 slots smaller
(upstream's deliberate draft-head reservation), and it carries upstream's six releases of engine work - the k8v4
KV option, the hipBLASLt path (inert on this ROCm), the setup/server changes and their HIP test suite.  Those are
categorical gains; the throughput edge over master is not yet one.

### Why the hipBLASLt table is inert here, exactly

`create_hipblaslt_state()` in their `src/prefill/gemm.cu:121-156` decides all of it:

```c
const char* path = std::getenv("STRATA_HIPBLASLT_TUNING");
if (!path || !*path) return nullptr;                        // no table -> no hipBLASLt path at all
...
if (!state->table.load(path, arch, version, error)) {
    std::fprintf(stderr, "prefill gemm: %s; using hipBLASEx\n", error.c_str());
    return nullptr;                                         // table rejected -> whole path abandoned
}
std::fprintf(stderr, "prefill gemm: hipBLASLt tuning enabled (%zu rows, %s, version %d)\n", ...);
```

Three consequences worth being precise about:

1. **The tuned table is the only way into the hipBLASLt path.**  There is no untuned or heuristic hipBLASLt
   mode: if `STRATA_HIPBLASLT_TUNING` is unset, or the table fails to load, `nullptr` is returned and the dense
   prefill GEMM runs through `hipblasGemmEx` - what their own messages call "hipBLASEx" (plain hipBLAS).  So on
   this box **hipBLASLt is not used at all**, not merely untuned.
2. **The version check is deliberate and correct.**  The shipped tables declare
   `STRATA_HIPBLASLT_TUNING_V1 gfx1100 100100` (31 rows) and `... 100200` (27 rows), and their own header comment
   says "solution IDs are scoped to this hipBLASLt version and device architecture".  A solution ID names a
   specific kernel and workspace configuration inside one library build, so replaying a 1.1.0 table against 1.4.1
   can select a kernel that no longer exists in that form.  Our runtime reports **100401** (= 1.4.1, matching
   `libhipblaslt.so.1.4` and the header's MAJOR 1 / MINOR 4), so **neither shipped table matches** - switching to
   the 100200 file would not help either.
3. **It handicaps their arm, not ours.**  Per the docs their published benchmark *did* use the table, so their
   numbers are not what was measured here; and because their master has no WMMA path while the merged build calls
   our WMMA kernels first, the missing hipBLASLt path costs their master far more than it costs the merged or our
   builds.  My "ours/merged is faster" comparisons are therefore conservative toward them.

To make it engage one would either use a ROCm whose hipBLASLt is 1.1.0/1.2.0 - which is what their
`setup.sh --backend hip` arranges by installing a pinned ROCm from TheRock wheels, ~10 GB, no sudo - or re-tune on
1.4.1 with their own `tools/hip/tune_hipblaslt.cpp` (419 lines, vendored here) and write a 100401 table.

## Three-way at 32K: our branch wins outright, and the merged build's decode is bimodal

The decision-relevant tier measured directly - our branch, upstream master, and the merge, at 32K depth
(32,767-token prompt, `--max-context 36864`, `--kv int8`, 128 decoded tokens), each arm on its own documented
configuration but with `--spec 2` common to all three (the optimum our engine measured at this depth), 3 reps with
the order rotated per repetition:

| arm | pp median (range) | tg median (range) | acceptance |
| --- | ---: | ---: | ---: |
| **ours** | **1141.4** (1012-1205) | **60.7** (54.4-61.2) | **1.000** |
| master | 809.9 (806-840) | 54.9 (51.6-55.2) | 0.836 |
| merged | 1127.4 (920-1176) | 48.4 (46.7-59.0) | 0.806 |

**Prefill**: ours and merged both beat master in **3/3 pairs each** - ours +41 %, merged +39 % - and are a wash
against each other (1141 against 1127).  **Decode**: ours wins **3/3 pairs against both** (+10.6 % over master,
+25 % over merged's median).

The draft accounting explains the decode column, and is the most interesting result of the whole exercise:

| arm | rounds r1/r2/r3 | acceptance r1/r2/r3 | tokens/round |
| --- | --- | --- | ---: |
| ours | 66 / 66 / 66 | 1.000 / 1.000 / 1.000 | 1.94 (identical) |
| master | 72 / 72 / 72 | 0.836 / 0.836 / 0.836 | 1.78 (identical) |
| merged | 66 / 74 / 75 | 1.000 / 0.806 / 0.794 | 1.94 / 1.73 / 1.72 |

Ours and master are *deterministic* in their drafting - the same rounds and the same acceptances in every
repetition.  The merged build is **bimodal**: one repetition drafted exactly like ours (1.000, giving tg 59), two
behaved like a degraded master (0.79-0.81, giving tg ~48).  That is precisely the wide decode range seen in every
earlier merged measurement, and it explains why the 4K comparisons kept flipping between "parity" and "-7 %": they
were sampling two different modes.

**Recommendation, revised on this evidence.**  For 32K work - the tier this model exists for - **stay on
`hip-gfx1100`**: the merge gains nothing on prefill (a wash), loses ~20 % on decode, and introduces the draft-mode
instability; master alone is 41 % behind on prefill.  The merge's real value is categorical rather than
throughput: the 13x cheaper MTP prompt, six releases of engine work, k8v4, the setup/server changes and their HIP
test suite.  Caveat on this comparison: `--spec 2` was imposed on all three arms, so master was not run at its
documented `--spec 4`; at this depth our own engine prefers 2, but theirs was not swept.

## Why the merge "loses" decode: it is timing-dependent nondeterminism, not a design deficit

Asked directly, because the merge is supposed to be upstream's engine plus our kernels and should therefore be at
least as good as either.  It is - when it computes the same way.  Two mechanical explanations were tested and both
are refuted; what remains is a run-to-run instability in the merged build itself.

**Test 1: the cache size.**  The merged build's cache is 46 slots smaller than ours (upstream's deliberate
draft-head reservation), which changes the resident set and, per this project's AGENTS.md, "different residency
sends an expert down a different (CPU vs GPU) arithmetic path, and the tokens diverge".  Both arms were forced to
the same 9,262 slots:

| arm | slots | tg median | accepted drafts over 3 reps | tokens/round |
| --- | ---: | ---: | --- | ---: |
| ours | 9,262 | **59.6** | 62 / 62 / 62 = 1.000 every time | 1.94 |
| merged | 9,262 | **57.1** | 70 / 62 / 62 = **0.814, then 1.000, 1.000** | 1.94 |

Equal caches did not equalise acceptance, so the 46 slots are not the cause.

**Test 2: the adaptive tier's timing-dependent swaps.**  With adaptation frozen (`--adapt-every 0`) the residency
is fixed, so any remaining flip is in the computation rather than the selection:

| configuration | tg per rep | rounds | acceptance |
| --- | --- | --- | --- |
| merged, `--adapt-every 0` | 48.49 / 38.18 / 39.01 | 74 / 66 / 66 | **0.809 / 1.000 / 1.000** |
| merged, `--adapt-every 2` (control) | 58.03 / 60.51 / … | 66 / 66 | 1.000 / 1.000 |

Frozen residency still flips - and is much slower overall (median 39.0, because without adaptation the CPU pool
does more work).  So the adaptation is not the driver either.

**What the data does say.**  In its good mode the merged build is *identical* to ours: 66 rounds, 1.94 tokens per
round, 62/62 drafts accepted, 58-61 tok/s.  Its deficit is that the same binary with the same flags and the same
input sometimes lands at 0.809 acceptance and ~48 tok/s instead.  With residency frozen, the only thing left that
differs between those repetitions is **timing** - so the merged build carries a timing-dependent nondeterminism in
its own compute path.

The prime suspect is the seam the merge created: our **VRAM-doorbell verify handshake** combined with upstream's
**reusable pinned staging buffer** for expert uploads.  Those are two different solutions to the same problem,
glued together by taking ours for `verify.cpp`/`verify_kernels.cu` and theirs for the staging in
`expert_cache.cpp` - and a race there would occasionally feed the draft head a different or stale hidden state,
which on a hard acceptance threshold shows up exactly as observed.

**Consequence for adoption.**  The merge does not lose decode by construction, so this is a bug to find rather
than a reason to reject it - and if it is fixed the merge would be at least our equal on decode while keeping
upstream's prefill and features.  The specific race is a hypothesis, not yet shown: it would be confirmed by
forcing the synchronisation in the verify path (or by instrumenting what the draft head actually reads) and seeing
the flip disappear.

## The merged build's decode instability, traced to expert placement - and fixed by a flag

Kept digging with `STRATA_VERIFY_DEBUG=1` traces (identical line counts, 11,302, so only *values* differ) and the
existing logs.  The first real difference between a good and a bad repetition is the **prefill line itself**:

```
good: prefill 32766 tokens in 16 chunks, 26033.1 ms (1258.6 tok/s); streamed 247,628; resident  32,956; PLE 237.3 ms
bad:  prefill 32766 tokens in 16 chunks, 31464.7 ms (1041.4 tok/s); streamed 258,911; resident 127,319; PLE 1006.0 ms
```

**The host-side variability is shared, so it is not the discriminator.**  Our own build's residency swings just as
widely (40,685 -> 106,344) and its PLE phase ranges 235 -> 1,164 ms, yet it emitted all-zeros with 1.000 acceptance
in all six reps examined.  Master is even tighter: streamed/resident are byte-identical in every repetition
(69,900 / 28,168 at 32K, 16,576 / 7,708 at 4K), and its per-round acceptance histogram is identical too
(`0:19 1:51 2:1 3:1`), because `--pcie-frac 0` removes the PCIe-share variability from the split.

**What differs is what the counters mean.**  Our own source says it, at `src/prefill/prefill.cpp:335`:

> a streamed run lends the prompt path exactly the slots a resident one does (**a lent expert runs on the CPU,
> which rounds differently**: without this an A/B compares two expert placements as well as two KV placements)

So any difference in *which* experts are served from the cache versus lent to the CPU changes the rounding, hence
the logits - and because draft acceptance is a hard threshold, the visible effect is a flip between 1.000 and
~0.81 acceptance with different emitted tokens.  The merged build showed two *tight, repeatable* prefill modes
(`resident 32,956` vs `127,319 / 127,321`), i.e. two expert placements, and its answers flipped with the mode.
Our build varies continuously but stayed on placements that do not cross the token boundary; master never varies
at all.

**The fix is a flag, not a code change.**  Running the merged build with `--pcie-frac 0` (as master does) removes
the variability:

| merged, `--pcie-frac 0 --adapt-every 2` | output | acceptance | tok/round | resident | tg |
| --- | --- | ---: | ---: | ---: | ---: |
| rep 1 | all-0 | 1.000 | 1.94 | 81,153 | 54.95 |
| rep 2 | all-0 | 1.000 | 1.94 | 81,163 | 54.87 |
| rep 3 | all-0 | 1.000 | 1.94 | 89,537 | 60.04 |

Three identical answers, 1.000 acceptance every time, and **54.9-60.0 tok/s, i.e. parity with our branch's
59.6-61.8** - so the merged build is not slower on decode once its expert placement is made deterministic; it was
flipping.  (`--adapt-every 0` is also deterministic but slow, 36-39 tok/s, because a frozen placement pushes more
work onto the CPU.)

This changes the adoption picture: the merge needs no bug fix for decode, only `--pcie-frac 0` in its
configuration - at the cost of whatever prefill the PCIe share was buying, which has *not* been measured for the
merged build at 32K and is the next thing to check.

## Does `--pcie-frac 0` buy better tg and pp?  No - it costs on both, and it only masks the flip

The previous section presented `--pcie-frac 0` as the fix for the merged build's decode instability.  Measured
properly - three arms, order rotated per repetition, three reps, 32K depth, `--expert-cache 9000 --spec 2
--adapt-every 2 --kv int8`, acceptance 1.000 in every run of every arm:

| arm | pp median (range) | tg median (range) |
| --- | ---: | ---: |
| merged, `--pcie-frac 0` | 1104.9 (1077-1163) | **52.4** (45.6-58.6) |
| merged, `--pcie-frac 0.30` | **1136.6** (1122-1282) | **57.6** (54.7-58.1) |
| ours, `--pcie-frac 0.30` | 1031.9 (1008-1070) | 55.0 (48.9-58.7) |

So `--pcie-frac 0` is **9 % worse on tg and 2.9 % worse on pp** than the 0.30 default on the merged build.  The
phase tables say why, and they also correct an assumption of mine: `--pcie-frac` was documented as governing the
*verify windows*, i.e. decode, but it moves prefill work too:

| phase (merged, one rep) | frac 0 | frac 0.30 |
| --- | ---: | ---: |
| wait copy | 2,468 | 3,903 |
| **dequant** | **2,423** | **621** |
| gemm gate/up | 5,283 | 2,478 |
| gdn | 4,443 | 4,243 |

Removing the PCIe share does not remove the work, it moves it: fewer copy-engine stalls, but far more host-side
dequantisation - a near-wash on prefill throughput and a real cost in decode, where the CPU must now compute every
miss.

**Correction to the earlier framing.**  In this run the `--pcie-frac 0.30` arm did **not** flip: three reps, all
1.000 acceptance.  So the flip is an *intermittent* timing effect rather than something 0.30 causes, and
`--pcie-frac 0` merely removes the PCIe path's timing dependence at a genuine performance cost.  Note also that
frac 0 stabilised the *answers* here but not the *speed* (45.6-58.6 tok/s): answers follow expert placement, which
frac 0 pins, while speed follows host conditions (PLE, arena load), which it does not.

**And the comparison sharpens the case for fixing the real thing.**  In this same session the merged build with
`--pcie-frac 0.30` was the **fastest of the three on both axes** - prefill 1,136.6 against our 1,031.9 (+10 %) and
decode 57.6 against our 55.0 (+4.7 %).  So when it does not flip it is the best build available; the remaining work
is to remove the intermittent placement nondeterminism in code, not to work around it with a flag that costs 9 %.

## The merge landed on hip-gfx1100

Merged as commit **2a59383** - zero conflicts.  It takes 47 commits from the merged branch (upstream's backend plus
the resolution in `dd51c89`) while keeping this branch's 14 bench commits.  The pre-merge binary is preserved at
`/tmp/strata-hip-premarge` (24 MB), and `build-hip/` was reconfigured for the merged sources and rebuilt: 0 errors,
and the binary's version string is now **0.1.28** where it was 0.1.23.

Post-merge check on the canonical path - both binaries at the same flags (`--expert-cache auto --prefill 2048
--spec 2 --pcie-frac 0.30 --adapt-every 2 --kv int8`), interleaved, 2 reps:

| tier | merged (0.1.28) | pre-merge (0.1.23) |
| --- | ---: | ---: |
| 32K | **pp 1153.2, tg 56.0** | pp 1060.3, tg 54.5 |
| 1K | pp 544.7, tg 41.4 | pp 518.3, tg 44.3 |

At the decision tier the merge is **+8.8 % prefill and +2.8 % decode**.  The 1K tier is noise-dominated at two
reps (pp favours the merge, tg the pre-merge) and should not be read as a result.  Both arms held 1.000 acceptance
throughout - the intermittent placement flip did not appear in this session.

Also updated: `tools/run-tuned.sh`'s header now records that the binary it launches is the merged engine, the two
known issues (intermittent placement nondeterminism, and the hipBLASLt table not engaging on this ROCm), and why
`--pcie-frac 0.30` rather than 0 is the default; `bench/tools/gate_merged.py` now points at the post-merge paths
(`build-hip/strata-hip` for the merged engine, `/tmp/strata-hip-premarge` for the pre-merge one).

## Three follow-ups: vision on HIP, --kv-resident, and the PLE read path (2026-10-01)

### 1. Image support is feasible without porting anything

The vision helper exists in-tree (`tools/vision/strata_vision.cpp` + its CMakeLists) and contains **zero** CUDA
references: it is a thin llama.cpp `mtmd` wrapper that loads the model's mmproj, tokenizes an image, and *writes a
file* - int32 `{0x31455653 'SVE1', n_tokens, nx, ny, n_embd}` followed by float32 rows. The engine's side
(`generate.cpp:3558`, the GENI request path, and the m-RoPE `(t, h, w)` table at `generate.cpp:278/1448`) is
architecture-neutral. The installer builds the helper with `-DSTRATA_VISION_CUDA=ON`, which only sets
`GGML_CUDA=ON` inside the helper's llama.cpp build - with it off (the default) the encoder runs on the CPU.

So the path for this deployment is: build `strata-vision` with `STRATA_VISION_CUDA=OFF` (a CPU encoder is fine for
interactive image rates), later optionally add a `GGML_HIP` option to the helper's CMakeLists for GPU encoding.
The engine needs no changes. **The blocker is an asset, not code**: the only mmproj on this box is the 27B one from
the old llama.cpp setup; there is no vision encoder for Qwen3.8-Flash-Next (and no `<|image_pad|>` visible in the
pack's vocab.json, though the engine's metadata pins `kImagePad = 248056` from `qwen4exp.ple.image_token_id`).
Sourcing a matching mmproj is the remaining step before an end-to-end image test.

### 2. `--kv-resident 32768` helps decode, and is now the service default

`--kv-resident 0` (the default) keeps the *whole* KV in VRAM; `N` keeps only N cells per QSA layer (min 20480) and
streams the rest from pinned host memory. Upstream's measured configuration passes 32768. Measured here, 32K prompt
and 4K prompt, two interleaved repetitions each:

| tier | kv-resident | pp | tg |
| --- | --- | --- | --- |
| 32K | 0 | 1203.6 / 1041.3 | 57.93 / 49.12 |
| 32K | 32768 | 1175.1 / 982.8 | **61.29 / 60.62** |
| 4K | 0 | 921.5 / 929.3 | 43.41 / 44.45 |
| 4K | 32768 | 934.5 / 954.1 | 44.10 / 45.54 |

Decode: +14 % at the median on 32K (61.0 vs 53.5) and +2 % at 4K; prefill unchanged within noise. The mechanism is
visible in the expert cache: the streaming arm reports *more* slots (9245 vs 9217 at 32K, 17.54 vs 17.48 GiB) -
freeing the KV's VRAM gives the cache room, and the hit-rate gain outweighs the PCIe cost of streaming the
non-resident tail. The kv-resident arm was also the tighter pair (61.29/60.62, spread 0.7, against 57.93/49.12,
spread 8.8 - the known host-side variance), though two repetitions is thin evidence for stability. Applied to the
live service: `~/.config/strata/serve.json` now passes `--kv-resident 32768`, and `/metrics` reports
`kv_resident: 32768`.

### 3. Pinning the PLE table in RAM hurts - keep `--ple-io direct`

`--ple-io mmap` (page-cached) against the default `direct` (unbuffered SSD), 32K prompt, two repetitions: mmap lost
on both axes (pp 771.6/918.2 against 1170.2/1053.8; tg 49.32/53.77 against 57.55/58.24). The PLE *phase* itself is
small either way (95-102 ms direct, 100-122 mmap), but the mmap run's whole prefill inflated - GPU timeline 42.4 s
against 27.9 s in the paired run, with `embed+steps` jumping 13x (5042 ms against 372 ms). The mechanism: the PLE
split is 27.5 GB, the box already holds a 47 GiB pinned expert arena plus ~63 GB RSS, so mapping the table into the
page cache invites reclaim pressure that stalls host-side work everywhere else. The direct path reads exactly the
rows it needs and never holds the table. Verdict: the default is already right; "pinning it completely" is not
viable at this table size on this box.

### 3b. PLE addendum: the mmap arm re-tested with a properly warmed cache

The A/B above interleaved the arms without a dedicated warm pass, so mmap's first repetition ran cold. Re-tested
with a full sequential read of the 27.5 GB table first (11 s at 2.51 GiB/s), instrumenting /proc/meminfo, then three
mmap repetitions and two direct:

| arm | pp | tg | PLE phase |
| --- | --- | --- | --- |
| mmap, warmed, 3 reps | 941.1 / 955.0 / 1122.2 | 46.55 / 59.75 / 55.94 | 96-101 ms |
| direct, 2 reps | 1187.9 / 1071.4 | 57.97 / 47.73 | 98-99 ms |

The warming was real but partial and short-lived: page cache grew by only 20.2 GiB (not the full table), and one
engine start - which re-reads 52 GB of model through the same cache beside the 47 GiB pinned arena - evicted most
of it (Cached 54,543 -> 36,700 MiB). The PLE phase is ~100 ms in every run, warm or not; the loss is systemic
residency cost, not read latency. Verdict unchanged: keep --ple-io direct. Prefill is the clear axis (direct's
worst rep beats mmap's median); on decode the two overlap within this host's variance.

### 1b. Vision: enabled and verified end to end on the HIP deployment

Following 1 above, image support is now **live** on the gfx1100 service:

* `strata-vision` built from `tools/vision` against the HIP-capable llama.cpp checkout with `STRATA_VISION_CUDA=OFF`
  (CPU encoder, the default) - 0 compile errors, an 8.4 MB binary. No porting was required, as predicted: the helper
  has no GPU code.
* The missing asset turned out to exist: **`unsloth/Qwen3.8-Flash-Next-GGUF` ships `mmproj-F16.gguf`** (0.84 GiB,
  projector `qwen3vl_merger`, `clip.vision.projection_dim` 2560) - downloaded to
  `/home/jianwei/Strata-data/models/IQ3_S/mmproj-Flash-Next-F16.gguf`. Its projection width matches the text model's
  `qwen4exp.embedding_length` (2560) exactly, and the model's own tokenizer carries the vision tokens
  (`<|vision_start|>` 248053, `<|image_pad|>` 248056 = the engine's kImagePad, `<|vision_end|>` 248054): the model is
  natively multimodal (`Qwen4ExpForConditionalGeneration` with a deepstack vision_config in the HF repo).
* Standalone check: the helper speaks `ENC <image> <out>` and answers `READY 2560`, encoding test PNGs to valid SVE1
  records (9 tokens, 3x3, n_embd 2560).
* Wired into the service: `serve.json` gained a `"vision"` entry (`exe`/`mmproj`/`model` - the server spawns the
  helper as a resident process speaking ENC at startup, `server.py:1723`) and the engine gained `--vision`.
  `/metrics` now reports `images: true`.
* **End-to-end correctness**: chat completions with OpenAI `image_url` parts (local file paths) - a red square
  answers "Red", a blue square answers "Blue". Real multimodal inference through PNG -> strata-vision -> SVE1 ->
  GENI -> m-RoPE -> text.

Costs to know: with the vision entry present the server waits for the encoder's READY at startup - the helper loads
the 52 GB text model (mmap) and runs a 4096-token CPU ViT warmup, so service start went from ~30 s to **~16.5 min**
(the unit's TimeoutStartSec was raised to 1800 accordingly). The encoder is CPU-only; a GGML_HIP build of the helper
is the later speed-up if image latency ever matters.

Correction: the mmproj was not unique to unsloth. The quant's own source repo,
ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF, also ships mmproj-Qwen3.8-Flash-Next-BF16.gguf - now downloaded to
mmproj-Flash-Next-BF16-ISTA.gguf and compared against the unsloth F16 in use: identical projector (qwen3vl_merger),
projection_dim 2560, image 768 / patch 16, 34 GGUF fields - the same tower converted twice, differing only in F16 vs
BF16 storage. The service keeps the verified unsloth F16; the provenance-matched ISTA BF16 is on disk as a
drop-in alternative (one-line mmproj path change + restart).

### 1c. Vision: switched to the provenance-matched mmproj, and startup cut 995 s -> 93 s

The service now uses the ISTA-DASLab BF16 mmproj (the quant's own repo) instead of the unsloth F16 copy, which is
deleted; a completion still verifies and the red/blue discrimination still passes ("Red" / "Blue"). Config:
`~/.config/strata/serve.json` vision.mmproj -> mmproj-Flash-Next-BF16-ISTA.gguf.

The 10-minute vision load was the helper's CPU ViT warmup at the 4096-token cap (`strata_vision.cpp` hardcoded a
2048 px warmup square, which mtmd scales to the cap; the model load itself is vocab-only mmap and costs seconds).
`tools/vision/strata_vision.cpp` now reads `STRATA_VISION_WARM_SIDE` (default 2048, upstream behavior unchanged;
bounds 256-8192) and the service sets 1024 - mtmd's own Qwen-VL minimum. Verified: standalone warmup reports
"1024 image tokens" and the whole helper cycle (model load + warmup + one encode) dropped from >600 s to **47 s**;
service startup dropped **995 s -> 93 s** (10.7x). The smaller warmup still pages the mmproj, grows the work
buffers and spins the thread pool; only the worst-case graph shape is deferred to the first big image, which is a
CPU allocator cost, not the GPU-buffer concern the original warmup comment guards against (that rationale is
CUDA-specific and this helper is CPU-only).

### hipBLASLt tuning on this box: the shipped script works, the table engages, and WMMA still wins

The repo ships `tools/hip/tune_hipblaslt` (built via `cmake --build build-hip --target tune_hipblaslt`). Ran it with
the documented recipe (cases from the shipped 100200 table): a few minutes on the idle GPU, producing
`tools/hip/gfx1100-hipblaslt-100401.txt` - a table matching this ROCm's hipBLASLt 1.4.1 (100401), which the shipped
100100/100200 tables never did (the runtime guard rejected both). The engine log confirms the calibrated path
engages with it.

4K head-to-head (the documented measured config, two interleaved reps per arm; ranking consistent in both pairs):

| arm | prefill (tok/s) | decode (tok/s) |
| --- | ---: | ---: |
| WMMA on (STRATA_WMMA_GEMM=1 + PA) | 1083.7 / 1308.4 | 32.1 / 41.3 |
| calibrated table (WMMA off) | 852.2 / 1176.6 | 38.4 / 40.7 |
| baseline (neither) | 658.6 / 552.8 | 37.3 / 37.3 |

So the calibrated table is worth ~+55-90 % prefill over the hipblasGemmEx baseline - the documented lever is real
here - but the WMMA kernels still beat the calibrated table on this card, and they dispatch first, so the service
keeps WMMA primary. The table is now wired as the service's fallback (it only touches shapes WMMA declines:
T < 16, K not a multiple of 16, other betas). It also settles the ordering question the reviewer raised: on this
card, WMMA-ahead-of-table is the faster arrangement, not just the simpler one.

### The intermittent all-token-0 ("!") output on long real prompts: engine-level, kernel-independent

Investigated after the service emitted repeated "!" in reasoning_content under concurrent load. Findings, each
verified by experiment:

* Reproduces on **real code** (~39K tokens of genuine source), not just pathological prompts; rate ~50-80 % per
  request at that size. Sampling does not escape it (temperature 0 / 0.6 / 0.6+top-k / 1.0 all loop).
* **Not our kernels**: with the WMMA envs removed (stock 0.1.30 arithmetic) the identical request still loops
  (1 of 2 on real code; 2 of 3 earlier). Our kernels: 4 of 5. CLI four-path comparison on a different prompt
  (32k.txt ids) showed no loop and bit-identical tokens across stock/hipBLASLt/WMMA arms - that prompt just does
  not trigger it.
* **Not the serve layer**: feeding the SAME token ids through the engine CLI (--tokens-file) reproduces it
  (one clean run, one all-zeros run) - the earlier "API-only" impression was an artifact of comparing different
  prompt content per path.
* **Not fixed by placement pinning**: --pcie-frac 0 still loops 3 of 4 via the API, and the CLI repro above looped
  with --pcie-frac 0 AND --adapt-every 0. The run-to-run divergence survives those knobs.
* Length threshold observed empirically: 4K diverse prompts never looped; ~15K and ~39K real content does.

Conclusion: an engine-level numerical nondeterminism (the same family the port documents: different residency
sends experts down different arithmetic paths) interacting with specific long prompts, present in stock 0.1.30.
Practical mitigations: serve-layer detect-and-retry on N consecutive identical tokens; shorter prompts.  The root
fix - placement-invariant expert numerics - is upstream engineering. A clean upstream repro exists: stock build,
real code ~39K tokens via --tokens-file, two runs, one emits all zeros.

## The all-token-0 ("!") decode collapse: root-caused to our batched expert gather, fixed engine-side

The intermittent all-"!" output (every generated token = id 0, starting at token 1, usually in reasoning; ~50-80 %
per request on ~39K-token real-code prompts) was chased to root cause by bisection across builds and files:

1. NOT the kernels: the stock 0.1.30 binary is clean 7/7 on the same tokens (interleaved with our merged build
   looping 8/17); our build with the WMMA envs removed still looped. (The earlier "stock also loops" observation
   was the merged binary minus our kernels - the port's engine changes were still running.)
2. NOT the serve layer (CLI reproduces), NOT any engine knob (pcie-frac, adapt-every, no-prefill-borrow,
   ple-inflight, io threads, prefill chunk, kv-resident 0, sampling 0..1.0 all failed to eliminate it).
3. Prefill-side: swapping our three prefill files (prefill.cpp, kernels.cu, moe_mmq.cu) to upstream's versions
   made it 6/6 clean; the GDN dispatch is upstream-equivalent (pipe default routes to upstream's kernel), so the
   suspect set reduced to the gather path.
4. The runtime knob STRATA_MMQ_GATHER=per_expert (upstream's per-expert gather, everything else ours) ran 6/6
   clean - isolating the bug to our port's batched expert gather (gather_native_batch, +80 lines in moe_mmq.cu,
   only in this branch).

Mechanism: the per-expert path copies each streamed expert blob to the GPU immediately and only then records the
ring slot's used event; the batched path holds raw blob_dev pointers for up to 15 further streaming steps and
copies the whole 16-expert group at once, so a streamed blob's staging can be recycled before the batch kernel
reads it. The call site already forces per-expert when !stream_all && ring < MMQ_GROUP - the hazard was known -
but not in the stream_all regime that long prompts run in. Wrong bytes for some experts poison the hidden state:
draft and target then agree on token 0 forever (the smoking-gun stat: LOOP runs show drafts accepted 22/22
(1.000) while clean runs sit at ~0.6), which is why no sampling setting escapes and why the collapse starts at
the first generated token.

Fix (prefill.cpp): the gather DEFAULTS to per-expert now; STRATA_MMQ_GATHER=batch re-enables the batched path
for whoever repairs its lifetime.  The batch measured neutral in time when introduced, so the default flip costs
nothing.

Verification: new default 6/6 clean via CLI; negative control STRATA_MMQ_GATHER=batch still loops 2/3 (the bug
remains reachable exactly through that path); the redeployed service ran the reproducing API request 6/6 clean
(was 50-80 %) with coherent reasoning in every response.

### Adopted upstream PR #322 (fast packed-byte intrinsics, v_perm_b32 + SWAR) - verified then tuned

Cherry-picked bsorensen110's fast-intrinsics commits (7da45db + 01e428a) onto hip-gfx1100, over our existing
strata_hip_nanosleep port (clean cherry-pick; the packed-byte region and the nanosleep addition don't overlap).

Correctness: exhaustive parity test passes on gfx1100 (262144 cases, 1048576 results, all 65536 byte pairs x 4096
selectors, 16 guards); our WMMA parity still 22/22. The intrinsics are header-inline and used by iq_kernels.cu /
native_mmvq.cu - compiled into our decode (v_perm_b32 confirmed in the objects).

The decode A/B turned out to be a two-factor story:

1. The intrinsics' benefit depends on being GPU-dequant-bound. At --adapt-every 0 our branch is CPU-pool-bound
   (verify window: wait-for-rings 30.4 + pool 44.7 ms/r vs upstream 18.8 + 24.8) and the patch gains only ~2%
   (four-way, adapt-0, interleaved: br 42.2 vs brp 42.9; upstream 47.8 vs 51.0).
2. Residency is the real lever: --adapt-every 2 keeps the decode's hot experts resident, moving the work back to
   the GPU where the intrinsics help. Four-way at adapt-2 (interleaved, 1K, 512 tokens): our branch 53.1 -> 65.8
   tok/s with the patch (+24%), against upstream 69.0 / 71.3. And the adapt-0 vs adapt-2 gap on our branch alone
   was huge: 45.4 vs 65.7 tok/s (+45%, three interleaved reps, non-overlapping).

At the SERVICE's actual configuration (spec 2, adapt-every 2, pcie-frac 0.30, int8 KV, 32K prompt, 256 new), a
head-to-head with upstream+patch: our branch pp 737 / tg 54.7 against upstream pp 651 / tg 36.8 - ahead on both,
which is where the service now sits (intrinsics adopted, adapt-every already 2, spec 2).
