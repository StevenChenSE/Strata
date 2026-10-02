# EVAL: a HIP (GPU) image encoder for the AMD backend

Question: setup says "the AMD backend has no GPU image encoder yet" (`setup.py:1467-1476`, `hip_vision()`), and
`--vision cpu` is the only image path on a HIP install. Is a HIP encoder possible, and what would it take?
Measured on: RX 7900 XTX 24 GiB (gfx1100, `rocm-smi --showproductname`), Ryzen 5 9600X (6C/12T, `powersave`
governor), IQ3_S pack + native GGUF, `strata.service` running. 2026-10-02.

**Verdict: yes, and it is a build-option gap, not an engine gap.** No Strata kernel is involved - the encoder is
llama.cpp's `mtmd` graph, and `ggml-hip` *is* `ggml-cuda` (hipified at configure time). The ViT's whole op set is
already implemented there, including the vision-M-RoPE kernel. The work is ~15 lines of CMake, ~20 lines of
`setup.py`, and one correctness A/B. The thing that actually needs thinking about is **VRAM**, not ops.

**Built and tested the same day (2026-10-02, results in section 7).** The service now runs the HIP encoder.

---

## 7. Results: built, measured, in service (2026-10-02 evening)

Build: `STRATA_VISION_HIP=ON` compiles clean - 278 targets in about 4 minutes (`-j 12`), not the tens of minutes
guessed in section 4. First run against a full card failed exactly as section 3 predicted:
`cudaMalloc failed: out of memory` asking **865.48 MiB** for the weights with 541 MB free.

With the service stopped (24.2 GB free), same 1024x768 picture, same binary (`~/Documents/llama.cpp` @ `4da633776`
with ggml-hip, gfx1100):

| path | encode (768 tok) | same photo at 300 tok |
|---|---:|---:|
| GPU (HIP) | **235 / 230 ms** (two runs) | **60 ms** |
| CPU, same binary, 6 threads | 25.8 s | 4.45 s |

That is ~110x, and the end-to-end request (image + "describe this in two sentences", 400 generated tokens) took
**5.2 s** through the server. The model read the picture correctly down to the fine print (the NYT
"MEN WALK ON MOON" front page: masthead, slogan box, dateline, both subheads).

Correctness A/B, same image, GPU vs CPU embeddings (768 rows x 2560 f32):

* CPU is bit-deterministic here: old 4-thread binary vs new 6-thread binary, max abs diff **0.0000**.
* GPU vs CPU: **mean cosine 0.9906, min 0.5647** on the synthetic noise picture; on the natural photo
  **mean 0.9946, min 0.792**. 626 of 768 rows above 0.99.
* `--flash-attn off` does **not** change it (mean 0.9883, min 0.5635) - not the FA path. The worst rows' norms
  differ too (row 254: GPU 2.13 vs CPU 1.30), so it is the BF16 matmul path (WMMA/hipblas accumulation), not a
  kernel bug I can name. Downgraded to **open**: is this the same on NVIDIA? Nobody has run that A/B here.
  Functionally it passes: the end-to-end answers are correct.

Cost in service (measured, this startup): `expert cache auto` now fills **8280 slots / 15.70 GiB** vs
8955 / 16.98 GiB on the CPU-encoder runs - the encoder took **~675 slots ≈ 1.28 GB**, matching the estimate.
The card sits at 25.24 / 25.75 GB used with everything up - tight, and the vision-restart caveat in section 3
stands: with the engine loaded, a restarted encoder cannot allocate.

Wired into setup: `--vision yes|gpu` on AMD now builds this (hip_vision / build_vision_hip in setup.py), stamped in
engine/BUILD.json (`vision`, `vision_archs`) so an arch or source change rebuilds. `tools/test_setup_choices.py`
17/17 (two tests added for the gpu flavor); `tools/test_setup_amd.py` has one failure,
`test_prebuilt_hip_zip`, which fails on the clean tree too (pre-existing).

### Text tg with the encoder resident (A/B, 2026-10-02 evening)

Identical text-only requests (2269-token prompt, 384 max tokens, temperature 0, streaming), 6 warm samples per arm,
one restart between arms, server = `strata.service`:

| arm | expert cache | warm tok/s (stream meter) | mean |
|---|---|---|---:|
| GPU vision encoder resident | 8280 slots / 15.70 GiB | 70.9 72.4 69.5 76.9 79.7 75.2 | **74.1** |
| CPU encoder (no VRAM held) | 8955 slots / 16.98 GiB | 71.6 81.5 75.0 79.2 76.1 81.7 | **77.5** |

Difference +3.4 tok/s (+4.6%) for the CPU arm, 95% CI +- 4.4 - **not resolvable**: run-to-run spread is +-4-5%
and the ranges overlap. Cache hit rates were the same in both arms (~93% warm). So the 675 lost slots cost
nothing measurable here, consistent with them being the coldest of the profile-ranked slots. First request after
a restart is always cold (54-63 tok/s, hit ~82%) in either arm - that is the restart, not the encoder.

### Text tg, second design: diverse topics and first-request-after-restart

The warm-identical design above has a blind spot the review rightly flagged: repeating one prompt keeps hitting the
same warmed experts, so steady state hides a smaller cache. Rerun with 13,357-token prompts on six different trades
(maritime, bakery, astronomy, tannery, orchard, printing - each request routes a largely fresh expert population),
metered by the server's own `done:` line (decode-only denominator):

| arm | cold first 13k | diverse 13k, per-request tok/s | mean (n=5) | hit range |
|---|---|---|---:|---|
| GPU vision (8280 slots) | 66.9 (hit 85.9%) | 77.5 66.1 71.3 81.2 78.7 | **74.9** | 88.5-90.4% |
| CPU encoder (8955 slots) | 71.3 (hit 87.3%) | 79.8 69.8 80.0 81.9 76.7 | **77.6** | 90.2-91.5% |

And the sharpest cache-pressure case - one ~60k-token diverse prompt as the **first** request after a restart,
where the working set cannot fit either cache:

| arm | 60k cold tok/s | hit |
|---|---:|---:|
| GPU vision | 60.6 | 86.0% |
| CPU encoder | 61.2 | 84.5% |

Reading: the direction is consistent - the GPU-vision arm is a little slower in three of four comparisons
(-6.2% cold 13k, -3.5% diverse mean, -1.0% 60k cold), and its hit rate sits ~1.4 points lower on the diverse
workload. But every difference is inside the +-4-5% run-to-run noise (5-6 samples per arm gives a CI of about
+-7 tok/s), and the 60k cold case - the one where a smaller cache should hurt most - shows none. The magnitude
agrees with the mechanism: 675 dropped slots are the coldest of the profile rank, costing ~1.4 hit points, which
the observational slope (+5.8 tok/s per 10 points) prices at under 1 tok/s. **Conclusion stands, now under the
harder design: a real cost of at most a few percent on cold or cache-pressured text, against 110x faster image
encoding.** Resolving a true 3% difference would take ~20 samples per arm; not worth the restarts for a bound
this size.

### Open items

* The NVIDIA-side embedding A/B (is min-cos 0.79 an RDNA3 artifact or a ggml CUDA-family property?).
* A tolerable story for the vision restart under a loaded card (today: don't turn on `--idle-unload` with images).
* The server still discards the encoder's `ms` (`serve/server.py:599`) - encode time stays invisible in the Monitor.
* `docs/AMD_HIP.md` and `docs/DETAILS.md` still say the AMD backend has no GPU image encoder.

---

## 1. What the code says (read, with lines)

| claim | evidence |
|---|---|
| The encoder picks a GPU backend by *type*, not by vendor | `build-hip/_deps/strata_llamacpp-src/tools/mtmd/clip.cpp:184-197` - `use_gpu` → `ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)`, then `…_IGPU`. No `GGML_USE_CUDA` / `#ifdef` anywhere in `clip.cpp`. |
| GPU is the *default* for mtmd | `tools/mtmd/mtmd.cpp:460` - `/* use_gpu */ true` in `mtmd_context_params_default()`. Strata's helper turns it off unless `--gpu` (`tools/vision/strata_vision.cpp:64,98`). |
| `ggml-hip` is `ggml-cuda`, hipified | `ggml/src/ggml-hip/CMakeLists.txt:60-73` globs `../ggml-cuda/*.cu`, the `fattn-tile*`, `fattn-mma*`, `mmq*`, `mmf*` template instances and `ggml_cuda_fattn_vec_instances`, then `:82-83` adds `GGML_USE_CUDA` publicly. So CUDA op coverage == HIP op coverage. |
| The ViT's ops are covered | `tools/mtmd/models/qwen3vl.cpp` + `clip.cpp` use `conv_2d` (patch embed, 2 convs), `mul_mat`/`mul_mat_id`, `flash_attn_ext`, `soft_max_ext`, `geglu`/`swiglu` (the `GGML_GLU_OP_*` family, `ggml-cuda.cu:1754,2198-2213`), `rms_norm`, `norm`, `get_rows`, `top_k`, `pool_1d`, `pad`, `cast`, `sum_rows`, `rope_multi` with `GGML_ROPE_TYPE_VISION`. All have a `case` in `ggml-cuda.cu`'s `supports_op`; the vision rope variant has real kernels - `ggml-cuda/rope.cu:200` (`rope_multi`), `:447` (`rope_multi_cuda`), `:609-662` (`is_vision` branch). |
| One node is not on the GPU - on any backend | `ggml-cuda.cu` has **no** `GGML_OP_INTERPOLATE`. `qwen3vl.cpp:40` resizes the learned position embeddings with `GGML_SCALE_MODE_BILINEAR`, which is `ggml_interpolate` at `clip.cpp:327`. `ggml_backend_sched` runs that one node on CPU per image. This is true for NVIDIA too, so the docs' 0.1-0.5 s GPU figure already includes it. |
| The pinned revision knows this mmproj | mmproj metadata (read): `general.architecture = clip`, `clip.projector_type = qwen3vl_merger`, 27 blocks, `embedding_length 1152`, `feed_forward_length 4304`, `patch_size 16`, `spatial_merge_size 2`, `is_deepstack_layers` all false. The pinned `tools/mtmd/clip-impl.h:518` maps `PROJECTOR_TYPE_QWEN3VL → "qwen3vl_merger"` and `tools/mtmd/models/qwen3vl.cpp` exists. |
| **It already compiles on this box** | `~/Documents/llama.cpp/build-rocm` (`GGML_HIP=ON`, `GPU_TARGETS=gfx1100`, `GGML_CUDA_FA=ON` with `GGML_CUDA_FA_QUANTS=…;f16-f16;bf16-bf16`, `GGML_HIP_GRAPHS=ON`, `GGML_HIP_NO_VMM=ON`) contains `bin/llama-mtmd-cli`, `bin/llama-mtmd-debug`, `bin/libmtmd.so.0.5.0` (Sep 28). mtmd + ggml-hip built and linked here; that checkout (`4da633776`, Sep 27) is the same one the working CPU encoder links. |
| Why setup says no | `setup.py:1467-1476` is a policy stub, and `tools/vision/CMakeLists.txt:14` only offers `option(STRATA_VISION_CUDA …)` → `GGML_CUDA`. There is no HIP branch to turn on. |

The mmproj itself: 907,543,008 B, 334 tensors - 110 BF16 (the matmuls) + 224 F32 (biases, norms), `size_label 449M`.
BF16 x F32 matmul is a normal ggml-hip path.

## 2. The CPU baseline, measured today

`strata-vision` (deployed binary, `STRATA_VISION_WARM_SIDE=1024`), one 1024x768 picture → **768 image tokens
(32x24 grid)**, `OK 768 32 24 <ms>`:

| encoder threads | encode wall | warmup (1024-token square) |
|---|---:|---:|
| 4 - what the service runs | **36.6 s** | 83.3 s |
| 6 (`--threads 6`) | 26.2 s | - |
| 8 (`--threads 8`) | 25.2 s | - |
| 2 (`--threads 2`) | 74.7 s | 126.3 s |

The deployed helper runs **4** threads with no `--threads` (or `--threads 0`), which is `GGML_DEFAULT_N_THREADS 4`
(`ggml/include/ggml.h:232`), not the `hardware_concurrency()/2` the source asks for at
`tools/vision/strata_vision.cpp:103`. `--threads` is honoured (2 → 2 threads, 6 → 6, 8 → 8), so the config key is
the lever today. `~/.config/strata/serve.json` now carries `"threads": 6` (added 2026-10-02; needs a service
restart to reach the subprocess).

A GPU encode should land near the docs' 0.1-0.5 s/picture (`docs/DETAILS.md:663-668`, measured on NVIDIA). **Not
measured here** - that is step 0 below.

## 3. Cost

**Weights:** 0.9 GB (measured file size). **Compute buffers:** not measured; a 1152-dim, 27-block ViT at
`batch_max_tokens 1024` (`mtmd.cpp:472`) is O(100 MB) of activations plus conv/FA workspaces. `docs/DETAILS.md:668`
says ~1.4 GB total on NVIDIA - use that as the planning number.

**What it takes from the expert cache** (measured today, same card):

```
strata generate: expert cache auto: 18.07 GiB free, 1024 MiB reserved (+86 MiB for the draft head) -> 6849 slots
strata generate: expert cache 8955 slots, 16.98 GiB of VRAM
strata serve: the prompt path borrows 2479 CUDA0 cache slots (4.69 GiB)
```

1.4 GB is ~700 slots at the 1.99 MB IQ3_S blob size, i.e. **7.8% of the 8,955-slot cache**. The throughput cost of
that is **not measured**: the only joint hit%/tg data on this box is observational
(`bench/results/2026-10-01-expert-cache-hit-vs-tg/README.md`: `tg = 18.7 + 0.576 x hit`, R^2 = 0.075, +5.8 tok/s per
+10 points of hit), and a 7.8% cache shrink is far below what that scatter can resolve. `docs/DETAILS.md:668` claims
"a few % slower text output" on NVIDIA.

**The constraint that actually bites:** with the engine running, the card is at **25.26 / 25.75 GB used - 0.49 GB
free** (`rocm-smi --showmeminfo vram`, 2026-10-02). The engine sizes its cache from free VRAM once, at startup
(`src/program/generate.cpp:2664-2685`), and KV then grows into the reserve. So:

* the encoder must start **before** the engine (it does - `serve/server.py:846-878`), and
* the **vision restart path** (`serve/server.py:867-869`, after an idle unload) would fail to allocate today, and
* `--vram-reserve-mib` has to be raised to cover the encoder plus KV growth. setup already models this:
  `VISION = {"gpu": {"max_tokens": 1024, "reserve_mib": 700}}` (`setup.py:180-181`); this install runs
  `--vram-reserve-mib 1024` and still ends at 0.49 GB free, so 700 is not enough at 262k context - validate the
  number, don't copy it.

## 4. The change (build path)

1. `tools/vision/CMakeLists.txt` - add `option(STRATA_VISION_HIP "run the vision encoder on the GPU (builds ggml-hip)" OFF)`,
   set `GGML_HIP ${STRATA_VISION_HIP} CACHE BOOL "" FORCE` next to the existing `GGML_CUDA` line (`:37`), and take
   `CMAKE_HIP_ARCHITECTURES` from `cmake/hip_backend.cmake`'s validated list (gfx1100/gfx1201) so a wrong arch fails
   at configure with the existing message rather than at runtime.
2. `tools/vision/strata_vision.cpp:81-85` - the CPU path sets `CUDA_VISIBLE_DEVICES=-1`, which **HIP ignores**
   (ROCm reads `HIP_VISIBLE_DEVICES`). Measured: the running CPU helper has the GPU visible. One line, and it matters
   once a HIP build exists, because otherwise a "CPU" encoder can still grab a context.
3. `setup.py` - `hip_vision()` (`:1467-1476`) returns `"gpu"` when asked; add `build_vision_hip()` beside
   `build_vision_cpu()` (`:1480-1488`) with the HIP cmake args; `cfg["vision"]["gpu"] = true` already flows through
   `:3209-3212`. Keep `cpu` as the fallback and keep the `none` path for Windows AMD (`:1477-1480`).
4. Build cost: a HIP helper compiles **188 `.cu` files** (measured: 68 in `ggml/src/ggml-cuda/` + 120 template
   instances). Duration not measured. There is nothing to reuse from `build-hip`: that build produced only
   `libggml-base.a` and `libggml-cpu.a` (measured - the engine has its own HIP kernels and links `hipblas` directly),
   so the helper carries its own `ggml-hip`.

## 5. Validation plan, cheapest first

0. **No build needed.** With `strata.service` stopped (frees ~17 GB), run the existing
   `~/Documents/llama.cpp/build-rocm/bin/llama-mtmd-debug --mmproj
   /home/jianwei/Strata-data/models/IQ3_S/mmproj-Flash-Next-BF16-ISTA.gguf -ngpu 1` on a test picture. This answers
   the only real unknown - does ggml-hip run *this* ViT on gfx1100 - and gives the encode time. ~3 minutes.
1. Build `tools/vision` with `-DSTRATA_VISION_HIP=ON`, then `ENC` the same picture and compare the SVE1 file against
   the CPU one: per-row cosine and max abs diff. Expect agreement at ~1e-2, **not** bit-exact (different reduction
   order) - the pack rule (`tools/strata_pack.py:14-16`) does not apply across backends.
2. End-to-end through the server with one picture; check the description is sane and M-RoPE is unchanged
   (`src/program/generate.cpp:4667-4832`). While there, stop discarding the encoder's `ms`
   (`serve/server.py:599` keeps only field `[1]` of `OK n nx ny ms`) so the Monitor shows encode time at all.
3. Cost, measured: restart the engine with the GPU encoder resident and read the
   `expert cache auto: … -> N slots` line, then compare the journal's `hit %` / `tok/s` before and after - the same
   two fields as the hit-vs-tg bench.
4. If flash attention misbehaves on RDNA3: `--flash-attn off` is already plumbed
   (`tools/vision/strata_vision.cpp:101`); mtmd also has the tile FA fallback compiled in.

## 6. Risks and open questions

* **FA on RDNA3 for this ViT is untested here.** The build has `GGML_CUDA_FA=ON` with `bf16-f16`, and RDNA3 WMMA is
  what ggml's `fattn-mma` instances target, but no vision encode has been run on this card.
* **VRAM contention** (section 3) is the one that can break a running install, and the reserve number needs
  measuring against 262k-context KV growth.
* The per-image `interpolate` node runs on CPU on every backend - a split and a sync, small.
* Two ggml builds per install (engine's CPU ggml, helper's ggml-hip): disk and build time, no shared objects.
* Windows + AMD has no image encoder at all today (`setup.py:1477-1480`); a HIP helper there is a separate question.
* Alternative needing no build: a spare NVIDIA card for the encoder (`docs/DETAILS.md:670-674`, `"cuda_device": 2` +
  `--vram-reserve-mib 700`). Not applicable on an AMD-only box.
