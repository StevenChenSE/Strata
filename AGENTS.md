# AGENTS.md — working agreement for this repo (Strata → ROCm gfx1100 port)

Branch `hip-gfx1100`. Current objective: **HIP performance (pp/tg) against the CUDA counterpart.**
Decode works as of 2026-09-30 (`39139d2` fixed the blocker — an unsigned underflow in
`strata_hip_nanosleep`); the measured baseline and its breakdown are in
`bench/results/2026-09-30-hip-gfx1100-speed/`, and the investigation history is in
`.rocm-eval/REVIEW-decode-hang.md`. Read `.rocm-eval/PORT-STATUS.md` (running log) and that review
**before touching anything**.

## 1. Who does the work

* **Investigation is the driving agent's job.** Deep dives, root-cause hunts, profiling attribution,
  measurement design and deciding what to do next are yours. A subagent is not smarter than you: making
  it the primary investigator costs a round-trip, throws away the context you have accumulated, and in
  this port it produced a 13-minute silence with zero edits and confident reports built on misread logs.
* **Implementation goes to a subagent**, pinned to `provider: jchen-icu`, `model: gemini-3.8-flash-high`
  (the only Gemini route configured on this machine, `~/.dsh/settings.yaml.imported`; there is no bare
  `gemini-3.8-flash`), via the `workflow` tool's
  `agent(prompt, { provider: "jchen-icu", model: "gemini-3.8-flash-high" })` hook — the plain
  `subagent`/`subagent_fork` tools have model selection switched off in this session. Give it the
  finding, the exact file and line numbers, the numbers measured, the constraint list, and the evidence
  format you want back; then review the result yourself per section 2 before trusting it.
* **Exception: a truly mechanical edit of a few lines** (removing a wrong argument, deleting a duplicated
  increment, fixing an offset, one-line arithmetic) may be made directly — waiting a round-trip for that
  is waste. If it needs a design decision, a new mechanism, or more than a few lines, delegate it.
* **A second look from a subagent is welcomed** on any load-bearing conclusion: independent verification
  or adversarial review of your work. Ask it to **falsify**, not to confirm, and treat its answer as a
  cross-check rather than an authority — it has already produced two confident false findings and missed
  a compile break in this port.

## 2. Self-review is not optional

**Self-review your own work, and treat a subagent's summary as a claim rather than a result.** Before
believing anything is done:

1. **Read the real diff yourself** (`git diff`, `git status`). Confirm the change is what the summary
   says, in the file you think, with no collateral edits.
2. **Check every claim against raw output** — the log line, the exit code, the binary/object mtime,
   `git log`. "It now works" with no log evidence means "unverified".
3. **Re-derive the reasoning** for anything load-bearing (an API's parameter meaning, a stream
   ordering argument, an offset/stride computation, a counter's arithmetic). Do not accept a
   mechanism because it sounds plausible.
4. **Re-run the decisive check when it is cheap** (a unit test, a probe binary, a grep, one bounded
   engine run).
5. **Downgrade what you could not check** to "unverified" in your own report, and say why. A
   confidently reported but unchecked fix is worse than an open question.

## 3. Experiment discipline (this is the only machine with the GPU)

* **The driving agent runs every GPU experiment itself — never delegate a run to a subagent.** A
  subagent's sandbox denies `/dev/kfd` (`EACCES` on a `crw-rw-rw-` node, observed), so a delegated
  "go run it" produces a ~115-byte log and a claim that has to be thrown away — it burns a whole
  delegation round. Subagents are for code, static review, CPU-only compiles/ISA probes and log
  analysis; the experiment's evidence must be produced in the driving session.
* Sandbox policy decides whether that is possible: under `danger-full-access` `/dev/kfd` opens and runs
  work; under `workspace-write` it is denied **and escalation prompts are unavailable**, so in that
  state write the exact command, hand it to the user, and analyse the log they return. Check with a
  one-liner before promising a run (`python3 -c "open('/dev/kfd','rb')"` — an
  `UnsupportedOperation: not seekable` error means it opened).
* **`pkill -f '<pattern>'` kills its own wrapper** when the pattern appears in that wrapper's command
  line: a background job beginning with `pkill -f 'build-hip/strata-hip --pack'` SIGTERMs itself before
  it launches anything (observed). Kill in a separate call, and use the bracket trick
  (`pkill -f '[s]trata-hip --pack'`) so the pattern cannot match the killer.
* The GPU is shared with the user's other work. **Never start a run without `timeout`** and never
  leave a wedged engine behind: `pgrep -f 'build-hip/strata-hip --pack'`, then
  `pkill -f 'build-hip/strata-hip --pack'` before every run.
* `pgrep -f` matches the wrapper/`timeout`/`bash -c` command lines too — **filter for the real
  engine PID** (the one whose `comm` is `strata-hip`), not the first match.
* Run long builds and runs as **background jobs**; do not busy-poll. Collect the job, then act.
* Logs: `/tmp/*.log`. Capture both streams (`> /tmp/<name>.log 2>&1`) so the evidence survives.
* Timeouts: `STRATA_VERIFY_TIMEOUT_S=0` means "wait forever", which is the mode for attaching a
  debugger; use a finite value (default 20 s) otherwise.
* A hang that ends inside the ROCm runtime cannot be interrupted from the host: if the engine is
  wedged, `pkill` it before the next run — do not assume it will exit on its own.

## 4. Build & run (incremental, Unix Makefiles)

```bash
cd /home/jianwei/Documents/Strata
cmake --build build-hip -j 12 --target strata-hip        # ~1-3 min after a core/ edit
# full reconfigure only if CMake changed:
# cmake -B build-hip -DSTRATA_ENABLE_HIP=ON -DSTRATA_GGML_DIR=$HOME/Documents/llama.cpp
```

The reproduction used for every round of this bug (adapt paths; `HIP_VISIBLE_DEVICES=0` selects the
7900 XTX):

```bash
cd /home/jianwei/Documents/Strata
timeout 700 env HIP_VISIBLE_DEVICES=0 ./build-hip/strata-hip \
  --pack /home/jianwei/Strata-data/packs/iq3_s \
  --native /home/jianwei/Strata-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
  --ple-gguf /home/jianwei/Strata-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf \
  --expert-profile data/expert-profile.bin --expert-cache 4096 --spec 2 --prefill 2048 \
  --max-context 4096 --max-new 8 \
  --tokens 9707,11,52903,12366,576,17413,32313,25,52903,12366,31337,0,13 \
  > /tmp/run.log 2>&1
```

Useful env: `STRATA_VERIFY_DEBUG=1` (per-layer capture/serve trace), `STRATA_VERIFY_PROFILE=1`
(stage stamps), `STRATA_VERIFY_TIMEOUT_S=<s>`.

## 5. Invariants this port must not break (learned the hard way)

* **RDNA3 has no system-scope shader access.** `~/rdna3.md` §4.1.1: for loads `GLC=1` means *device*
  scope (L2, not system); "all stores/atomic ops are device scope". `__threadfence_system()` on
  gfx11 compiles to waitcnts only. Therefore:
  * a shader spin on **mapped host memory** can never observe a host store — deadline by design;
  * a shader store to **mapped host memory** is not guaranteed to be visible to the host;
  * only **command-processor / copy-engine** operations are system-coherent: `hipStreamWaitValue32`,
    `hipStreamWriteValue32`, `hipMemcpyAsync` (DMA), host-function signals.
* Consequently the verify-window handshake uses **VRAM doorbells** (`d_flag_/d_flagA_/d_flagB_`,
  `d_seq_`) raised by CP writes, polling kernels on the GPU side, and **D2H DMA** for anything the
  host reads back (per-layer plan, pool rows, doorbell payload).
* `hipStreamWaitValue32(stream, ptr, value, flags, mask)` — the 5th argument is a **mask**, not a
  timeout (`hip_runtime_api.h`). Passing a "timeout" there silently makes the comparison
  unsatisfiable. There is no timeout in the stream-ops API; bound the wait on the host.
* A wait packet blocks everything behind it on its stream: never put a `hipStreamWaitValue32` and the
  `hipStreamWriteValue32` that would satisfy it on the same stream.
* GPU-visible state that the host must re-read has to travel through the copy engine in **both**
  directions; the doorbell and its payload must take the same path, or the doorbell can arrive before
  the payload has left L2.

## 6. Traps seen in this port (check these first)

* **THE ONE THAT COST A DAY: unsigned countdown loops in the shim.**
  `compat/hip/cuda_runtime.h`'s `strata_hip_nanosleep` was `for (; ns; ns -= 32u) s_sleep(8)` with an
  **unsigned** `ns`. For `ns = 100` the sequence is 100, 68, 36, 4, 4294967268, … and never hits 0
  (the value is invariant mod 32; 2^32 is a multiple of 32), so every device-side wait hung forever on
  its first *unsatisfied* poll and never re-read its flag. It presented as a nondeterministic
  "wait kernel spinning at 95 % GPU while the raise sits in memory". Fixed by counting down from
  `(ns + 31) / 32`. **Audit every `for`/`while` with an unsigned counter that decrements, in the shim
  and in kernels.**
* **A probe that bypasses the shim can mask a shim bug.** Five probes "proved" the doorbell primitives
  worked because each defined its own `__nanosleep` via `__builtin_amdgcn_s_sleep` directly and so never
  entered `strata_hip_nanosleep`. When a probe says a primitive works but the engine says it does not,
  check that the probe executes the *same code path* (same headers, same inline functions).
* **Wave dumps are evidence; the answer was in one an hour before it was understood.** rocgdb printed
  `strata_hip_nanosleep (ns=2583458180)`; `2583458180 % 32 == 4`. Read your diagnostics numerically.
* `~Verifier` syncs the stream at destruction; a timed-out window therefore cannot exit and keeps the
  GPU. `std::_Exit()` after the diagnostic is the escape hatch.
* `clock64()` is per-CU: GPU stage "stamps" are not comparable across launches. Use an atomically
  incremented global counter for ordering.
* The MMQ/tensor-core paths compile to `__trap()` under `__CUDA_ARCH__=750`; HIP must refuse the
  batch rather than launch them.
* HIP contracts `__fmul_rn`/`__fadd_rn` into `v_fmac_f32`; the shim routes them through inline asm.
  Anything bit-exact depends on that.
* `src/core/verify.cpp` currently uses `hip*` symbols in code that a CUDA build also compiles
  (`hip_wait_ge`, `hip_raise`, `HIP_CK`). Keep HIP-only code inside `#ifdef STRATA_BACKEND_HIP` — the
  CUDA path is still the product.
* The non-verify decode path (`session.cpp:842` → `elementwise.cu:215` `doorbell_wait_kernel`) still
  spins on a **mapped-host** flag, which gfx11 serves stale for in-flight re-reads. It is unreachable
  for a native IQ pack (`--spec T` with `T >= 2` is mandatory, so the verify window is the only decode
  path) but it is broken by design for other configurations.
