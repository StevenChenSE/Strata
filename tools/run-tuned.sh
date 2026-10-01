#!/usr/bin/env bash
# Launch Strata with the configuration measured on this box (ROCm gfx1100, PCIe 4.0 x16, 92 GiB RAM).
#
# The binary this points at is now the MERGED engine: upstream's AMD HIP backend (origin/main, engine 0.1.28,
# merged into this branch as 2a59383) plus this port's kernels - our WMMA dense GEMMs and prompt attention, our
# VRAM-doorbell verify handshake, the GDN sub-phases and the MMQ gather batching.  The defaults below were
# measured on the pre-merge engine and re-checked on the merged one at 32K: prefill ~1137 against ~1032 tok/s and
# decode ~57.6 against ~55.0, so they remain the right defaults.
#
# TWO known issues with the merged engine are recorded in
# bench/results/2026-09-30-hip-gfx1100-speed/REBASE-UPSTREAM.md and are worth knowing before trusting a single
# run: its expert placement occasionally differs between runs (served from cache versus lent to the CPU, and a
# lent expert "rounds differently" per src/prefill/prefill.cpp), which can flip draft acceptance from 1.000 to
# ~0.81 and cost ~9% decode; and the hipBLASLt tuning table does not engage on this ROCm, so the GEMM fallback is
# hipblasGemmEx.  Setting PCIE_FRAC=0 removes the flip but costs 9% tg and 2.9% pp - a workaround, not a fix.
#
# The defaults below are not guesses; each is the winner of a measured comparison recorded in
# bench/results/2026-09-30-hip-gfx1100-speed/README.md:
#
#   --kv int8            half the KV VRAM of fp16 (1,056 B/cell).  At 256K context it buys back 3.15 GiB of
#                        expert cache (11.13 -> 14.28 GiB).  q4_0 (576 B/cell) buys 4.67 GiB but its quality
#                        is not validated here - see bench/results/2026-09-27-kv-q4 and the G-C gate note.
#   --max-context 262144 the model's own context (GGUF: qwen4exp.context_length = 262144).  Verified to start
#                        at 32K/64K/128K/256K; 8x the context costs ~6 GiB of expert cache at fp16 KV.
#   --spec 4             the draft length that now wins at >=32K context on this build (60.2 vs 54.5 for
#                        spec 2 - re-measured after the intrinsics + residency tuning; the wider window
#                        amortises the structural per-round verify wait).  SPEC=2 is a fallback for short/small.
#
#   STRATA_WMMA_GEMM / STRATA_PA_WMMA are OPT-IN switches on this branch (review agreement with upstream): the
#   WMMA dense GEMMs and prompt attention run only when set to 1.  This launcher enables both by default via
#   WMMA=1 and PA=1; set WMMA=0 and/or PA=0 to disable (values are parsed, so bare "0" turns them off).
#   --pcie-frac 0.30     the measured optimum on the merged engine at 32K: pp 1136.6 and tg 57.6 against
#                        1104.9 and 52.4 with 0 (see the note above), and +10% decode at 1K versus the 0.55
#                        default on the pre-merge engine.
#   --expert-cache auto  policy default; the capacity above is what sizes it.
#
# Usage:
#   tools/run-tuned.sh --tokens 9707,11,52903 --max-new 64
#   tools/run-tuned.sh --tokens-file /tmp/prompt.txt --max-new 256
#   TEXT=notes.md TEXT_MAX_TOKENS=4096 tools/run-tuned.sh --max-new 128
#   DRY_RUN=1 tools/run-tuned.sh --tokens 9707            # print the command without running it
#
# Overridable via the environment: GPU CTX KV SPEC PREFILL PCIE_FRAC CACHE MAXNEW TIMEOUT_S
# KV_RESIDENT TIMING DATA BIN OUT WMMA PA.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${DATA:-/home/jianwei/Strata-data}"
BIN="${BIN:-$REPO/build-hip/strata-hip}"
OUT="${OUT:-/tmp/strata-run.log}"

GPU="${GPU:-0}"
CTX="${CTX:-262144}"
KV="${KV:-int8}"
SPEC="${SPEC:-4}"
PREFILL="${PREFILL:-2048}"
PCIE_FRAC="${PCIE_FRAC:-0.30}"
CACHE="${CACHE:-auto}"
MAXNEW="${MAXNEW:-64}"
TIMEOUT_S="${TIMEOUT_S:-900}"
KV_RESIDENT="${KV_RESIDENT:-0}"
TIMING="${TIMING:-0}"
SPEC_MIN_P="${SPEC_MIN_P:-0.5}"
ADAPT_EVERY="${ADAPT_EVERY:-2}"

PACK="${PACK:-$DATA/packs/iq3_s}"
NATIVE="${NATIVE:-$DATA/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf}"
PLE="${PLE:-$DATA/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf}"
PROFILE="${PROFILE:-$REPO/data/expert-profile.bin}"
MTP="${MTP:-$DATA/models/IQ3_S/mtp/rt}"

TOKENS=""
TOKENS_FILE=""
DRY_RUN="${DRY_RUN:-0}"

die() { echo "run-tuned: $*" >&2; exit 2; }

usage() {
  sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  echo
  echo "Script options (engine flags pass through unchanged):"
  echo "  --tokens LIST        comma-separated prompt token ids"
  echo "  --tokens-file PATH   pretokenized prompt (commas or whitespace)"
  echo "  --max-new N          tokens to generate (default $MAXNEW)"
  echo "  -h, --help           this text"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tokens)      TOKENS="${2:-}"; shift 2 ;;
    --tokens-file) TOKENS_FILE="${2:-}"; shift 2 ;;
    --max-new)     MAXNEW="${2:-}"; shift 2 ;;
    -h|--help)     usage; exit 0 ;;
    *) die "unknown option '$1' (see --help)" ;;
  esac
done

# A text file is the friendlier input: tokenise it with the pack's own vocabulary.
if [[ -n "${TEXT:-}" ]]; then
  [[ -f "$TEXT" ]] || die "TEXT=$TEXT does not exist"
  TOKENS_FILE="${TOKENS_FILE:-/tmp/strata-prompt.$$.txt}"
  tok_args=(--pack "$PACK" --text "$TEXT" --out "$TOKENS_FILE")
  [[ -n "${TEXT_MAX_TOKENS:-}" ]] && tok_args+=(--max-tokens "$TEXT_MAX_TOKENS")
  python3 "$REPO/bench/tools/tok_ascii.py" "${tok_args[@]}"
fi

[[ -n "$TOKENS$TOKENS_FILE" ]] || die "give a prompt: --tokens, --tokens-file, or TEXT=<file>"
[[ -x "$BIN" ]] || die "engine not built: $BIN (cmake --build build-hip -j 12 --target strata-hip)"
for f in "$PACK" "$NATIVE" "$PLE" "$PROFILE" "$MTP"; do
  [[ -e "$f" ]] || die "missing: $f"
done

cmd=(env "HIP_VISIBLE_DEVICES=$GPU")
[[ "$TIMING" == "1" ]] && cmd+=(STRATA_PREFILL_TIMING=1)
[[ "${WMMA:-1}" == "1" ]] && cmd+=(STRATA_WMMA_GEMM=1)
[[ "${PA:-1}" == "1" ]] && cmd+=(STRATA_PA_WMMA=1)
cmd+=("$BIN"
  --pack "$PACK" --native "$NATIVE" --ple-gguf "$PLE"
  --expert-profile "$PROFILE" --expert-cache "$CACHE" --prefill "$PREFILL"
  --spec "$SPEC" --spec-min-p "$SPEC_MIN_P" --adapt-every "$ADAPT_EVERY"
  --pcie-frac "$PCIE_FRAC" --kv "$KV" --max-context "$CTX" --max-new "$MAXNEW"
  --mtp "$MTP")
[[ "$KV_RESIDENT" != "0" ]] && cmd+=(--kv-resident "$KV_RESIDENT")
if [[ -n "$TOKENS_FILE" ]]; then cmd+=(--tokens-file "$TOKENS_FILE"); else cmd+=(--tokens "$TOKENS"); fi

echo "run-tuned: ctx=$CTX kv=$KV spec=$SPEC prefill=$PREFILL pcie_frac=$PCIE_FRAC cache=$CACHE max_new=$MAXNEW"
if [[ "$DRY_RUN" == "1" ]]; then
  printf 'run-tuned:'
  printf ' %q' "${cmd[@]}"
  echo
  exit 0
fi

# A run must never be able to wedge the GPU (AGENTS.md section 3).
pkill -f '[s]trata-hip --pack' 2>/dev/null || true
echo "run-tuned: logging to $OUT (timeout ${TIMEOUT_S}s)"
timeout "$TIMEOUT_S" "${cmd[@]}" > "$OUT" 2>&1
rc=$?
echo "run-tuned: exit $rc"
grep -E 'prefill|decode|expert cache|session is up' "$OUT" | tail -4 || true
exit $rc
