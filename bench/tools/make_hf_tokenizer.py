#!/usr/bin/env python3
"""Emit a HuggingFace-format tokenizer for a Strata pack, for external harnesses (llama-benchy et al).

Why this exists: a Strata pack ships `vocab.json` + `merges.txt` + a small descriptor `tokenizer.json` written by
`tools/strata_tokenizer.py` (keys: model/pre/vocab_size/n_merges/special_ids/pre_pattern/...).  That descriptor is
NOT a HuggingFace tokenizer.json, so tools that load one - llama-benchy, transformers - silently fall back to a
different vocabulary (gpt2), which makes their token counts and any per-token rate wrong.

How it works: the pack's vocabulary is already GPT-2 byte-level (`Ġ` for space and so on) exactly ike the
reference implementation, so the HF recipe is the same shape as Qwen's own:
    Split(Regex(pre_pattern), isolated)  ->  ByteLevel(add_prefix_space=False)  ->  BPE(vocab, merges)
and the decoder is ByteLevel.

Validation is not optional here: the script encodes a sample of real text with BOTH the new tokenizer and the
pack's own reference implementation (`bench/tools/tok_ascii.py`, which is what the engine uses) and reports the
exact-match rate, the first disagreement, and the round-trip.  A tokenizer that does not match the engine's own is
worse than the gpt2 fallback it replaces, because it looks right.

Usage (needs the `tokenizers` and `regex` packages):
    uv run --no-project --with tokenizers --with regex python bench/tools/make_hf_tokenizer.py \
        --pack /home/jianwei/Strata-data/packs/iq3_s --out /tmp/iq3_s-hf [--sample PATH] [--max-bytes N]
"""
import argparse
import json
import os
import sys

from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from tokenizers import Regex

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
import strata_tokenizer as ST  # noqa: E402  the engine's own tokenizer


def build(pack_dir: str) -> Tokenizer:
    tok_dir = os.path.join(pack_dir, "tokenizer")
    with open(os.path.join(tok_dir, "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    with open(os.path.join(tok_dir, "merges.txt"), encoding="utf-8") as f:
        # this tokenizers version wants pairs, not "a b" strings
        merges = [tuple(ln.split()) for ln in f if ln.strip() and not ln.startswith("#version")]
    with open(os.path.join(tok_dir, "tokenizer.json"), encoding="utf-8") as f:
        desc = json.load(f)

    tk = Tokenizer(models.BPE(vocab=vocab, merges=merges, unk_token=None))
    tk.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(desc["pre_pattern"]), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tk.decoder = decoders.ByteLevel()
    return tk


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True, help="directory to write tokenizer.json into")
    ap.add_argument("--sample", default="", help="text file to validate against (default: the pack's merges.txt)")
    ap.add_argument("--gguf", default="", help="native GGUF shard for the engine's tokenizer (validation reference)")
    ap.add_argument("--max-bytes", type=int, default=20000)
    a = ap.parse_args()

    tk = build(a.pack)
    os.makedirs(a.out, exist_ok=True)
    tk.save(os.path.join(a.out, "tokenizer.json"))
    print(f"wrote {a.out}/tokenizer.json  (vocab {tk.get_vocab_size()})")

    sample_path = a.sample or os.path.join(a.pack, "tokenizer", "merges.txt")
    with open(sample_path, encoding="utf-8", errors="ignore") as f:
        text = f.read(a.max_bytes)
    print(f"validating on {sample_path} ({len(text)} bytes)")

    ref = ST.Tokenizer.from_gguf(a.gguf)
    ref_ids = ref.encode(text)
    new_ids = tk.encode(text, add_special_tokens=False).ids
    rt = tk.decode(new_ids) == text
    print(f"  reference ids {len(ref_ids)}, hf ids {len(new_ids)}, round-trip {rt}")
    if ref_ids == new_ids:
        print("  EXACT MATCH with the engine's own tokenizer")
        return 0
    # report where and how badly they diverge
    n = min(len(ref_ids), len(new_ids))
    first = next((i for i in range(n) if ref_ids[i] != new_ids[i]), n)
    same = sum(1 for i in range(n) if ref_ids[i] == new_ids[i])
    print(f"  MISMATCH: first at index {first}; {same}/{n} ids agree ({100.0 * same / max(n, 1):.1f}%)")
    lo = max(0, first - 4)
    print(f"    reference {ref_ids[lo:first + 6]}")
    print(f"    hf        {new_ids[lo:first + 6]}")
    print(f"    ref decodes  {ref.decode(ref_ids[lo:first + 6])!r}")
    print(f"    hf  decodes  {tk.decode(new_ids[lo:first + 6])!r}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
