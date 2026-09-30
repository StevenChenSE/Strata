#!/usr/bin/env python3
"""Tokenize ASCII text with a pack's GPT-2-style byte-level BPE, using only the stdlib.

The pack ships vocab.json + merges.txt + the exact Qwen3.5 pre-tokenizer pattern
(tokenizer.json's pre_pattern, transcribed from llama.cpp LLAMA_VOCAB_PRE_TYPE_QWEN35).  That pattern's
\\p{L}/\\p{M}/\\p{N} classes are Unicode; for pure-ASCII text they are exactly [A-Za-z], empty and
[0-9], so the pattern below is the ASCII-equivalent and stdlib `re` can compile it.

Why this exists: `tools/strata_tokenizer.py` needs the third-party `regex` module, which is not
installable on this box (no pip), and real-text prompts are required to measure *realistic* speculative
acceptance (a repetitive prompt accepts ~100 %, pseudo-random ids ~0 %, neither is representative).

Self-check: decode(encode(text)) == text, so the byte-level mapping round-trips exactly.

    ./tok_ascii.py --pack DIR --text FILE [--max-tokens N] [--out FILE]
"""
import argparse, json, re, sys

PATTERN = (r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
           r"|[^\r\nA-Za-z0-9]?[A-Za-z]+|[0-9]| ?[^\sA-Za-z0-9]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + \
         list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}

class Tok:
    def __init__(self, pack):
        v = json.load(open(f"{pack}/tokenizer/vocab.json"))
        self.enc = {k: int(i) for k, i in v.items()}          # token string -> id
        self.dec = {int(i): k for k, i in v.items()}
        self.b2u = bytes_to_unicode()
        self.u2b = {c: b for b, c in self.b2u.items()}
        self.rank = {}
        with open(f"{pack}/tokenizer/merges.txt") as f:
            for i, line in enumerate(f):
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                a, _, b = line.partition(" ")
                self.rank[(a, b)] = len(self.rank)
        self.re = re.compile(PATTERN)

    def _bpe(self, chunk: str):
        syms = [c for c in chunk]
        if len(syms) < 2:
            return syms
        while True:
            best, bi = None, -1
            for i in range(len(syms) - 1):
                r = self.rank.get((syms[i], syms[i + 1]))
                if r is not None and (best is None or r < best):
                    best, bi = r, i
            if best is None:
                return syms
            a, b = syms[bi], syms[bi + 1]
            out, i = [], 0
            while i < len(syms):
                if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
                    out.append(a + b); i += 2
                else:
                    out.append(syms[i]); i += 1
            syms = out
            if len(syms) == 1:
                return syms

    def encode(self, text: str):
        ids = []
        for m in self.re.finditer(text):
            chunk = "".join(self.b2u[b] for b in m.group(0).encode("utf-8"))
            for sym in self._bpe(chunk):
                ids.append(self.enc[sym])
        return ids

    def decode(self, ids):
        s = "".join(self.dec[i] for i in ids)
        return bytes(self.u2b[c] for c in s).decode("utf-8", errors="replace")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--sample", type=int, default=0)
    a = ap.parse_args()
    t = Tok(a.pack)
    text = open(a.text, encoding="utf-8", errors="ignore").read()
    if a.max_tokens:
        # trim on a token boundary: encode a prefix, keep what fits
        lo, hi = 1, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if len(t.encode(text[:mid])) <= a.max_tokens: lo = mid
            else: hi = mid - 1
        text = text[:lo]
    ids = t.encode(text)
    ok = t.decode(ids) == text
    print(f"tokens={len(ids)} chars={len(text)} ratio={len(text)/max(1,len(ids)):.2f} roundtrip={ok}", file=sys.stderr)
    if a.sample:
        print("sample:", " | ".join(repr(t.decode([i])) for i in ids[:a.sample]), file=sys.stderr)
    if not ok:
        print("WARNING: round-trip mismatch", file=sys.stderr)
    out = ",".join(str(i) for i in ids)
    if a.out: open(a.out, "w").write(out + "\n")
    else: print(out)

if __name__ == "__main__":
    main()
