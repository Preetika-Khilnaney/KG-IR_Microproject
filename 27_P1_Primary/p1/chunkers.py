"""The three chunking strategies (+ "fulldoc" = no-chunking baseline).

All sizes are in the embedder's (BGE WordPiece) tokens. Overlap = C.CHUNK_OVERLAP for every strategy.
  fixed     - consecutive windows of exactly `size` tokens; cuts ignore sentence boundaries
  recursive - split on paragraph -> line -> sentence -> word, greedily pack pieces up to `size`
  semantic  - embed sentences, start a new chunk where adjacent-sentence cosine distance exceeds
              the document's SEMANTIC_PERCENTILE-th percentile; `size` is a hard cap
"""
import re
import time

import numpy as np

from . import config as C
from .embed import billed

SEPARATORS = ["\n\n", "\n", ". ", " "]
_SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'(\[])|\n+')


def split_sentences(text):
    return [s.strip() for s in _SENT_RE.split(text) if s and s.strip()]


class Chunker:
    """`emb` must provide .tokenizer (fast HF tokenizer), .n_tokens(texts) and .embed(texts)."""

    def __init__(self, emb):
        self.emb = emb
        self.tok = emb.tokenizer

    # ---- fixed-length ------------------------------------------------------
    def fixed(self, text, size):
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
        offs = enc["offset_mapping"]
        if not offs:
            return [text.strip()] if text.strip() else []
        step = max(1, size - C.CHUNK_OVERLAP)
        out = []
        for i in range(0, len(offs), step):
            s, e = offs[i][0], offs[min(i + size, len(offs)) - 1][1]
            out.append(text[s:e].strip())
            if i + size >= len(offs):
                break
        return [c for c in out if c]

    # ---- recursive ---------------------------------------------------------
    def recursive(self, text, size, seps=SEPARATORS):
        text = text.strip()
        if not text:
            return []
        if self.emb.n_tokens(text)[0] <= size:
            return [text]
        for i, sep in enumerate(seps):
            if sep in text:
                break
        else:
            return self.fixed(text, size)
        parts = text.split(sep)
        pieces = [p + sep for p in parts[:-1]] + [parts[-1]]
        pieces = [p for p in pieces if p.strip()]
        lens = self.emb.n_tokens(pieces)
        chunks, cur, cur_len = [], "", 0
        for p, l in zip(pieces, lens):
            if l > size:
                if cur.strip():
                    chunks.append(cur.strip())
                cur, cur_len = "", 0
                chunks.extend(self.recursive(p, size, seps[i + 1:]))
            elif cur_len + l <= size:
                cur += p
                cur_len += l
            else:
                chunks.append(cur.strip())
                cur, cur_len = p, l
        if cur.strip():
            chunks.append(cur.strip())
        out = []
        for c, l in zip(chunks, self.emb.n_tokens(chunks)):     # hard cap (token counts only ~additive)
            out.extend(self.fixed(c, size) if l > size else [c])
        return out

    # ---- semantic ----------------------------------------------------------
    def prepare_sentences(self, docs):
        """Sentence split + sentence embeddings, shared by both chunk sizes. Its full time and tokens
        are charged to EACH semantic configuration in T2."""
        t0 = time.perf_counter()
        per_doc = [split_sentences(d) for d in docs]
        flat = [s for ss in per_doc for s in ss]
        lens = self.emb.n_tokens(flat)
        E = self.emb.embed(flat)
        sents, embs, lns, k = [], [], [], 0
        for ss in per_doc:
            sents.append(ss)
            embs.append(E[k:k + len(ss)])
            lns.append(lens[k:k + len(ss)])
            k += len(ss)
        return dict(sents=sents, embs=embs, lens=lns, time=time.perf_counter() - t0,
                    tokens=billed(lens), n=len(flat))

    def semantic(self, sents, embs, lens, size, pct=C.SEMANTIC_PERCENTILE):
        if not sents:
            return []
        groups = [[0]]
        if len(sents) > 1:
            dist = 1.0 - np.sum(embs[:-1] * embs[1:], axis=1)
            thr = np.percentile(dist, pct)
            for i in range(1, len(sents)):
                if dist[i - 1] > thr:
                    groups.append([i])
                else:
                    groups[-1].append(i)
        chunks = []
        for g in groups:
            cur, cur_len = [], 0
            for i in g:
                if lens[i] > size:                      # single over-long sentence -> token windows
                    if cur:
                        chunks.append(" ".join(sents[j] for j in cur))
                    cur, cur_len = [], 0
                    chunks.extend(self.fixed(sents[i], size))
                    continue
                if cur and cur_len + lens[i] > size:
                    chunks.append(" ".join(sents[j] for j in cur))
                    cur, cur_len = [], 0
                cur.append(i)
                cur_len += lens[i]
            if cur:
                chunks.append(" ".join(sents[j] for j in cur))
        return chunks

    # ---- corpus ------------------------------------------------------------
    def chunk_corpus(self, docs, strategy, size, sent_cache=None):
        """Returns (chunks, owner) where owner[i] = index of the doc chunk i came from (sorted)."""
        if strategy == "semantic" and sent_cache is None:
            sent_cache = self.prepare_sentences(docs)
        chunks, owner = [], []
        for di, d in enumerate(docs):
            if strategy == "fixed":
                cs = self.fixed(d, size)
            elif strategy == "recursive":
                cs = self.recursive(d, size)
            elif strategy == "semantic":
                cs = self.semantic(sent_cache["sents"][di], sent_cache["embs"][di], sent_cache["lens"][di], size)
            elif strategy == "fulldoc":
                cs = [d]
            else:
                raise ValueError(strategy)
            if not cs:
                cs = [d if d.strip() else "(empty)"]  # every doc keeps >= 1 chunk
            chunks.extend(cs)
            owner.extend([di] * len(cs))
        return chunks, np.asarray(owner)
