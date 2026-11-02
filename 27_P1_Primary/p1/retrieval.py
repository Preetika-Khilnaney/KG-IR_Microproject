"""Dense retrieval with MaxP (doc score = best chunk score) and the BM25 lexical reference."""
import re

import numpy as np

from . import config as C

_tok_re = re.compile(r"\w+")


def bm25_tok(t):
    return _tok_re.findall(t.lower())


def chunk_starts(owner, n_docs):
    return np.searchsorted(owner, np.arange(n_docs))      # owner is sorted, every doc has >= 1 chunk


def doc_scores_maxp(Q, C_emb, owner, n_docs):
    """(n_queries, n_docs) matrix of best-chunk cosine scores."""
    return np.maximum.reduceat(Q @ C_emb.T, chunk_starts(owner, n_docs), axis=1)


def top_k(D, k=C.K_MAX):
    return np.argsort(-D, axis=1, kind="stable")[:, :k]
