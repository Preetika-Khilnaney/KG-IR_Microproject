"""Run the whole chunking benchmark:  python -m p1.run_experiment

For every (dataset, seed): sample a subset, build chunks with each strategy x chunk size (plus the
no-chunking baseline), embed them with BGE-small, retrieve, collapse chunk hits to documents (best chunk
score per document) and score Recall@1..20, MRR@10 and NDCG@10. BM25 on whole docs is a lexical reference.

Writes to results/: results.csv (one row per dataset x seed x configuration) and run_info.json
"""
import gc
import json
import platform
import random
import sys
import time

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from .chunkers import Chunker
from .embed import billed
from .metrics import evaluate
from .retrieval import bm25_tok, doc_scores_maxp, top_k

CONFIGS = [(C.BASELINE, None)] + [(s, c) for s in C.STRATEGIES for c in C.CHUNK_SIZES]


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    try:
        import torch
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
    except ImportError:
        pass


def _versions():
    out = {"python": platform.python_version()}
    for mod in ["torch", "sentence_transformers", "transformers", "datasets", "rank_bm25", "numpy", "pandas"]:
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            pass
    return out


def run_dataset_seed(name, seed, sub, emb, chunker):
    from rank_bm25 import BM25Okapi
    rows = []
    n_docs, nq = len(sub["docs"]), len(sub["qids"])
    doc_ids = np.array(sub["doc_ids"])
    base = dict(dataset=name, seed=seed, n_queries=nq, n_docs=n_docs)

    t0 = time.perf_counter()
    Q = emb.embed(sub["queries"], is_query=True)
    q_time = time.perf_counter() - t0
    q_tok = billed(emb.n_tokens([C.QUERY_PREFIX + q for q in sub["queries"]]))

    # BM25 reference (lexical, whole docs)
    t0 = time.perf_counter()
    bm = BM25Okapi([bm25_tok(d) for d in sub["docs"]])
    t_idx = time.perf_counter() - t0
    t0 = time.perf_counter()
    Dm = np.vstack([bm.get_scores(bm25_tok(q)) for q in sub["queries"]])
    t_s = time.perf_counter() - t0
    m = evaluate(doc_ids[top_k(Dm)].tolist(), sub["qids"], sub["qrels"])
    rows.append({**base, "strategy": "bm25", "chunk_size": "full doc", "n_chunks": n_docs,
                 "avg_chunk_tokens": np.nan, "embedded_tokens": 0, "chunk_time_s": 0.0, "embed_time_s": 0.0,
                 "index_time_s": t_idx, "search_ms_per_query": 1000 * t_s / nq, "query_tokens": 0, **m})

    sent_cache = chunker.prepare_sentences(sub["docs"]) if "semantic" in C.STRATEGIES else None
    for strat, size in CONFIGS:
        t0 = time.perf_counter()
        chunks, owner = chunker.chunk_corpus(sub["docs"], strat, size, sent_cache)
        t_chunk = time.perf_counter() - t0 + (sent_cache["time"] if strat == "semantic" else 0.0)
        lens = emb.n_tokens(chunks)
        t0 = time.perf_counter()
        CE = emb.embed(chunks)
        t_emb = time.perf_counter() - t0
        t0 = time.perf_counter()
        idx = top_k(doc_scores_maxp(Q, CE, owner, n_docs))
        t_s = time.perf_counter() - t0
        m = evaluate(doc_ids[idx].tolist(), sub["qids"], sub["qrels"])
        rows.append({**base, "strategy": strat, "chunk_size": "full doc" if size is None else str(size),
                     "n_chunks": len(chunks),
                     "avg_chunk_tokens": float(np.mean(np.minimum(lens, C.MAX_SEQ_LEN - 2))),
                     "embedded_tokens": billed(lens) + (sent_cache["tokens"] if strat == "semantic" else 0),
                     "chunk_time_s": t_chunk, "embed_time_s": t_emb, "index_time_s": t_chunk + t_emb,
                     "search_ms_per_query": 1000 * (t_s + q_time) / nq, "query_tokens": q_tok, **m})
        print(f"{name:8s} seed={seed:<5d} {strat:9s} {str(size or 'full'):>4s}  chunks={len(chunks):6,d}  "
              f"R@5={m['R@5']:.4f}  R@10={m['R@10']:.4f}  MRR@10={m['MRR@10']:.4f}  "
              f"NDCG@10={m['NDCG@10']:.4f}  index={t_chunk + t_emb:6.1f}s", flush=True)
    return rows


def main(emb=None, data=None):
    """Returns (results DataFrame, embedder, data) so a notebook can reuse them for the frontend."""
    t_start = time.perf_counter()
    if C.SMOKE:
        print("*** P1_SMOKE=1: tiny test run. Do NOT report these numbers. ***")
    set_seed(C.SEEDS[0])
    if emb is None:
        from .embed import Embedder
        emb = Embedder()
    print(f"Embedder: {C.EMBED_MODEL} | device: {getattr(emb, 'device', '?')} ({emb.hardware})")
    if data is None:
        data = D.load_all()
    chunker = Chunker(emb)
    rows = []
    for name in C.DATASETS:
        for seed in C.SEEDS:
            set_seed(seed)
            sub = D.make_subset(data, name, seed)
            rows += run_dataset_seed(name, seed, sub, emb, chunker)
            gc.collect()
    df = pd.DataFrame(rows)
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.RESULTS / "results.csv", index=False)
    total = time.perf_counter() - t_start
    info = {"embed_model": C.EMBED_MODEL, "device": getattr(emb, "device", "?"), "gpu": emb.hardware,
            "llm": None, "llm_tokens": 0, "smoke": C.SMOKE, "total_runtime_s": round(total, 1),
            "total_tokens_embedded": int(df.embedded_tokens.sum() + df.query_tokens.sum()),
            **_versions(),
            "config": {k: getattr(C, k) for k in ["DATASETS", "STRATEGIES", "CHUNK_SIZES", "CHUNK_OVERLAP",
                                                  "SEMANTIC_PERCENTILE", "BASELINE", "SEEDS", "N_QUERIES",
                                                  "CORPUS_SIZE", "K_MAX", "MAX_SEQ_LEN"]},
            "command": " ".join(sys.argv)}
    (C.RESULTS / "run_info.json").write_text(json.dumps(info, indent=2))
    print(f"\nDone: {len(df)} rows -> {C.RESULTS / 'results.csv'} | total wall time {total / 60:.1f} min")
    return df, emb, data


if __name__ == "__main__":
    main()
