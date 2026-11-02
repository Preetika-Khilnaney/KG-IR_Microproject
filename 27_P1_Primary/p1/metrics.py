"""Recall@1..K, MRR@10, NDCG@10 (linear graded gain).   Self-test:  python -m p1.metrics"""
import math
from collections import defaultdict

import numpy as np

from . import config as C


def evaluate(ranked_ids, qids, qrels, k_max=C.K_MAX):
    """ranked_ids[i] = ranked doc ids for qids[i]; qrels[q] = {doc_id: grade>0}. Returns mean metrics."""
    per = defaultdict(list)
    for qi, q in enumerate(qids):
        rel, ranked = qrels[q], ranked_ids[qi]
        hits = [1 if d in rel else 0 for d in ranked]
        for k in range(1, k_max + 1):
            per[f"R@{k}"].append(sum(hits[:k]) / len(rel))
        per["MRR@10"].append(next((1.0 / (i + 1) for i, h in enumerate(hits[:10]) if h), 0.0))
        dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
        idcg = sum(g / math.log2(i + 2) for i, g in enumerate(sorted(rel.values(), reverse=True)[:10]))
        per["NDCG@10"].append(dcg / idcg if idcg else 0.0)
    return {k: float(np.mean(v)) for k, v in per.items()}


def self_test():
    ranked = [["d1", "d2", "d3"] + [f"x{i}" for i in range(17)]]
    m = evaluate(ranked, ["a"], {"a": {"d2": 1, "d9": 1}})
    assert abs(m["MRR@10"] - 0.5) < 1e-9
    assert abs(m["R@1"] - 0.0) < 1e-9 and abs(m["R@5"] - 0.5) < 1e-9 and abs(m["R@20"] - 0.5) < 1e-9
    assert abs(m["NDCG@10"] - (1 / math.log2(3)) / (1 + 1 / math.log2(3))) < 1e-9
    m = evaluate([["d9", "d2"] + [f"x{i}" for i in range(18)]], ["a"], {"a": {"d2": 1, "d9": 1}})
    assert abs(m["NDCG@10"] - 1.0) < 1e-9 and abs(m["MRR@10"] - 1.0) < 1e-9
    print("metric self-test passed")


if __name__ == "__main__":
    self_test()
