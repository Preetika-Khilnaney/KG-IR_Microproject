"""Offline end-to-end test with a mock tokenizer, mock embedder and synthetic BEIR-shaped data.
No downloads, no GPU. Checks the pipeline runs and the chunk-size caps hold; its numbers mean nothing.

    P1_SMOKE=1 P1_ROOT=/tmp/p1_mock python tests/test_pipeline_mock.py
"""
import hashlib
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("P1_SMOKE", "1")
os.environ.setdefault("P1_ROOT", "/tmp/p1_mock")

from p1 import config as C  # noqa: E402
from p1 import report, run_experiment  # noqa: E402
from p1.metrics import self_test  # noqa: E402

_TOK = re.compile(r"\w+|[^\w\s]")


class MockTok:
    def __call__(self, texts, add_special_tokens=False, truncation=False, return_offsets_mapping=False):
        single = isinstance(texts, str)
        ids, offs = [], []
        for t in [texts] if single else texts:
            ms = list(_TOK.finditer(t))
            ids.append([hash(m.group()) % 30000 for m in ms])
            offs.append([(m.start(), m.end()) for m in ms])
        out = {"input_ids": ids[0] if single else ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offs[0] if single else offs
        return out


class MockEmbedder:
    tokenizer, dim, device, hardware = MockTok(), 64, "cpu", "MockCPU"

    def n_tokens(self, texts):
        texts = [texts] if isinstance(texts, str) else texts
        return [len(x) for x in self.tokenizer(list(texts))["input_ids"]] if texts else []

    def embed(self, texts, is_query=False):
        E = np.full((len(texts), 64), 1e-3, dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"\w+", t.lower()):
                if len(w) > 3:
                    E[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
        return E / np.linalg.norm(E, axis=1, keepdims=True)


def fake_data(seed, n_docs, n_q, nsent):
    rng = random.Random(seed)
    vocab = [f"word{i}" for i in range(2000)]
    topics = [f"topic{i}" for i in range(40)]
    corpus, queries, qrels = {}, {}, defaultdict(dict)
    for d in range(n_docs):
        tp = rng.sample(topics, 2)
        sents = []
        for _ in range(rng.randint(*nsent)):
            w = rng.sample(vocab, rng.randint(6, 25)) + [rng.choice(tp) + "term"] * 2
            rng.shuffle(w)
            sents.append(" ".join(w).capitalize() + ".")
        corpus[f"D{d}"] = {"title": f"Title {tp[0]}" if seed == 1 else "",
                           "text": "\n".join(" ".join(sents[i:i + 4]) for i in range(0, len(sents), 4))}
    for q in range(n_q):
        golds = rng.sample(list(corpus), rng.randint(1, 3))
        queries[f"Q{q}"] = "What about " + " ".join(corpus[golds[0]]["text"].split()[:6]) + "?"
        for g in golds:
            qrels[f"Q{q}"][g] = 1
    return corpus, queries, dict(qrels)


def main():
    self_test()
    emb = MockEmbedder()
    data = {"scifact": fake_data(1, 800, 80, (3, 25)), "fiqa": fake_data(2, 800, 80, (1, 8))}
    df, _, _ = run_experiment.main(emb=emb, data=data)
    assert len(df) == len(C.DATASETS) * len(C.SEEDS) * (2 + len(C.STRATEGIES) * len(C.CHUNK_SIZES))
    report.main()
    for f in ["T1_quality.csv", "T2_cost.csv", "tables.md", "F1_ndcg_by_strategy.png", "F2_recall_curves.png",
              "findings.txt", "cost_summary.json", "run_info.json", "results.csv"]:
        assert (C.RESULTS / f).exists(), f

    from p1.app import LiveIndex, build_demo
    live = LiveIndex(emb, data)
    for ds in C.DATASETS:
        for st in C.STRATEGIES:
            for cs in C.CHUNK_SIZES:
                mx = max(emb.n_tokens(live.index(ds, st, cs)["chunks"]))
                assert mx <= cs + 3, (ds, st, cs, mx)
    demo = build_demo(emb, data)
    fns = demo._p1_fns
    h = fns["live_search"]("scifact", ["★ Baseline: no chunking", "Fixed-length", "Semantic"], "128",
                           fns["pick_example"]("scifact", fns["example_choices"]("scifact")[0]), 5)
    assert "score" in h
    assert "chunk(s)" in fns["inspect_doc"]("fiqa", fns["doc_choices"]("fiqa")[0])
    print("ALL MOCK TESTS PASSED (numbers are meaningless)")
    return demo


if __name__ == "__main__":
    main()
