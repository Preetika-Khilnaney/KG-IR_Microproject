"""Load SciFact / FiQA (BEIR test split) and build seeded subsets (never the full datasets)."""
import io
import json
import os
import zipfile
from collections import defaultdict

import numpy as np

from . import config as C

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"


def _load_hf(name):
    from datasets import load_dataset
    c = load_dataset(f"BeIR/{name}", "corpus", split="corpus")
    q = load_dataset(f"BeIR/{name}", "queries", split="queries")
    r = load_dataset(f"BeIR/{name}-qrels", split="test")
    corpus = {str(x["_id"]): {"title": x.get("title") or "", "text": x.get("text") or ""} for x in c}
    queries = {str(x["_id"]): x["text"] for x in q}
    qrels = defaultdict(dict)
    for x in r:
        if int(x["score"]) > 0:
            qrels[str(x["query-id"])][str(x["corpus-id"])] = int(x["score"])
    return corpus, queries, dict(qrels)


def _load_zip(name, root="beir_data"):
    import requests
    path = os.path.join(root, name)
    if not os.path.exists(path):
        os.makedirs(root, exist_ok=True)
        data = requests.get(BEIR_URL.format(name), timeout=300).content
        zipfile.ZipFile(io.BytesIO(data)).extractall(root)
    corpus, queries, qrels = {}, {}, defaultdict(dict)
    with open(os.path.join(path, "corpus.jsonl")) as f:
        for line in f:
            x = json.loads(line)
            corpus[str(x["_id"])] = {"title": x.get("title") or "", "text": x.get("text") or ""}
    with open(os.path.join(path, "queries.jsonl")) as f:
        for line in f:
            x = json.loads(line)
            queries[str(x["_id"])] = x["text"]
    with open(os.path.join(path, "qrels", "test.tsv")) as f:
        next(f)
        for line in f:
            qid, did, s = line.rstrip("\n").split("\t")
            if int(s) > 0:
                qrels[qid][did] = int(s)
    return corpus, queries, dict(qrels)


def load_all(datasets=None):
    """Returns {name: (corpus, queries, qrels)}; Hugging Face first, official BEIR zip as fallback."""
    out = {}
    for name in datasets or C.DATASETS:
        try:
            out[name] = _load_hf(name)
            src = "Hugging Face"
        except Exception as e:
            print(f"[{name}] HF load failed ({type(e).__name__}); using BEIR zip")
            out[name] = _load_zip(name)
            src = "BEIR zip"
        c, q, r = out[name]
        print(f"{name:8s} ({src}): corpus={len(c):,}  queries={len(q):,}  test queries with qrels={len(r):,}")
    return out


def doc_text(d):
    t, x = d["title"].strip(), d["text"].strip()
    return f"{t}\n\n{x}" if t else x


def make_subset(data, name, seed):
    corpus, queries, qrels = data[name]
    rng = np.random.default_rng(seed)
    eligible = sorted(q for q, rel in qrels.items() if q in queries and rel and all(d in corpus for d in rel))
    qids = sorted(rng.choice(eligible, min(C.N_QUERIES, len(eligible)), replace=False).tolist())
    gold = set(d for q in qids for d in qrels[q])
    pool = sorted(set(corpus) - gold)
    n_extra = max(0, min(C.CORPUS_SIZE - len(gold), len(pool)))
    extra = rng.choice(pool, n_extra, replace=False).tolist() if n_extra else []
    doc_ids = sorted(gold | set(extra))
    if len(gold) > C.CORPUS_SIZE:
        print(f"  note: {len(gold)} gold docs > CORPUS_SIZE; corpus holds gold docs only")
    return dict(name=name, seed=seed, qids=qids, queries=[queries[q] for q in qids],
                qrels={q: qrels[q] for q in qids}, doc_ids=doc_ids,
                docs=[doc_text(corpus[d]) for d in doc_ids])
