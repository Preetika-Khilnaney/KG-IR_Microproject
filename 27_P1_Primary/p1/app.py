"""Gradio frontend:  python -m p1.app [--share]

Tabs: Results (T1/T2/F1/F2 + findings from results/), Live search (compare strategies on any query,
gold docs marked), Chunk inspector (where each strategy cuts a document), Method.
Live indexes are built on demand for the first seed's subset and cached in memory.
"""
import argparse
import html
import json
import os

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from .chunkers import Chunker
from .report import LABEL

CSS = """
.hero{padding:18px 22px;border-radius:14px;background:linear-gradient(120deg,#1e1b4b,#3730a3 60%,#4f46e5);color:#fff;margin-bottom:6px}
.hero h1{margin:0 0 4px;font-size:1.55rem;color:#fff}.hero p{margin:0;opacity:.85}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:10px 0}
.card{border:1px solid rgba(127,127,127,.25);border-radius:12px;padding:10px 14px}
.card .k{font-size:.75rem;text-transform:uppercase;letter-spacing:.04em;opacity:.65}.card .v{font-size:1.2rem;font-weight:650}
.hit{border:1px solid rgba(127,127,127,.25);border-radius:12px;padding:10px 14px;margin-bottom:8px}
.hit.gold{border-color:#16a34a;box-shadow:0 0 0 1px #16a34a inset}
.hit .meta{font-size:.8rem;opacity:.7;margin-bottom:4px}.hit mark{background:#fde68a;color:#111;border-radius:3px;padding:0 2px}
.col-h{font-weight:650;margin:4px 0 6px}
.chunk{display:inline;padding:1px 2px;border-radius:3px;line-height:1.75;color:#111}
.cbadge{display:inline-block;font-size:.65rem;font-weight:700;background:#111;color:#fff;border-radius:4px;padding:0 4px;margin:0 3px}
.grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}
.panel{border:1px solid rgba(127,127,127,.25);border-radius:12px;padding:10px 12px;font-size:.86rem}
"""
PALETTE = ["#c7d2fe", "#fde68a", "#bbf7d0", "#fecaca", "#e9d5ff", "#bae6fd"]
STRAT_CHOICES = {"★ Baseline: no chunking": "fulldoc", "Fixed-length": "fixed", "Recursive": "recursive", "Semantic": "semantic"}


class LiveIndex:
    """Seed-0 subsets and lazily built chunk indexes for the frontend."""

    def __init__(self, emb, data):
        self.emb, self.chunker = emb, Chunker(emb)
        self.subs = {ds: D.make_subset(data, ds, C.SEEDS[0]) for ds in C.DATASETS}
        self.cache, self.sent = {}, {}

    def index(self, ds, strat, size):
        key = (ds, strat, size)
        if key not in self.cache:
            docs = self.subs[ds]["docs"]
            if strat == "semantic" and ds not in self.sent:
                self.sent[ds] = self.chunker.prepare_sentences(docs)
            chunks, owner = self.chunker.chunk_corpus(docs, strat, size, self.sent.get(ds))
            self.cache[key] = dict(chunks=chunks, C=self.emb.embed(chunks).astype(np.float16),
                                   starts=np.searchsorted(owner, np.arange(len(docs))))
        return self.cache[key]

    def doc_chunks(self, ds, di, strat, size):
        doc = self.subs[ds]["docs"][di]
        if strat == "semantic":
            sc = self.chunker.prepare_sentences([doc])
            return self.chunker.semantic(sc["sents"][0], sc["embs"][0], sc["lens"][0], size)
        return self.chunker.chunk_corpus([doc], strat, size)[0]


def _highlight(doc, chunk):
    i = doc.find(chunk[:60]) if chunk else -1
    if i < 0:
        return html.escape(chunk)
    j = i + len(chunk)
    s, e = max(0, i - 160), min(len(doc), j + 160)
    return (("…" if s else "") + html.escape(doc[s:i]) + "<mark>" + html.escape(doc[i:j]) + "</mark>"
            + html.escape(doc[j:e]) + ("…" if e < len(doc) else ""))


def build_demo(emb=None, data=None):
    import gradio as gr
    if emb is None:
        from .embed import Embedder
        emb = Embedder()
    if data is None:
        data = D.load_all()
    live = LiveIndex(emb, data)

    R = C.RESULTS
    T1 = pd.read_csv(R / "T1_quality.csv", dtype=str) if (R / "T1_quality.csv").exists() else None
    T2 = pd.read_csv(R / "T2_cost.csv", dtype=str) if (R / "T2_cost.csv").exists() else None
    COST = json.loads((R / "cost_summary.json").read_text()) if (R / "cost_summary.json").exists() else {}
    FND = (R / "findings.txt").read_text().splitlines() if (R / "findings.txt").exists() else []
    RES = pd.read_csv(R / "results.csv", dtype={"chunk_size": str}) if (R / "results.csv").exists() else None

    def cards():
        items = [("Hardware", COST.get("hardware", emb.hardware)), ("Total runtime", f"{COST.get('total_runtime_min', '—')} min"),
                 ("Tokens embedded", f"{COST['total_tokens_embedded']:,}" if COST else "—"), ("LLM tokens", "0")]
        if RES is not None:
            for ds in C.DATASETS:
                d = RES[(RES.dataset == ds) & ~RES.strategy.isin([C.BASELINE, "bm25"])]
                if len(d):
                    a = d.groupby(["strategy", "chunk_size"])["NDCG@10"].mean()
                    i = a.idxmax()
                    items.append((f"Best on {ds}", f"{LABEL[i[0]]} · {i[1]} — {a.max():.3f}"))
        return '<div class="cards">' + "".join(
            f'<div class="card"><div class="k">{k}</div><div class="v">{html.escape(str(v))}</div></div>' for k, v in items) + "</div>"

    def filter_tables(ds):
        f = lambda T: T if (T is None or ds == "All") else T[T.Dataset == ds]
        return f(T1), f(T2)

    def example_choices(ds):
        s = live.subs[ds]
        return [f"{q} │ {t[:110]}" for q, t in zip(s["qids"][:80], s["queries"][:80])]

    def pick_example(ds, choice):
        if not choice:
            return gr.update()
        s = live.subs[ds]
        return s["queries"][s["qids"].index(choice.split(" │ ")[0])]

    def gold_for(ds, query):
        s = live.subs[ds]
        for q, t in zip(s["qids"], s["queries"]):
            if t.strip() == query.strip():
                return set(s["qrels"][q])
        return set()

    def live_search(ds, strategies, size, query, k):
        if not query or not query.strip():
            return "<p>Type a query or pick a benchmark query.</p>"
        if not strategies:
            return "<p>Select at least one strategy.</p>"
        s = live.subs[ds]
        gold, k = gold_for(ds, query), int(k)
        qv = emb.embed([query], is_query=True)[0]
        cols = []
        for lab in strategies:
            st = STRAT_CHOICES[lab]
            sz = None if st == C.BASELINE else int(size)
            L = live.index(ds, st, sz)
            sc = L["C"].astype(np.float32) @ qv
            Dm = np.maximum.reduceat(sc, L["starts"])
            top = np.argsort(-Dm, kind="stable")[:k]
            n_gold = sum(s["doc_ids"][d] in gold for d in top)
            head = lab + ("" if sz is None else f" · {sz} tok") + (f" — {n_gold}/{len(gold)} gold in top-{k}" if gold else "")
            body = ""
            for r, di in enumerate(top, 1):
                lo = L["starts"][di]
                hi = L["starts"][di + 1] if di + 1 < len(L["starts"]) else len(sc)
                ch = L["chunks"][lo + int(np.argmax(sc[lo:hi]))]
                did = s["doc_ids"][di]
                g = did in gold
                body += (f'<div class="hit {"gold" if g else ""}"><div class="meta">#{r} · doc {html.escape(did)} · '
                         f'score {Dm[di]:.3f}{" · ✅ gold" if g else ""}</div>{_highlight(s["docs"][di], ch)}</div>')
            cols.append(f'<div><div class="col-h">{html.escape(head)}</div>{body}</div>')
        note = "" if gold else "<p style='opacity:.7'>Free-text query: no gold labels, so hits are not marked.</p>"
        return note + '<div class="grid3">' + "".join(cols) + "</div>"

    def doc_choices(ds):
        s = live.subs[ds]
        gold = sorted(set(d for q in s["qids"] for d in s["qrels"][q]))
        idx = {d: i for i, d in enumerate(s["doc_ids"])}
        return [f"{d} │ {s['docs'][idx[d]][:90].replace(chr(10), ' ')}" for d in gold[:150]]

    def inspect_doc(ds, choice):
        if not choice:
            return ""
        s = live.subs[ds]
        di = s["doc_ids"].index(choice.split(" │ ")[0])
        panels = []
        for st in C.STRATEGIES:
            for cs in C.CHUNK_SIZES:
                chs = live.doc_chunks(ds, di, st, cs)
                lens = emb.n_tokens(chs)
                spans = "".join(f'<span class="cbadge">{i + 1}·{l}t</span><span class="chunk" style="background:'
                                f'{PALETTE[i % len(PALETTE)]}">{html.escape(c)}</span> ' for i, (c, l) in enumerate(zip(chs, lens)))
                panels.append(f'<div class="panel"><div class="col-h">{LABEL[st]} · {cs} tok — {len(chs)} chunk(s)</div>{spans}</div>')
        n = emb.n_tokens(s["docs"][di])[0]
        return f"<p><b>Doc {html.escape(s['doc_ids'][di])}</b> · {n} tokens</p>" + '<div class="grid3">' + "".join(panels) + "</div>"

    method_md = f"""
**Datasets:** {", ".join(d.upper() for d in C.DATASETS)} (BEIR test split). Per seed: {C.N_QUERIES} queries, {C.CORPUS_SIZE} docs (all gold + random distractors). Seeds {C.SEEDS}.
**Embedder:** `{C.EMBED_MODEL}` (fp16 on GPU), BGE query instruction, cosine similarity. **No LLM** (retrieval-only project).
**Strategies** (budget in embedder tokens, overlap {C.CHUNK_OVERLAP}):
- *Fixed-length* — consecutive windows of exactly N tokens.
- *Recursive* — split on paragraph → line → sentence → word, greedily packed up to N tokens.
- *Semantic* — sentence embeddings; new chunk where adjacent-sentence distance exceeds the doc's {C.SEMANTIC_PERCENTILE}th percentile; N is a hard cap.
- ★ *Baseline* — whole document embedded as one vector (truncated at {C.MAX_SEQ_LEN} tokens). BM25 on whole docs is a lexical reference.
**Doc ranking:** MaxP (doc score = best chunk score). **Metrics:** Recall@5/10, MRR@10, NDCG@10, Recall@1–20; mean ± std over seeds.
"""
    theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="amber", neutral_hue="slate",
                           font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"])
    gr6 = int(gr.__version__.split(".")[0]) >= 6           # Gradio 6 moved theme/css from Blocks() to launch()
    blocks_kw = {} if gr6 else dict(theme=theme, css=CSS)
    with gr.Blocks(title="P1 · Chunking Benchmark", **blocks_kw) as demo:
        gr.HTML('<div class="hero"><h1>Chunking Benchmark · SciFact & FiQA</h1>'
                '<p>Fixed-length vs recursive vs semantic chunking at 128 / 256 tokens · BGE-small · 3 seeds</p></div>')
        gr.HTML(cards())
        with gr.Tab("Results"):
            if T1 is None:
                gr.Markdown("No results yet: run `python -m p1.run_experiment` and `python -m p1.report` first.")
            else:
                ds_f = gr.Radio(["All"] + C.DATASETS, value="All", label="Dataset")
                gr.Markdown("**T1 · Retrieval quality** (mean ± std over seeds; ★ = baseline; Δ is paired per seed)")
                t1_out = gr.Dataframe(value=T1, wrap=True, interactive=False)
                gr.Markdown("**T2 · Number of chunks and indexing cost**")
                t2_out = gr.Dataframe(value=T2, wrap=True, interactive=False)
                ds_f.change(filter_tables, ds_f, [t1_out, t2_out])
                for f, lab in [("F1_ndcg_by_strategy.png", "F1 · NDCG@10 by strategy and chunk size"),
                               ("F2_recall_curves.png", "F2 · Recall@K, K = 1–20")]:
                    if (R / f).exists():
                        gr.Image(str(R / f), label=lab)
                if FND:
                    gr.Markdown("**Findings (computed from results.csv):**\n\n" + "\n".join(f"- {x.strip()}" for x in FND))
        with gr.Tab("Live search"):
            with gr.Row():
                ds_s = gr.Dropdown(C.DATASETS, value=C.DATASETS[0], label="Dataset")
                size_s = gr.Radio([str(c) for c in C.CHUNK_SIZES], value=str(C.CHUNK_SIZES[0]), label="Chunk size (tokens)")
                k_s = gr.Slider(1, 10, value=5, step=1, label="Top-k docs")
            strat_s = gr.CheckboxGroup(list(STRAT_CHOICES), value=["Fixed-length", "Recursive", "Semantic"], label="Strategies to compare")
            ex_s = gr.Dropdown(example_choices(C.DATASETS[0]), label="Benchmark query (gold docs get ✅)", value=None)
            q_s = gr.Textbox(label="Query", placeholder="e.g. Does vitamin D supplementation reduce fracture risk?")
            go = gr.Button("Search", variant="primary")
            gr.Markdown("<small>The first search for a dataset / strategy / size builds its index (a few seconds on GPU).</small>")
            res_s = gr.HTML()
            ds_s.change(lambda d: gr.update(choices=example_choices(d), value=None), ds_s, ex_s)
            ex_s.change(pick_example, [ds_s, ex_s], q_s)
            go.click(live_search, [ds_s, strat_s, size_s, q_s, k_s], res_s)
            q_s.submit(live_search, [ds_s, strat_s, size_s, q_s, k_s], res_s)
        with gr.Tab("Chunk inspector"):
            ds_c = gr.Dropdown(C.DATASETS, value=C.DATASETS[0], label="Dataset")
            doc_c = gr.Dropdown(doc_choices(C.DATASETS[0]), label=f"Document (gold docs of the seed-{C.SEEDS[0]} subset)", value=None)
            out_c = gr.HTML()
            ds_c.change(lambda d: gr.update(choices=doc_choices(d), value=None), ds_c, doc_c)
            doc_c.change(inspect_doc, [ds_c, doc_c], out_c)
        with gr.Tab("Method"):
            gr.Markdown(method_md)

    demo._p1_launch_kw = dict(theme=theme, css=CSS) if gr6 else {}
    demo._p1_fns = dict(live_search=live_search, inspect_doc=inspect_doc, doc_choices=doc_choices,
                        example_choices=example_choices, pick_example=pick_example, filter_tables=filter_tables)
    return demo


def launch(demo, **kw):
    return demo.launch(**{**demo._p1_launch_kw, **kw})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="create a public gradio.live link")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    a = ap.parse_args()
    launch(build_demo(), share=a.share, server_port=a.port)
