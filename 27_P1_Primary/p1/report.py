"""Build the report tables and figures from results/results.csv:  python -m p1.report

Outputs in results/: T1_quality.csv, T2_cost.csv, tables.md, F1_ndcg_by_strategy.png,
F2_recall_curves.png, findings.txt, cost_summary.json
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

LABEL = {"bm25": "BM25 (full doc, lexical ref.)", "fulldoc": "★ BASELINE: dense, no chunking",
         "fixed": "Fixed-length", "recursive": "Recursive", "semantic": "Semantic"}
ORDER = ["fulldoc", "bm25"] + C.STRATEGIES
METRICS = [("R@5", "Recall@5"), ("R@10", "Recall@10"), ("MRR@10", "MRR@10"), ("NDCG@10", "NDCG@10")]
SIZE_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
STRAT_COLORS = {"fixed": "#4C78A8", "recursive": "#F58518", "semantic": "#54A24B", "fulldoc": "#222222", "bm25": "#888888"}


def pm(x, fmt="{:.3f}"):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return fmt.format(x.mean())
    return f"{fmt.format(x.mean())} ± {fmt.format(x.std(ddof=1))}"


def _sorted(df):
    df = df.copy()
    df["_o"] = df["strategy"].map({s: i for i, s in enumerate(ORDER)})
    df["_c"] = df["chunk_size"].map(lambda c: 0 if c == "full doc" else int(c))
    return df.sort_values(["dataset", "_o", "_c"]).drop(columns=["_o", "_c"])


def load():
    df = pd.read_csv(C.RESULTS / "results.csv", dtype={"chunk_size": str})
    bl = df[df.strategy == C.BASELINE].set_index(["dataset", "seed"])["NDCG@10"]
    df["dNDCG_vs_base"] = df["NDCG@10"].values - bl.loc[list(zip(df.dataset, df.seed))].values
    return df


def make_t1(df):
    rows = []
    for (ds, st, cs), d in df.groupby(["dataset", "strategy", "chunk_size"], sort=False):
        r = {"dataset": ds, "strategy": st, "chunk_size": cs, "Dataset": ds, "Strategy": LABEL[st], "Chunk size (tok)": cs}
        for k, lab in METRICS:
            r[lab] = pm(d[k])
        r["ΔNDCG@10 vs baseline (paired)"] = "—" if st == C.BASELINE else pm(d["dNDCG_vs_base"], "{:+.3f}").replace("± +", "± ")
        rows.append(r)
    return _sorted(pd.DataFrame(rows)).drop(columns=["dataset", "strategy", "chunk_size"]).reset_index(drop=True)


def make_t2(df):
    rows = []
    for (ds, st, cs), d in df.groupby(["dataset", "strategy", "chunk_size"], sort=False):
        rows.append({"dataset": ds, "strategy": st, "chunk_size": cs, "Dataset": ds, "Strategy": LABEL[st],
                     "Chunk size (tok)": cs, "# chunks": pm(d.n_chunks, "{:,.0f}"),
                     "Avg tokens / chunk": "—" if d.avg_chunk_tokens.isna().all() else f"{d.avg_chunk_tokens.mean():.1f}",
                     "Tokens embedded (index)": f"{d.embedded_tokens.mean():,.0f}",
                     "Chunking time (s)": pm(d.chunk_time_s), "Embedding time (s)": pm(d.embed_time_s),
                     "Indexing time (s)": pm(d.index_time_s), "Search latency (ms/query)": pm(d.search_ms_per_query)})
    return _sorted(pd.DataFrame(rows)).drop(columns=["dataset", "strategy", "chunk_size"]).reset_index(drop=True)


def _agg(df):
    cols = list(dict.fromkeys([k for k, _ in METRICS] + [f"R@{k}" for k in range(1, C.K_MAX + 1)]))
    return df.groupby(["dataset", "strategy", "chunk_size"])[cols].agg(["mean", "std"]).sort_index()


def _ms(agg, ds, st, cs, col):
    row = agg.loc[(ds, st, cs)]
    s = row[(col, "std")]
    return float(row[(col, "mean")]), float(0.0 if np.isnan(s) else s)


def make_f1(df, agg, path):
    datasets = list(dict.fromkeys(df.dataset))
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.2 * len(datasets), 4.6), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        x = np.arange(len(C.STRATEGIES))
        w = 0.8 / len(C.CHUNK_SIZES)
        tops = []
        for j, cs in enumerate(C.CHUNK_SIZES):
            ms = [_ms(agg, ds, st, str(cs), "NDCG@10") for st in C.STRATEGIES]
            bars = ax.bar(x + (j - (len(C.CHUNK_SIZES) - 1) / 2) * w, [m for m, _ in ms], w,
                          yerr=[s for _, s in ms], capsize=4, color=SIZE_COLORS[j], label=f"chunk size {cs} tokens")
            for b, (m, sd) in zip(bars, ms):
                ax.text(b.get_x() + b.get_width() / 2, m + sd + 0.01, f"{m:.3f}", ha="center", va="bottom", fontsize=8)
                tops.append(m + sd)
        bm, bs = _ms(agg, ds, C.BASELINE, "full doc", "NDCG@10")
        ax.axhline(bm, color="black", ls="--", lw=1.4, label=f"★ baseline: no chunking ({bm:.3f})")
        ax.axhspan(bm - bs, bm + bs, color="black", alpha=0.08)
        m25, _ = _ms(agg, ds, "bm25", "full doc", "NDCG@10")
        ax.axhline(m25, color="#888888", ls=":", lw=1.4, label=f"BM25 full doc ({m25:.3f})")
        ax.set_xticks(x, [LABEL[s] for s in C.STRATEGIES])
        ax.set_xlabel("Chunking strategy")
        ax.set_ylabel("NDCG@10 (mean ± std over seeds)")
        ax.set_title(ds.upper())
        ax.set_ylim(0, min(1.0, max(tops + [m25, bm + bs]) * 1.35))
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("F1: NDCG@10 by chunking strategy and chunk size", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_f2(df, agg, path):
    datasets = list(dict.fromkeys(df.dataset))
    ls_map = {C.CHUNK_SIZES[0]: "--", C.CHUNK_SIZES[-1]: "-"}
    Ks = np.arange(1, C.K_MAX + 1)
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.4 * len(datasets), 4.8), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        curves = [(C.BASELINE, "full doc"), ("bm25", "full doc")] + [(s, str(c)) for s in C.STRATEGIES for c in C.CHUNK_SIZES]
        for st, cs in curves:
            vals = [_ms(agg, ds, st, cs, f"R@{k}") for k in Ks]
            m = np.array([v[0] for v in vals])
            s = np.array([v[1] for v in vals])
            lab = LABEL[st] if cs == "full doc" else f"{LABEL[st]} · {cs} tok"
            ls = ls_map.get(int(cs), "-") if cs != "full doc" else ("-" if st == C.BASELINE else ":")
            ax.plot(Ks, m, ls=ls, lw=2.2 if st == C.BASELINE else 1.5, color=STRAT_COLORS[st], label=lab,
                    marker="o" if st == C.BASELINE else None, ms=3)
            ax.fill_between(Ks, m - s, m + s, color=STRAT_COLORS[st], alpha=0.08)
        ax.set_xlabel("K (number of retrieved documents)")
        ax.set_ylabel("Recall@K (fraction of gold docs retrieved)")
        ax.set_title(ds.upper())
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7.5, loc="lower right")
    fig.suptitle(f"F2: Recall@K for each strategy (dashed = {C.CHUNK_SIZES[0]} tok, solid = {C.CHUNK_SIZES[-1]} tok)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def findings(df):
    out = []
    for ds in dict.fromkeys(df.dataset):
        d = df[df.dataset == ds]
        agg = d.groupby(["strategy", "chunk_size"])[["NDCG@10", "dNDCG_vs_base"]].agg(["mean", "std"]).fillna(0.0)
        chunked = agg.drop(index=[i for i in agg.index if i[0] in (C.BASELINE, "bm25")])
        best = chunked[("NDCG@10", "mean")].idxmax()
        worst = chunked[("NDCG@10", "mean")].idxmin()
        base = agg.loc[(C.BASELINE, "full doc"), ("NDCG@10", "mean")]
        dm, dsd = chunked.loc[best, ("dNDCG_vs_base", "mean")], chunked.loc[best, ("dNDCG_vs_base", "std")]
        verdict = "beats" if dm - dsd > 0 else "is below" if dm + dsd < 0 else "is within seed noise of"
        out.append(f"{ds.upper()}: best chunked config = {LABEL[best[0]]} @ {best[1]} tok (NDCG@10 "
                   f"{chunked.loc[best, ('NDCG@10', 'mean')]:.3f}); it {verdict} the no-chunking baseline "
                   f"({base:.3f}; paired Δ = {dm:+.3f} ± {dsd:.3f}). Worst = {LABEL[worst[0]]} @ {worst[1]} tok "
                   f"({chunked.loc[worst, ('NDCG@10', 'mean')]:.3f}).")
        for st in C.STRATEGIES:
            a, b = (agg.loc[(st, str(c)), ("NDCG@10", "mean")] for c in C.CHUNK_SIZES)
            out.append(f"   {LABEL[st]}: {C.CHUNK_SIZES[0]} → {C.CHUNK_SIZES[-1]} tok changes NDCG@10 by {b - a:+.3f}.")
        nb = int((chunked[("dNDCG_vs_base", "mean")] < 0).sum())
        if nb:
            out.append(f"   Negative result: {nb}/{len(chunked)} chunked configs score below the no-chunking baseline on {ds}.")
    return out


def main():
    df = load()
    agg = _agg(df)
    info = json.loads((C.RESULTS / "run_info.json").read_text()) if (C.RESULTS / "run_info.json").exists() else {}
    t1, t2 = make_t1(df), make_t2(df)
    t1.to_csv(C.RESULTS / "T1_quality.csv", index=False)
    t2.to_csv(C.RESULTS / "T2_cost.csv", index=False)
    make_f1(df, agg, C.RESULTS / "F1_ndcg_by_strategy.png")
    make_f2(df, agg, C.RESULTS / "F2_recall_curves.png")
    fnd = findings(df)
    (C.RESULTS / "findings.txt").write_text("\n".join(fnd) + "\n")
    cost = {"hardware": info.get("gpu", "?"), "embedder": C.EMBED_MODEL,
            "total_runtime_min": round(info.get("total_runtime_s", float("nan")) / 60, 2),
            "total_tokens_embedded": int(df.embedded_tokens.sum() + df.query_tokens.sum()),
            "llm_tokens": 0, "seeds": sorted(df.seed.unique().tolist()),
            "n_queries_per_seed": int(df.n_queries.iloc[0]), "corpus_size_per_seed": int(df.n_docs.iloc[0])}
    (C.RESULTS / "cost_summary.json").write_text(json.dumps(cost, indent=2))
    smoke = "\n> **SMOKE RUN — not reportable.**\n" if info.get("smoke") else ""
    md = (f"# P1 Primary results (SciFact, FiQA){smoke}\n\n"
          f"## T1: retrieval quality (mean ± std over seeds {cost['seeds']}; ★ = baseline; MRR is MRR@10)\n\n"
          + t1.to_markdown(index=False)
          + f"\n\n## T2: chunks and indexing cost (hardware: {cost['hardware']})\n"
          "Semantic indexing time includes sentence embedding; search latency includes query embedding.\n\n"
          + t2.to_markdown(index=False)
          + f"\n\n## Cost\n\nTotal runtime {cost['total_runtime_min']} min · {cost['total_tokens_embedded']:,} tokens "
          "embedded · 0 LLM tokens\n\n## Findings (computed from results.csv)\n\n```\n" + "\n".join(fnd) + "\n```\n")
    (C.RESULTS / "tables.md").write_text(md)
    print(md)
    return t1, t2, fnd


if __name__ == "__main__":
    main()
