# P1 Chunking benchmark (Primary): SciFact and FiQA

Compares fixed-length, recursive and semantic chunking at 128 and 256 tokens, plus a no-chunking
baseline, using BAAI/bge-small-en-v1.5. Chunk hits are collapsed to documents (best chunk per document).
Reports Recall@5, Recall@10, MRR@10 and NDCG@10 (T1), chunk counts and indexing cost (T2), an NDCG@10
bar chart (F1) and Recall@K curves for K = 1..20 (F2). Includes a Gradio web frontend.

## Layout
```
p1/         code package
  config.py          all settings (datasets, strategies, sizes, seeds, subset size)
  embed.py           BGE-small embedder + token counting
  data.py            SciFact / FiQA loading (BEIR test split) and seeded subsets
  chunkers.py        fixed-length, recursive, semantic chunkers (+ no-chunking baseline)
  retrieval.py       MaxP doc scoring, BM25 tokenizer
  metrics.py         Recall@K, MRR@10, NDCG@10 (+ self-test)
  run_experiment.py  runs the grid, writes results/results.csv and run_info.json
  report.py          T1, T2, F1, F2, tables.md, findings.txt, cost_summary.json
  app.py             Gradio frontend (Results / Live search / Chunk inspector / Method)
colab/      self-contained Colab notebook + its builder
tests/      offline mock test of the whole pipeline (no downloads, no GPU)
results/    outputs of the real Colab run (empty until it is run)
report/     final report
```

## Run
Easiest: upload `colab/P1_Primary_Colab.ipynb` to Colab, choose a T4 GPU, run all cells. It is self-contained,
downloads `results.zip` (unzip into `results/` here), and the last cell opens the frontend via a `gradio.live` link.

Locally (GPU recommended):
```bash
pip install sentence-transformers datasets rank_bm25 gradio tabulate matplotlib pandas
python -m p1.metrics          # metric self-test
python -m p1.run_experiment   # results/results.csv, results/run_info.json
python -m p1.report           # T1, T2, F1, F2, tables.md, findings.txt, cost_summary.json
python -m p1.app --share      # web frontend
python tests/test_pipeline_mock.py   # offline pipeline test (numbers meaningless)
```
`P1_SMOKE=1` runs a tiny version for testing only; never report its numbers.
After editing anything in `p1/`, rebuild the notebook with `python colab/build_notebook.py`.

## Design decisions (state in the report)
- **Data:** SciFact and FiQA, BEIR test split. Per seed: 200 test queries sampled at random, and a corpus of
  3,000 documents = all gold documents of those queries + random distractors. Subset size is set in `p1/config.py`.
- **Seeds:** 13, 42, 2024 = three different query/distractor samples (mean ± std). Chunking and embedding are
  deterministic, so the std measures sampling variation. Δ vs baseline is paired per seed.
- **Sizes** are in BGE-tokenizer tokens; overlap 0 for every strategy; titles are part of the document text.
- **Fixed-length:** consecutive windows of exactly N tokens (slices of the original text).
- **Recursive:** split on paragraph → line → sentence → word, greedily packed up to N tokens (LangChain-style).
- **Semantic:** sentence embeddings with the same BGE model; cut where adjacent-sentence cosine distance exceeds the
  document's 90th percentile; N is a hard cap, so semantic chunks are shorter on average (see T2). Sentence-embedding
  time and tokens are charged to each semantic configuration.
- **Baseline (★):** whole document embedded without chunking (model truncates at 512 tokens).
  BM25 (rank_bm25, whole docs) is reported as a lexical reference.
- **Metrics:** Recall@K = share of relevant documents in the top K; MRR@10; NDCG@10 with linear graded gain.
- **Cost:** chunks, tokens embedded, chunking / embedding / indexing time, search latency; 0 LLM tokens.
