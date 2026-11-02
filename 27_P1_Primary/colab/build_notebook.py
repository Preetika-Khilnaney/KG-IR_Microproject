"""Builds colab/P1_Primary_Colab.ipynb: a single self-contained notebook (no uploads needed).
The p1/*.py sources are embedded as %%writefile cells, so the notebook always matches the package.
    Run: python colab/build_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = ["config", "embed", "data", "chunkers", "retrieval", "metrics", "run_experiment", "report", "app"]
cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)})


def code(t):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": t.strip("\n").splitlines(True)})


md("""
# P1 Chunking benchmark (Primary): SciFact and FiQA

Compares **fixed-length, recursive and semantic chunking** at two chunk sizes (128 and 256 tokens)
against a no-chunking baseline (★), with the open `BAAI/bge-small-en-v1.5` embedder. BM25 on whole documents
is added as a lexical reference. No LLM is used (retrieval-only project).

Reports Recall@5, Recall@10, MRR@10 and NDCG@10 as mean ± std over 3 seeds (T1), number of chunks and
indexing cost (T2), an NDCG@10 grouped bar chart (F1) and Recall@K curves for K = 1..20 (F2).
Each seed samples 200 test queries and a 3,000-document corpus (all gold docs + random distractors);
change `N_QUERIES` / `CORPUS_SIZE` in `p1/config.py` (cell 2) if your problem statement gives another subset size.

**Before you start:** Runtime > Change runtime type > **T4 GPU**, then Runtime > Run all.
No uploads needed. Results are written to `results/`, zipped and downloaded; the last cell starts the
**web frontend** (a public `gradio.live` link). Expected run time: roughly 10-20 minutes on a T4 (not yet timed).
""")

md("## 1. Check the GPU and install packages")
code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip install -q sentence-transformers datasets rank_bm25 gradio tabulate matplotlib pandas
""")

md("## 2. Write the project code (p1 package)")
code("import os\nos.makedirs('/content/p1proj/p1', exist_ok=True)\n%cd /content/p1proj\nopen('p1/__init__.py','w').write('')")
for name in MODULES:
    src = (ROOT / "p1" / f"{name}.py").read_text(encoding="utf-8")
    code(f"%%writefile p1/{name}.py\n{src}")

md("## 3. Self-test the metrics, then run the full experiment")
code("!python -m p1.metrics")
md("""
Main step. Runs in this kernel (so the frontend can reuse the loaded model and data) and prints one line per
configuration: chunks, Recall@5/10, MRR@10, NDCG@10 and indexing time, for 2 datasets x 3 seeds.
""")
code("""
from p1 import run_experiment
results, EMB, DATA = run_experiment.main()
""")

md("## 4. Tables T1, T2 and figures F1, F2")
code("""
from p1 import report
from IPython.display import Image, display
T1, T2, FINDINGS = report.main()
display(Image("results/F1_ndcg_by_strategy.png"))
display(Image("results/F2_recall_curves.png"))
""")

md("## 5. Download the results folder\nUnzip it into `27_P1_Primary/results/` in the repository and commit.")
code("""
import shutil
shutil.make_archive("results", "zip", "results")
try:
    from google.colab import files
    files.download("results.zip")
except Exception:
    print("Saved results.zip")
""")

md("""
## 6. Frontend
Starts the Gradio app with four tabs: **Results** (T1/T2/F1/F2 + findings), **Live search** (compare strategies
on any query; gold docs marked ✅), **Chunk inspector** (where each strategy cuts a document) and **Method**.
Click the `gradio.live` link it prints. Stop the cell to shut it down.
""")
code("""
from p1.app import build_demo, launch
demo = build_demo(EMB, DATA)
launch(demo, share=True, debug=True)
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}
out = ROOT / "colab" / "P1_Primary_Colab.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", out, "with", len(cells), "cells")
