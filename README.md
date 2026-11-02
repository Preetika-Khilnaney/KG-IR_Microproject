# KG-IR mini-project

## P1 Chunking benchmark (Primary): [27_P1_Primary](27_P1_Primary)
Compares fixed-length, recursive and semantic chunking (128 and 256 tokens) against a no-chunking baseline
on SciFact and FiQA subsets, using BGE-small. Runs on a Colab T4; results go in `27_P1_Primary/results`.
Includes a Gradio web frontend (results dashboard, live search across strategies, chunk inspector).

```
27_P1_Primary/
  p1/          code package (config, embed, data, chunkers, retrieval, metrics, run_experiment, report, app)
  colab/       self-contained Colab notebook + its builder
  tests/       offline mock test of the pipeline
  results/     results.csv, T1/T2 tables, F1/F2 figures, findings, run_info.json (after the Colab run)
  report/      (report to be added)
  README.md, EXPERIMENT_LOG.md
```
See `27_P1_Primary/README.md` for how to run it.
