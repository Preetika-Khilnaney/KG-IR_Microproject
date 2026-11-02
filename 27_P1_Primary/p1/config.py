"""Configuration for P1 (Primary): chunking benchmark on SciFact and FiQA.

Environment overrides: P1_ROOT (output root), P1_SMOKE=1 (tiny run for testing only; never report it).
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("P1_ROOT", Path(__file__).resolve().parents[1]))
RESULTS = ROOT / "results"

SMOKE = os.environ.get("P1_SMOKE") == "1"

# ---- model (open, small) -------------------------------------------------
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "   # BGE retrieval instruction
MAX_SEQ_LEN = 512
EMBED_BATCH = 128
# No LLM: P1 is a retrieval-only benchmark (0 LLM tokens).

# ---- experiment grid -----------------------------------------------------
DATASETS = ["scifact", "fiqa"]
STRATEGIES = ["fixed", "recursive", "semantic"]
CHUNK_SIZES = [128, 256]            # max tokens per chunk (BGE tokenizer tokens)
CHUNK_OVERLAP = 0                   # same (zero) overlap for every strategy
SEMANTIC_PERCENTILE = 90            # cut where adjacent-sentence distance > this per-doc percentile
BASELINE = "fulldoc"                # no chunking: whole document (model truncates at MAX_SEQ_LEN)

# ---- data subset ---------------------------------------------------------
SEEDS = [13, 42, 2024]              # each seed = a different query sample + distractor sample
N_QUERIES = 200                     # test queries per seed
CORPUS_SIZE = 3000                  # docs per seed: all gold docs of those queries + random distractors

# ---- evaluation ----------------------------------------------------------
K_MAX = 20                          # Recall@K curve, K = 1..20

if SMOKE:                           # never use for reported numbers
    SEEDS = [13, 42]
    N_QUERIES = 30
    CORPUS_SIZE = 300
