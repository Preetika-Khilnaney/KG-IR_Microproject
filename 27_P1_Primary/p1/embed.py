"""BGE-small embedder + tokenizer helpers shared by the chunkers, the runner and the frontend."""
import platform

import numpy as np

from . import config as C


class Embedder:
    def __init__(self, model_name=C.EMBED_MODEL):
        import torch
        import transformers
        from sentence_transformers import SentenceTransformer
        transformers.logging.set_verbosity_error()
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.max_seq_length = C.MAX_SEQ_LEN
        if self.device == "cuda":
            self.model.half()                       # fp16 on GPU for speed
        self.tokenizer = self.model.tokenizer
        self.dim = self.model.get_sentence_embedding_dimension()
        self.hardware = torch.cuda.get_device_name(0) if self.device == "cuda" else (platform.processor() or "CPU")
        self.embed(["warm up the model"] * 8)       # keep CUDA start-up out of the timings

    def n_tokens(self, texts):
        """Content tokens (no [CLS]/[SEP]) with the embedder's own tokenizer."""
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return []
        return [len(x) for x in self.tokenizer(list(texts), add_special_tokens=False, truncation=False)["input_ids"]]

    def embed(self, texts, is_query=False):
        texts = [C.QUERY_PREFIX + t for t in texts] if is_query else list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        E = self.model.encode(texts, batch_size=C.EMBED_BATCH, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=False)
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        return E.astype(np.float32)


def billed(lens):
    """Tokens actually seen by the encoder (truncated at MAX_SEQ_LEN, incl. 2 special tokens)."""
    return int(sum(min(l, C.MAX_SEQ_LEN - 2) + 2 for l in lens))
