"""Embeds the corpus and serves nearest-neighbour retrieval.

Uses FAISS (IndexFlatIP over normalized vectors, i.e. cosine similarity) when
the optional `faiss-cpu` package is installed; otherwise falls back to a
brute-force NumPy search. The corpus here is small (tens of chunks), so the
fallback is not a performance concern — it exists purely so the stub
pipeline needs no extra dependency.
"""
from __future__ import annotations

import numpy as np

from common.backends import EmbeddingBackend
from common.schemas import RAGPassage, RetrievedPassage


class RAGIndex:
    def __init__(self, passages: list[RAGPassage], embedding_backend: EmbeddingBackend):
        self.passages = passages
        self._by_chunk_id = {p.chunk_id: p for p in passages}
        self._embedding_backend = embedding_backend
        # The section heading is embedded with the text: short paraphrased
        # passages often never name their topic ("Contributing factors are...")
        # and would otherwise be unreachable for a plain "What causes...?" query.
        self._vectors = embedding_backend.embed(
            [f"{p.section}. {p.text}" if p.section else p.text for p in passages]
        ).astype("float32")

        self._faiss_index = None
        try:
            import faiss  # optional

            dim = self._vectors.shape[1]
            self._faiss_index = faiss.IndexFlatIP(dim)
            self._faiss_index.add(self._vectors)
        except ImportError:
            pass

    def get_passage(self, chunk_id: str) -> RAGPassage:
        return self._by_chunk_id[chunk_id]

    def search(self, query: str, top_k: int) -> list[RetrievedPassage]:
        qvec = self._embedding_backend.embed([query]).astype("float32")

        if self._faiss_index is not None:
            scores, idx = self._faiss_index.search(qvec, top_k)
            idx, scores = idx[0], scores[0]
        else:
            sims = self._vectors @ qvec[0]
            idx = np.argsort(-sims)[:top_k]
            scores = sims[idx]

        results = []
        for i, s in zip(idx, scores):
            if i < 0:
                continue
            results.append(RetrievedPassage(chunk_id=self.passages[int(i)].chunk_id, similarity=float(s)))
        return results
