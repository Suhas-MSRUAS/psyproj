"""Small, versioned RAG reference corpus (spec section A3).

Passages are short paraphrases of publicly available WHO / NIMH / MedlinePlus
/ NICE patient-facing material, not verbatim copies, to respect source
copyright while keeping full provenance metadata (title, url, population,
retrieval date, chunk id) for citation.
"""
from __future__ import annotations

import json
from pathlib import Path

from common.schemas import RAGPassage


def load_corpus(corpus_dir: Path) -> list[RAGPassage]:
    data = json.loads((corpus_dir / "corpus.json").read_text(encoding="utf-8"))
    return [RAGPassage(**item) for item in data]
