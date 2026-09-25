"""Базовая линия: BM25 по чанкам.

Без неё любые цифры плотного поиска не с чем сравнивать. Считается на CPU,
индекс держится в памяти.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from .text import tokenize


@dataclass
class BM25Index:
    ids: list[str]
    bm25: BM25Okapi
    build_seconds: float

    @classmethod
    def build(cls, chunk_ids: list[str], texts: list[str]) -> "BM25Index":
        t0 = time.monotonic()
        index = BM25Okapi([tokenize(t) for t in texts])
        return cls(ids=list(chunk_ids), bm25=index, build_seconds=time.monotonic() - t0)

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        if k >= len(scores):
            top = np.argsort(-scores)
        else:
            part = np.argpartition(-scores, k)[:k]
            top = part[np.argsort(-scores[part])]
        return [(self.ids[i], float(scores[i])) for i in top[:k]]
