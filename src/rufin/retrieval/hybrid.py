"""Объединение выдач: Reciprocal Rank Fusion.

RRF складывает не оценки, а обратные ранги: 1/(k + позиция). Это важно,
потому что оценки BM25 и косинусной близости живут в разных шкалах и
несопоставимы напрямую — любая нормировка «в ноль-единицу» зависит от
разброса конкретного запроса и добавляет произвол. Ранги же сопоставимы
всегда, а константа k=60 сглаживает вклад первых позиций.
"""
from __future__ import annotations

from collections.abc import Sequence

RRF_K = 60


def rrf(runs: Sequence[Sequence[tuple[str, float]]], k: int = RRF_K,
        top: int = 100) -> list[tuple[str, float]]:
    """Слить несколько выдач. Каждая выдача — список (id, оценка) по убыванию."""
    scores: dict[str, float] = {}
    for run in runs:
        for rank, (doc_id, _) in enumerate(run, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])[:top]
