#!/usr/bin/env python3
"""Сверка своей реализации BM25 с rank_bm25 на малом корпусе.

Реализация переписана ради памяти (см. rufin/retrieval/bm25.py), поэтому
нужна проверка, что формула осталась той же. Сверяются оценки по всем
документам, а не только порядок: совпадение порядка можно получить
и с неверной формулой.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin.retrieval.bm25 import BM25Index          # noqa: E402
from rufin.retrieval.text import tokenize           # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main() -> None:
    path = os.path.join(ROOT, "data", "chunks", "base.jsonl")
    texts, ids = [], []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 500:
                break
            c = json.loads(line)
            ids.append(c["chunk_id"])
            texts.append(c["text"])
    print(f"корпус для сверки: {len(texts)} фрагментов")

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("rank_bm25 не установлен, сверить не с чем")
        return

    reference = BM25Okapi([tokenize(t) for t in texts])
    mine = BM25Index.build(ids, texts)
    print(f"память: своя реализация {mine.size_mb:.1f} МБ")

    queries = ["порядок формирования резервов по ссудам",
               "требования к отчетности кредитных организаций",
               "590-П оценка кредитного риска",
               "срок представления уведомления в Банк России"]
    worst = 0.0
    for q in queries:
        a = np.asarray(reference.get_scores(tokenize(q)), dtype=np.float64)
        b = mine.scores(q).astype(np.float64)
        diff = float(np.max(np.abs(a - b)))
        worst = max(worst, diff)
        print(f"  {q[:44]:<44} наибольшее расхождение {diff:.2e}")
    print(f"\nнаибольшее расхождение по всем запросам: {worst:.2e}")
    print("совпадает" if worst < 1e-4 else "РАСХОЖДЕНИЕ: формула отличается")


if __name__ == "__main__":
    main()
