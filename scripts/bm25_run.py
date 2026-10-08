#!/usr/bin/env python3
"""Выдача BM25 по нарезке, лежащей в проекте.

Считается здесь, а не на видеокарте. Своя реализация держит корпус
разреженной матрицей частот — около ста мегабайт вместо двух-трёх
гигабайтов у rank_bm25 (см. rufin/retrieval/bm25.py), и восьми гигабайтов
памяти на это хватает с запасом. Видеокарта в BM25 не участвует вовсе.

Нужно для опытов, которые меняют только запросы: если перестроить индекс
стоит полторы минуты, за прогоном на Kaggle ходить незачем.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin.retrieval.bm25 import BM25Index  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOP = 50


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default=os.path.join(ROOT, "data", "chunks", "base.jsonl"))
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=TOP)
    args = ap.parse_args()

    t0 = time.time()
    ids: list[str] = []
    texts: list[str] = []
    with open(args.chunks, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            ids.append(rec["chunk_id"])
            texts.append(rec["text"])
    print(f"фрагментов {len(ids)}, прочитаны за {time.time() - t0:.0f} с", flush=True)

    index = BM25Index.build(ids, texts)
    del texts
    print(f"индекс построен за {index.build_seconds:.0f} с, "
          f"{index.size_mb:.0f} МБ, слов в словаре {len(index.vocab)}", flush=True)

    with open(args.queries, encoding="utf-8") as f:
        queries = [json.loads(l) for l in f if l.strip()]
    t0 = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for q in queries:
            hits = index.search(q["text"], args.top)
            f.write(json.dumps({"query_id": q["query_id"],
                                "ranked": [c for c, _ in hits],
                                "scores": [round(s, 4) for _, s in hits]},
                               ensure_ascii=False) + "\n")
    print(f"{len(queries)} запросов за {time.time() - t0:.1f} с "
          f"({1000 * (time.time() - t0) / max(1, len(queries)):.0f} мс на запрос)")
    print(f"файл: {os.path.relpath(args.out, ROOT)}")


if __name__ == "__main__":
    main()
