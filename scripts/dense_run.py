#!/usr/bin/env python3
"""Выдача плотного поиска по готовой матрице фрагментов.

Матрица фрагментов считается на видеокарте один раз. Запросы — другое
дело: их сотни, а не десятки тысяч, и прогнать их через модель можно
здесь. Это снимает видеокарту с любого опыта, который меняет только
запросы: перефразирование, удаление ссылок на акты, префиксы.

Модель берётся из локальной папки весов — ключ `--weights`. Скачивать
ничего не нужно и не следует: веса уже лежат рядом, а путь из описания
модели указывает на будущую публикацию и в этой машине не существует.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin.retrieval.dense import DenseIndex, Encoder, MeanPoolEncoder  # noqa: E402
from rufin.retrieval.model_specs import MODELS  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOP = 50


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rufin", choices=sorted(MODELS))
    ap.add_argument("--weights", default=None,
                    help="папка с весами; без неё берётся путь из описания модели")
    ap.add_argument("--embeddings", default=None,
                    help="папка с vectors.npy и ids.txt; по умолчанию "
                         "data/embeddings/<нарезка>/<модель>")
    ap.add_argument("--chunks", default="base")
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=TOP)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    spec = MODELS[args.model]
    if args.weights:
        spec = dataclasses.replace(spec, path=args.weights)
    emb = args.embeddings or os.path.join(ROOT, "data", "embeddings",
                                          args.chunks, args.model)
    # Поиск на numpy, не на FAISS: в одном процессе с моделью FAISS
    # на маке роняет процесс по SIGSEGV (две сборки libomp). Перебор
    # по шестидесяти тысячам векторов стоит доли секунды.
    index = DenseIndex.from_files(spec, os.path.join(emb, "vectors.npy"),
                                  os.path.join(emb, "ids.txt"),
                                  backend="numpy")
    with open(args.queries, encoding="utf-8") as f:
        queries = [json.loads(l) for l in f if l.strip()]
    print(f"модель {spec.name} из {spec.path}")
    print(f"матрица {index.vectors.shape}, запросов {len(queries)}", flush=True)

    # Свои веса поднимаются напрямую через transformers: в modules.json
    # записаны имена классов той версии sentence-transformers, что стояла
    # в сессии Kaggle, и здешняя их не находит. Готовые модели из таблицы
    # читаются обычным путём.
    making = MeanPoolEncoder if args.weights else Encoder
    enc = making(spec, batch_size=args.batch_size)
    t0 = time.time()
    qvec = enc.encode([q["text"] for q in queries], is_query=True)
    found = index.search_vectors(qvec, k=args.top)
    print(f"закодировано и найдено за {time.time() - t0:.0f} с "
          f"на {enc.device}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for q, hits in zip(queries, found):
            f.write(json.dumps({"query_id": q["query_id"],
                                "ranked": [c for c, _ in hits],
                                "scores": [round(s, 5) for _, s in hits]},
                               ensure_ascii=False) + "\n")
    print(f"файл: {os.path.relpath(args.out, ROOT)}")


if __name__ == "__main__":
    main()
