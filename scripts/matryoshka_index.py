#!/usr/bin/env python3
"""Индексы урезанных размерностей из матрицы полной размерности.

Матрёшка не требует второго прогона модели: вектор считается один раз
на 384 измерения, а 256 и 128 получаются срезом и повторной нормировкой.
Поэтому с видеокарты едет одна матрица, а индексы под замер задержки
и размера собираются здесь — это numpy, памяти почти не нужно.

Нормировка обязательна: индекс считает косинус внутренним произведением,
и без неё разница в выдаче была бы разницей длин векторов, а не следствием
обрезки. Молча.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin.retrieval.dense import truncate  # noqa: E402
from rufin.retrieval.model_specs import MODELS, TRAINED  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EMBDIR = os.path.join(ROOT, "data", "embeddings")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="base")
    ap.add_argument("--from-model", default="rufin",
                   help="папка с матрицей полной размерности")
    args = ap.parse_args()

    источник = os.path.join(EMBDIR, args.chunks, args.from_model)
    vectors = os.path.join(источник, "vectors.npy")
    if not os.path.exists(vectors):
        raise SystemExit(
            f"нет матрицы {os.path.relpath(vectors, ROOT)}. Её считает "
            f"`run_phase_d.py --step final` на Kaggle и кладёт в "
            f"embeddings/base/rufin/; привезти целиком, вместе с ids.txt.")
    vec = np.load(vectors)
    ids_path = os.path.join(источник, "ids.txt")
    with open(ids_path, encoding="utf-8") as f:
        ids = [l.rstrip("\n") for l in f if l.strip()]
    if len(ids) != vec.shape[0]:
        raise SystemExit(f"идентификаторов {len(ids)}, векторов {vec.shape[0]}")
    print(f"источник: {os.path.relpath(источник, ROOT)}, {vec.shape[0]} × {vec.shape[1]}, "
          f"{vec.nbytes / 1048576:.0f} МБ")

    for имя in TRAINED:
        spec = MODELS[имя]
        if not spec.truncate_dim:
            continue
        куда = os.path.join(EMBDIR, args.chunks, имя)
        os.makedirs(куда, exist_ok=True)
        срез = truncate(vec.astype("float32"), spec.truncate_dim).astype("float16")
        np.save(os.path.join(куда, "vectors.npy"), срез)
        with open(os.path.join(куда, "ids.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(ids) + "\n")
        with open(os.path.join(куда, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"model": имя, "из": args.from_model, "chunks": len(ids),
                       "dim": spec.truncate_dim, "dtype": "float16",
                       "size_mb": round(срез.nbytes / 1048576, 1)}, f,
                      ensure_ascii=False, indent=1)
        print(f"   {имя:<12} dim {spec.truncate_dim:>4}  "
              f"{срез.nbytes / 1048576:>6.1f} МБ  -> "
              f"{os.path.relpath(куда, ROOT)}")
    print("\nдальше: make latency DENSE=rufin-128")


if __name__ == "__main__":
    main()
