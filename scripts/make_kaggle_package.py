#!/usr/bin/env python3
"""Сборка пакета для загрузки на Kaggle.

Кладёт в dist/kaggle всё, что нужно для индексации на чужой видеокарте:
корпус актов в сжатом виде и ровно тот код нарезки и те описания моделей,
которыми пользуется проект локально. Копируется, а не переписывается:
разойдись нарезка или префиксы — идентификаторы фрагментов и качество
поиска разъедутся, а причина будет неочевидна.
"""
from __future__ import annotations

import gzip
import os
import shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DIST = os.path.join(ROOT, "dist", "kaggle")

FILES = [
    ("src/rufin/chunking.py", "chunking.py"),
    ("src/rufin/chunk_configs.py", "chunk_configs.py"),
    ("src/rufin/retrieval/model_specs.py", "model_specs.py"),
    ("kaggle/index_kaggle.py", "index_kaggle.py"),
    ("kaggle/README.md", "README.md"),
]


def main() -> None:
    # каталог пересобирается с нуля: иначе в датасет уезжает мусор вроде
    # __pycache__, оставшийся от пробного запуска
    shutil.rmtree(DIST, ignore_errors=True)
    os.makedirs(DIST, exist_ok=True)
    for src, dst in FILES:
        shutil.copyfile(os.path.join(ROOT, src), os.path.join(DIST, dst))

    acts = os.path.join(ROOT, "data", "corpus", "acts.jsonl")
    out = os.path.join(DIST, "acts.jsonl.gz")
    with open(acts, "rb") as fi, gzip.open(out, "wb", compresslevel=6) as fo:
        shutil.copyfileobj(fi, fo, length=1 << 20)

    print(f"пакет собран: dist/kaggle")
    total = 0
    for name in sorted(os.listdir(DIST)):
        size = os.path.getsize(os.path.join(DIST, name))
        total += size
        print(f"   {name:<22} {size / 1048576:>7.1f} МБ")
    print(f"   {'итого':<22} {total / 1048576:>7.1f} МБ")


if __name__ == "__main__":
    main()
