#!/usr/bin/env python3
"""Сборка пакета для загрузки на Kaggle.

Складывает в dist/kaggle всё, что нужно для тяжёлых вычислений на чужой
видеокарте: корпус актов в сжатом виде и ровно тот код, которым пользуется
проект локально.

Код копируется с сохранением структуры пакета `rufin`, а не переписывается
плоскими файлами: импорты на Kaggle те же, что локально, и разойтись
параметрам нарезки, префиксам моделей или токенизации для BM25 неоткуда.
Разойдись они — идентификаторы фрагментов не совпадут, а причина
несовпадения будет неочевидна.
"""
from __future__ import annotations

import gzip
import os
import shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DIST = os.path.join(ROOT, "dist", "kaggle")

# модули проекта, которые нужны на видеокарте
PACKAGE = [
    "src/rufin/__init__.py",
    "src/rufin/chunking.py",
    "src/rufin/chunk_configs.py",
    "src/rufin/queryfilter.py",
    "src/rufin/benchmark.py",
    "src/rufin/retrieval/__init__.py",
    "src/rufin/retrieval/text.py",
    "src/rufin/retrieval/bm25.py",
    "src/rufin/retrieval/hybrid.py",
    "src/rufin/retrieval/model_specs.py",
]

# скрипты, запускаемые в ноутбуке
SCRIPTS = [
    "kaggle/gpu_common.py",
    "kaggle/run_phase_a.py",
    "kaggle/run_phase_b.py",
    "kaggle/gpu_embed.py",
    "kaggle/gpu_queries.py",
    "kaggle/gpu_search.py",
    "kaggle/README.md",
]


def main() -> None:
    # каталог пересобирается с нуля: иначе в датасет уезжает мусор вроде
    # __pycache__, оставшийся от пробного запуска
    shutil.rmtree(DIST, ignore_errors=True)
    os.makedirs(DIST, exist_ok=True)

    for rel in PACKAGE:
        dst = os.path.join(DIST, rel.replace("src/", "", 1))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(ROOT, rel), dst)

    for rel in SCRIPTS:
        shutil.copyfile(os.path.join(ROOT, rel), os.path.join(DIST, os.path.basename(rel)))

    acts = os.path.join(ROOT, "data", "corpus", "acts.jsonl")
    with open(acts, "rb") as fi, gzip.open(os.path.join(DIST, "acts.jsonl.gz"), "wb",
                                          compresslevel=6) as fo:
        shutil.copyfileobj(fi, fo, length=1 << 20)

    # Один zip вместо папки: Kaggle распаковывает архив сам и сохраняет
    # вложенную структуру, а перетащить один файл проще, чем дерево каталогов.
    archive = shutil.make_archive(os.path.join(ROOT, "dist", "ru-fin-retrieval"),
                                  "zip", DIST)

    total = 0
    print("пакет собран: dist/kaggle")
    for base, _, files in os.walk(DIST):
        for name in sorted(files):
            path = os.path.join(base, name)
            size = os.path.getsize(path)
            total += size
            print(f"   {os.path.relpath(path, DIST):<34} {size / 1048576:>7.2f} МБ")
    print(f"   {'итого':<34} {total / 1048576:>7.2f} МБ")
    print(f"\nдля загрузки на Kaggle: {os.path.relpath(archive, ROOT)} "
          f"({os.path.getsize(archive) / 1048576:.1f} МБ)")
    print("порядок действий — kaggle/README.md")


if __name__ == "__main__":
    main()
