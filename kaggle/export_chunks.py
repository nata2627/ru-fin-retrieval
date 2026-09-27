#!/usr/bin/env python3
"""Выгрузка фрагментов базовой нарезки — чтобы у проекта была одна нарезка.

Граница фрагмента считается в токенах, и версия токенизатора решает, где она
пройдёт. Поэтому нарезка, сделанная на видеокарте, и нарезка, сделанная
на ноутбуке, могут разойтись: имена вида «акт#номер» останутся прежними,
а текст под ними станет другим. Ошибки при этом не возникает, и заметить
подмену по самим метрикам нельзя.

Истиной считается нарезка, на которой посчитаны эмбеддинги. Этот скрипт
её воспроизводит, сверяет со списком имён из матрицы и отдаёт файлом.
Дальше и разметка, и разбор ошибок опираются на него, а не на локальную
нарезку.

Занимает несколько минут: считать ничего не надо, только нарезать.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gpu_common as common          # noqa: E402


def find_ids_file(embeddings: str, config: str) -> str | None:
    hits = sorted(glob.glob(os.path.join(embeddings, config, "*", "ids.txt")))
    return hits[0] if hits else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default=None, help="путь к корпусу; по умолчанию ищется сам")
    ap.add_argument("--embeddings", default=None,
                    help="папка embeddings этапа A; по умолчанию ищется сама")
    ap.add_argument("--config", default="base")
    ap.add_argument("--out", default="/kaggle/working/chunks_base.jsonl.gz")
    args = ap.parse_args()

    import tokenizers as _tk
    import transformers as _tf
    print(f"transformers {_tf.__version__}, tokenizers {_tk.__version__}", flush=True)

    embeddings = args.embeddings
    if embeddings is None:
        found = sorted(glob.glob("/kaggle/input/*/embeddings")) \
            + sorted(glob.glob("/kaggle/input/*/*/embeddings"))
        embeddings = found[0] if found else None
    if embeddings is None:
        raise SystemExit("не найдена папка embeddings этапа A: подключите его вывод "
                         "через + Add Input -> Your Work")
    print("эмбеддинги этапа A:", embeddings, flush=True)

    ruler = common.make_ruler()
    acts = common.load_acts(args.acts)
    chunks = common.build_chunks(acts, args.config, ruler)

    ids_file = find_ids_file(embeddings, args.config)
    if ids_file is None:
        raise SystemExit(f"в {embeddings}/{args.config} нет ids.txt — нечем сверять")
    with open(ids_file, encoding="utf-8") as f:
        saved = [l.rstrip("\n") for l in f if l.strip()]
    mine = [c["chunk_id"] for c in chunks]

    if saved != mine:
        raise SystemExit(
            f"нарезка не совпала с той, на которой считались эмбеддинги: "
            f"здесь {len(mine)} фрагментов, там {len(saved)}. Значит версия "
            f"токенизатора в этой сессии отличается от той, что была на этапе A. "
            f"Так выгружать нельзя — фрагменты будут не те.")
    print(f"сверка пройдена: {len(mine)} фрагментов, имена совпадают до единого", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with gzip.open(args.out, "wt", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"готово: {args.out}, {os.path.getsize(args.out) / 1048576:.0f} МБ", flush=True)
    print("скачать и распаковать в проект как data/chunks/base.jsonl", flush=True)


if __name__ == "__main__":
    main()
