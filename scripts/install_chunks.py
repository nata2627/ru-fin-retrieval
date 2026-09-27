#!/usr/bin/env python3
"""Установка фрагментов, выгруженных с видеокарты, в качестве рабочей нарезки.

Нарезка, по которой считались эмбеддинги и писались вопросы, считается
единственно верной: граница фрагмента зависит от версии токенизатора,
и повторять нарезку у себя нельзя — под теми же именами окажется другой
текст. Скрипт распаковывает файл на место и сразу сверяет, что под
эталонными фрагментами лежит именно тот текст, по которому писались
вопросы.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHUNKDIR = os.path.join(ROOT, "data", "chunks")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help="скачанный chunks_base.jsonl.gz")
    ap.add_argument("--config", default="base")
    ap.add_argument("--no-check", action="store_true", help="не сверять с запросами")
    args = ap.parse_args()

    src = os.path.expanduser(args.file)
    if not os.path.exists(src):
        raise SystemExit(f"файла нет: {src}")

    os.makedirs(CHUNKDIR, exist_ok=True)
    dst = os.path.join(CHUNKDIR, f"{args.config}.jsonl")
    if os.path.exists(dst):
        backup = dst + ".local"
        shutil.move(dst, backup)
        print(f"прежняя нарезка сохранена как {os.path.relpath(backup, ROOT)}")

    opener = gzip.open if src.endswith(".gz") else open
    count = 0
    with opener(src, "rt", encoding="utf-8") as fi, open(dst, "w", encoding="utf-8") as fo:
        for line in fi:
            if not line.strip():
                continue
            json.loads(line)          # заодно проверяем, что файл не побился при скачивании
            fo.write(line)
            count += 1

    print(f"установлено: {count} фрагментов -> {os.path.relpath(dst, ROOT)} "
          f"({os.path.getsize(dst) / 1048576:.0f} МБ)")

    if args.no_check:
        return
    queries = os.path.join(ROOT, "data", "queries", "synthetic.jsonl")
    if not os.path.exists(queries):
        print("\nзапросов ещё нет, сверка пропущена")
        return
    print()
    sys.exit(subprocess.call([sys.executable, os.path.join(ROOT, "scripts", "check_alignment.py"),
                              "--chunks", args.config]))


if __name__ == "__main__":
    main()
