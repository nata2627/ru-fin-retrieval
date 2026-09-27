#!/usr/bin/env python3
"""Сверка: тот ли текст лежит под эталонным фрагментом.

Запросы генерируются на видеокарте, метрики считаются здесь, и обе стороны
режут корпус на фрагменты своим кодом. Код один и тот же, но граница
фрагмента считается в токенах, а токенизатор между версиями библиотеки
ведёт себя чуть иначе. Достаточно сдвига в одно слово, чтобы под прежним
именем «акт#номер» оказался другой текст: ключ найдётся, ошибки не будет,
а эталоном станет не тот фрагмент, по которому писался вопрос.

Проверка опирается на то, что при генерации сохранено: `copy_run` —
длина самой длинной цепочки слов, общей у вопроса и его фрагмента.
Пересчитываем её по локальному фрагменту с тем же именем. Совпало —
тексты те же. Не совпало — нарезки разошлись, и дальше идти нельзя.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin.queryfilter import copy_score       # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHUNKDIR = os.path.join(ROOT, "data", "chunks")
QDIR = os.path.join(ROOT, "data", "queries")


def body(chunk: dict) -> str:
    return chunk["text"].split("\n\n", 1)[-1] if "\n\n" in chunk["text"] else chunk["text"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="base")
    ap.add_argument("--queries", default=os.path.join(QDIR, "synthetic.jsonl"))
    args = ap.parse_args()

    path = os.path.join(CHUNKDIR, f"{args.chunks}.jsonl")
    if not os.path.exists(path):
        print(f"нет нарезки {args.chunks}: сначала `make chunks`")
        raise SystemExit(2)

    chunks = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunks[c["chunk_id"]] = c

    queries = [json.loads(l) for l in open(args.queries, encoding="utf-8") if l.strip()]
    checkable = [q for q in queries if "copy_run" in q]
    if not checkable:
        print("в запросах нет поля copy_run — сверять нечем")
        raise SystemExit(2)

    missing, mismatched, ok = [], [], 0
    for q in checkable:
        chunk = chunks.get(q["gold_chunk_id"])
        if chunk is None:
            missing.append(q)
            continue
        run, _ = copy_score(q["text"], body(chunk))
        if run == q["copy_run"]:
            ok += 1
        else:
            mismatched.append((q, run))

    total = len(checkable)
    print(f"нарезка: {args.chunks}, фрагментов {len(chunks)}")
    print(f"проверено запросов: {total}")
    print(f"  совпадает содержимое эталона: {ok} ({100 * ok / total:.0f}%)")
    print(f"  фрагмент не найден:           {len(missing)}")
    print(f"  под тем же именем другой текст: {len(mismatched)}")

    if missing or mismatched:
        print("\nНарезки разошлись. Эталонные фрагменты указывают не на тот текст,")
        print("по которому писались вопросы, и метрики считать нельзя.")
        print("Причина почти всегда одна: версия токенизатора на видеокарте")
        print("отличается от локальной, поэтому границы фрагментов сдвинулись.")
        print("Сверьте transformers и tokenizers с requirements.txt по обе стороны.")
        for q, run in mismatched[:3]:
            print(f"\n  {q['gold_chunk_id']}: при генерации совпадало {q['copy_run']} слов, "
                  f"здесь {run}")
            print(f"    вопрос: {q['text'][:90]}")
        raise SystemExit(1)

    print("\nнарезки совпадают, можно считать метрики")


if __name__ == "__main__":
    main()
