#!/usr/bin/env python3
"""Подбор эталонного фрагмента для вопроса, написанного руками.

Искать ответ среди десятков тысяч чанков вручную нереально, поэтому запрос
прогоняется через BM25 (и через плотный поиск, если матрица посчитана),
а решение — какой фрагмент считать эталоном — остаётся за человеком.

Работает и по одному вопросу, и пакетом: файл с вопросами по строке
превращается в такой же лист проверки, какой готовится для вопросов
из «Разъяснений», — чтобы работа была одинаковой и формат один.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin.retrieval.bm25 import BM25Index  # noqa: E402
from rufin.retrieval.hybrid import rrf  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHUNKDIR = os.path.join(ROOT, "data", "chunks")
EMBDIR = os.path.join(ROOT, "data", "embeddings")
QDIR = os.path.join(ROOT, "data", "queries")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", help="один вопрос")
    ap.add_argument("--file", help="файл с вопросами, по одному в строке")
    ap.add_argument("--chunks", default="base")
    ap.add_argument("--model", default="bge-m3", help="модель для плотного поиска")
    ap.add_argument("--dense", action="store_true",
                    help="добавить плотный поиск: грузит модель на два с лишним "
                         "гигабайта, включать только когда память свободна")
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(QDIR, "manual_candidates.tsv"))
    args = ap.parse_args()

    chunks = [json.loads(l) for l in open(os.path.join(CHUNKDIR, f"{args.chunks}.jsonl"),
                                          encoding="utf-8")]
    by_id = {c["chunk_id"]: c for c in chunks}
    ids = [c["chunk_id"] for c in chunks]
    bm25 = BM25Index.build(ids, [c["text"] for c in chunks])

    # По умолчанию только BM25: он занимает пятьдесят мегабайт и отвечает
    # за миллисекунду, тогда как энкодер требует двух с лишним гигабайт.
    # На восьми гигабайтах общей памяти это разница между «работает»
    # и «система ушла в подкачку».
    dense = None
    emb = os.path.join(EMBDIR, args.chunks, args.model)
    if args.dense:
        from rufin.retrieval.dense import MODELS, DenseIndex, Encoder
        if not os.path.exists(os.path.join(emb, "vectors.npy")):
            raise SystemExit(f"нет матрицы эмбеддингов {args.chunks}/{args.model}: "
                             f"её считает этап A на видеокарте")
        index = DenseIndex.from_files(MODELS[args.model],
                                      os.path.join(emb, "vectors.npy"),
                                      os.path.join(emb, "ids.txt"))
        # матрица, посчитанная на другой нарезке, даст ссылки на несуществующие
        # фрагменты: имена совпадут, а тексты будут не те
        if len(index.ids) != len(chunks):
            raise SystemExit(f"матрица посчитана на другой нарезке: в ней "
                             f"{len(index.ids)} векторов, в корпусе {len(chunks)} "
                             f"фрагментов")
        dense = (index, Encoder(MODELS[args.model]))

    questions = []
    if args.query:
        questions.append(args.query)
    if args.file:
        # строки, начинающиеся с решётки, — пояснения в файле, а не вопросы
        questions += [l.strip() for l in open(args.file, encoding="utf-8")
                      if l.strip() and not l.lstrip().startswith("#")]
    if not questions:
        ap.error("нужен --query или --file")

    def body(c: dict) -> str:
        return c["text"].split("\n\n", 1)[-1] if "\n\n" in c["text"] else c["text"]

    rows = []
    for n, q in enumerate(questions):
        runs = [bm25.search(q, 50)]
        if dense:
            index, enc = dense
            runs.append(index.search_vectors(enc.encode([q], is_query=True), 50)[0])
        found = rrf(runs, top=args.top) if len(runs) > 1 else runs[0][:args.top]
        print(f"\n=== {q}")
        for i, (cid, _) in enumerate(found, start=1):
            c = by_id[cid]
            print(f"  [{i}] {c['number']} от {c['date']}  п. {c.get('units', '')}  "
                  f"{body(c)[:150].replace(chr(10), ' ')}")
            rows.append({
                "query_id": f"man{n:04d}",
                "vybor_1_2_3_4_ili_0": "",
                "vopros": q if i == 1 else "",
                "variant": i,
                "chunk_id": cid,
                "akt": f"{c['number']} от {c['date']}",
                "punkty": c.get("units", ""),
                "razdel": c.get("section", "")[:60],
                "nachalo_fragmenta": body(c)[:220].replace("\n", " "),
            })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"\nлист проверки: {os.path.relpath(args.out, ROOT)}")
    print("заполняется так же, как explan_candidates.tsv: номер подходящего "
          "варианта или 0")


if __name__ == "__main__":
    main()
