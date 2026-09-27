#!/usr/bin/env python3
"""Подбор эталонного фрагмента для вопроса, написанного руками.

Искать ответ среди десятков тысяч чанков вручную нереально, поэтому запрос
прогоняется через BM25 (и через плотный поиск, если матрица посчитана),
а решение — какой фрагмент считать эталоном — остаётся за человеком.

Работает и по одному вопросу, и пакетом: файл с вопросами по строке
превращается в заготовку manual.jsonl, где остаётся проставить выбранный
номер варианта.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin.retrieval.bm25 import BM25Index             # noqa: E402
from rufin.retrieval.hybrid import rrf                 # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHUNKDIR = os.path.join(ROOT, "data", "chunks")
EMBDIR = os.path.join(ROOT, "data", "embeddings")
QDIR = os.path.join(ROOT, "data", "queries")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", help="один вопрос")
    ap.add_argument("--file", help="файл с вопросами, по одному в строке")
    ap.add_argument("--chunks", default="base")
    ap.add_argument("--model", default="bge-m3", help="модель для плотного поиска, если посчитана")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(QDIR, "manual_draft.jsonl"))
    args = ap.parse_args()

    chunks = [json.loads(l) for l in open(os.path.join(CHUNKDIR, f"{args.chunks}.jsonl"),
                                          encoding="utf-8")]
    by_id = {c["chunk_id"]: c for c in chunks}
    ids = [c["chunk_id"] for c in chunks]
    bm25 = BM25Index.build(ids, [c["text"] for c in chunks])

    dense = None
    emb = os.path.join(EMBDIR, args.chunks, args.model)
    if os.path.exists(os.path.join(emb, "vectors.npy")):
        from rufin.retrieval.dense import MODELS, DenseIndex, Encoder
        dense = (DenseIndex.from_files(MODELS[args.model],
                                       os.path.join(emb, "vectors.npy"),
                                       os.path.join(emb, "ids.txt")),
                 Encoder(MODELS[args.model]))

    questions = []
    if args.query:
        questions.append(args.query)
    if args.file:
        # строки, начинающиеся с решётки, — пояснения в файле, а не вопросы
        questions += [l.strip() for l in open(args.file, encoding="utf-8")
                      if l.strip() and not l.lstrip().startswith("#")]
    if not questions:
        ap.error("нужен --query или --file")

    drafts = []
    for n, q in enumerate(questions):
        runs = [bm25.search(q, 50)]
        if dense:
            index, enc = dense
            runs.append(index.search_vectors(enc.encode([q], is_query=True), 50)[0])
        found = rrf(runs, top=args.top) if len(runs) > 1 else runs[0][:args.top]
        print(f"\n=== {q}")
        for i, (cid, _) in enumerate(found, start=1):
            c = by_id[cid]
            body = c["text"].split("\n\n", 1)[-1].replace("\n", " ")
            print(f"  [{i}] {c['number']} от {c['date']}  {c.get('section', '')[:50]}  "
                  f"п. {c.get('units', '')}")
            print(f"      {body[:190]}")
        drafts.append({"query_id": f"man{n:04d}", "origin": "ручной", "text": q,
                       "gold_chunk_id": "", "варианты": [cid for cid, _ in found]})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for d in drafts:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"\nзаготовка: {os.path.relpath(args.out, ROOT)} — проставьте gold_chunk_id "
          f"из списка «варианты» и сохраните как data/queries/manual.jsonl")


if __name__ == "__main__":
    main()
