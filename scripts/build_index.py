#!/usr/bin/env python3
"""Построение индексов: BM25 и матрицы эмбеддингов.

Индексация — разовая операция. Матрица эмбеддингов сохраняется файлом рядом
со списком идентификаторов, и дальше поиск работает по готовым векторам
без модели и без видеокарты.

Устройство и время индексации пишутся в meta.json: замеры времени без
указания железа сравнивать нельзя.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin.retrieval.bm25 import BM25Index              # noqa: E402
from rufin.retrieval.dense import MODELS, encode_corpus, pick_device  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHUNKDIR = os.path.join(ROOT, "data", "chunks")
EMBDIR = os.path.join(ROOT, "data", "embeddings")
REPORT = os.path.join(ROOT, "docs", "raw", "build_index.txt")


def machine() -> str:
    return f"{platform.system()} {platform.machine()}, Python {platform.python_version()}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="base", help="какая нарезка (имя файла в data/chunks)")
    ap.add_argument("--model", default="bge-m3", choices=list(MODELS) + ["bm25"])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    path = os.path.join(CHUNKDIR, f"{args.chunks}.jsonl")
    chunks = [json.loads(l) for l in open(path, encoding="utf-8")]
    if args.limit:
        chunks = chunks[:args.limit]
    ids = [c["chunk_id"] for c in chunks]
    texts = [c["text"] for c in chunks]

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    line = ""

    if args.model == "bm25":
        t0 = time.monotonic()
        idx = BM25Index.build(ids, texts)
        line = (f"BM25   нарезка={args.chunks} чанков={len(ids)} "
                f"построение={idx.build_seconds:.1f} с  ({machine()})")
    else:
        spec = MODELS[args.model]
        out_dir = os.path.join(EMBDIR, args.chunks, args.model)
        meta = encode_corpus(spec, texts, ids, out_dir,
                             batch_size=args.batch_size, device=args.device)
        meta["chunks_config"] = args.chunks
        meta["machine"] = machine()
        meta["batch_size"] = args.batch_size
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        line = (f"{spec.name:<9} нарезка={args.chunks} чанков={meta['chunks']} "
                f"dim={meta['dim']} устройство={meta['device']} "
                f"время={meta['seconds']:.0f} с ({meta['per_second']:.1f} чанк/с) "
                f"матрица={meta['size_mb']:.0f} МБ  ({machine()})")

    print(line)
    with open(REPORT, "a", encoding="utf-8") as f:
        f.write(line + "\n")


if __name__ == "__main__":
    main()
