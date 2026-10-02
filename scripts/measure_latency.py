#!/usr/bin/env python3
"""Замеры задержки поиска — на маке, где система и работает.

Всё остальное тяжёлое считается на видеокарте Kaggle, но задержку там мерить
бессмысленно: пользователь работает не на T4. Поэтому замеры снимаются здесь,
и в отчёте указывается, на каком железе.

Методика: три прогона на прогрев, затем десять замеров на каждый запрос,
из которых берётся медиана; по запросам считаются медиана и 95-й перцентиль.
Прогрев нужен, чтобы не измерять загрузку весов вместо самого поиска.

Память. Конфигурации меряются по одной и модели выгружаются между ними:
на восьми гигабайтах два энкодера одновременно не помещаются. Перед каждым
шагом печатается пик памяти процесса — если он подбирается к четырём
гигабайтам, шаг лучше запустить отдельным вызовом через --only.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import platform
import resource
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin.retrieval.bm25 import BM25Index  # noqa: E402
from rufin.retrieval.hybrid import rrf  # noqa: E402
from rufin.retrieval.model_specs import MODELS  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHUNKDIR = os.path.join(ROOT, "data", "chunks")
EMBDIR = os.path.join(ROOT, "data", "embeddings")
QDIR = os.path.join(ROOT, "data", "queries")
RAW = os.path.join(ROOT, "docs", "raw")

TOP_RETRIEVE = 50
TOP_REPORT = 10


def peak_gb() -> float:
    """Пик памяти процесса. На macOS ru_maxrss возвращается в байтах."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1073741824 if sys.platform == "darwin" else raw / 1048576


def machine() -> str:
    return f"{platform.system()} {platform.machine()}, {platform.processor() or 'arm64'}"


def measure(fn, keys: list[str], warmup: int = 3, runs: int = 10, sample: int = 20,
            seed: int = 5) -> dict:
    rng = np.random.default_rng(seed)
    pick = [keys[i] for i in rng.choice(len(keys), size=min(sample, len(keys)), replace=False)]
    for _ in range(warmup):
        for k in pick:
            fn(k)
    per_query = []
    for k in pick:
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            fn(k)
            times.append((time.perf_counter() - t0) * 1000)
        per_query.append(statistics.median(times))
    per_query.sort()
    i95 = min(len(per_query) - 1, int(round(0.95 * (len(per_query) - 1))))
    return {"median_ms": round(statistics.median(per_query), 1),
            "p95_ms": round(per_query[i95], 1), "queries": len(pick),
            "runs": runs, "warmup": warmup}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="base")
    ap.add_argument("--queries", default=os.path.join(QDIR, "queries.jsonl"))
    ap.add_argument("--dense", default="bge-m3", help="модель для плотного поиска и гибрида")
    ap.add_argument("--model-path", default=None,
                    help="откуда брать веса вместо пути из model_specs. Нужно "
                         "дообученной модели до публикации на HuggingFace: "
                         "задержка меряется по весам, приехавшим с видеокарты, "
                         "а не по тому, чего ещё нет в сети")
    ap.add_argument("--rerank-model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--only", nargs="*", default=None,
                    help="что мерить: bm25 dense hybrid rerank")
    ap.add_argument("--sample", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(RAW, exist_ok=True)
    chunks = [json.loads(l) for l in open(os.path.join(CHUNKDIR, f"{args.chunks}.jsonl"),
                                          encoding="utf-8")]
    ids = [c["chunk_id"] for c in chunks]
    texts = {c["chunk_id"]: c["text"] for c in chunks}
    queries = [json.loads(l) for l in open(args.queries, encoding="utf-8") if l.strip()]
    qtext = {q["query_id"]: q["text"] for q in queries}
    qids = list(qtext)

    def want(name: str) -> bool:
        return args.only is None or name in args.only

    report: list[str] = []
    result: dict = {"machine": machine(), "chunks": args.chunks, "chunk_count": len(ids),
                    "latency": {}, "index": {}}

    def say(s: str = "") -> None:
        print(s, flush=True)
        report.append(s)

    say(f"машина: {machine()}")
    say(f"нарезка: {args.chunks}, фрагментов {len(ids)}, запросов {len(qids)}")
    say()

    bm25 = None
    if want("bm25") or want("hybrid") or want("rerank"):
        bm25 = BM25Index.build(ids, [texts[i] for i in ids])
        result["index"]["bm25"] = {"build_seconds": round(bm25.build_seconds, 1),
                                   "size_mb": round(bm25.size_mb, 1),
                                   "vocab": len(bm25.vocab)}
        say(f"BM25: построен за {bm25.build_seconds:.0f} с, индекс {bm25.size_mb:.0f} МБ, "
            f"словарь {len(bm25.vocab)} слов, пик памяти {peak_gb():.2f} ГБ")
        if want("bm25"):
            result["latency"]["bm25"] = measure(
                lambda k: bm25.search(qtext[k], TOP_REPORT), qids, sample=args.sample)

    index = enc = None
    # Урезанные размерности матрёшки это та же модель с другим полем
    # truncate_dim, и путь к весам у них общий: подмена пути должна
    # распространяться на все три записи сразу.
    spec = MODELS[args.dense]
    if args.model_path:
        spec = dataclasses.replace(spec, path=args.model_path)
        say(f"веса взяты из {args.model_path} вместо {MODELS[args.dense].path}")
    emb = os.path.join(EMBDIR, args.chunks, args.dense)
    have_dense = os.path.exists(os.path.join(emb, "vectors.npy"))
    if (want("dense") or want("hybrid") or want("rerank")) and have_dense:
        from rufin.retrieval.dense import DenseIndex, Encoder
        index = DenseIndex.from_files(spec, os.path.join(emb, "vectors.npy"),
                                      os.path.join(emb, "ids.txt"))
        enc = Encoder(spec)
        result["index"][f"dense:{args.dense}"] = {"size_mb": round(index.size_mb, 1),
                                                  "dim": int(index.vectors.shape[1]),
                                                  "device": enc.device}
        say(f"плотный поиск {args.dense}: матрица {index.size_mb:.0f} МБ, "
            f"энкодер на {enc.device}, пик памяти {peak_gb():.2f} ГБ")
        if want("dense"):
            result["latency"][f"dense:{args.dense}"] = measure(
                lambda k: index.search_vectors(enc.encode([qtext[k]], is_query=True),
                                               TOP_REPORT), qids, sample=args.sample)
        if want("hybrid"):
            result["latency"]["hybrid"] = measure(
                lambda k: rrf([bm25.search(qtext[k], TOP_RETRIEVE),
                               index.search_vectors(enc.encode([qtext[k]], is_query=True),
                                                    TOP_RETRIEVE)[0]], top=TOP_RETRIEVE),
                qids, sample=args.sample)
    elif not have_dense:
        say(f"матрицы эмбеддингов нет ({args.chunks}/{args.dense}) — "
            f"плотный поиск и гибрид не меряются")

    if want("rerank") and index is not None:
        from rufin.retrieval.rerank import Reranker
        rr = Reranker(model_path=args.rerank_model)
        say(f"реранкер {rr.model_path} на {rr.device}, пик памяти {peak_gb():.2f} ГБ")

        def hybrid_then_rerank(k: str):
            cand = rrf([bm25.search(qtext[k], TOP_RETRIEVE),
                        index.search_vectors(enc.encode([qtext[k]], is_query=True),
                                             TOP_RETRIEVE)[0]], top=TOP_RETRIEVE)
            return rr.rerank(qtext[k], cand, texts, top=TOP_REPORT)

        # реранкер на порядок дороже поиска, поэтому выборка меньше
        result["latency"]["hybrid+rerank"] = measure(
            hybrid_then_rerank, qids, sample=min(args.sample, 10))
        result["index"]["reranker"] = {"model": rr.model_path, "device": rr.device,
                                      "candidates": TOP_RETRIEVE}
        del rr
        gc.collect()

    say()
    say(f"{'конфигурация':<18} {'медиана, мс':>12} {'p95, мс':>10}   методика")
    for cfg, lat in result["latency"].items():
        say(f"{cfg:<18} {lat['median_ms']:>12.1f} {lat['p95_ms']:>10.1f}   "
            f"{lat['queries']} запросов, {lat['warmup']} прогрева + {lat['runs']} замеров")
    say()
    say("индексы:")
    for k, v in result["index"].items():
        say(f"   {k:<18} {v}")
    result["peak_memory_gb"] = round(peak_gb(), 2)
    say(f"\nпик памяти процесса: {result['peak_memory_gb']:.2f} ГБ")

    with open(os.path.join(ROOT, "data", "results", f"latency_{args.chunks}.json"),
              "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    with open(os.path.join(RAW, f"latency_{args.chunks}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    say(f"сырой вывод: docs/raw/latency_{args.chunks}.txt")


if __name__ == "__main__":
    main()
