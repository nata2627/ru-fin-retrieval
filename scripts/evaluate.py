#!/usr/bin/env python3
"""Прогон поисковых конфигураций и подсчёт метрик.

Конфигурации:
  bm25            базовая линия;
  dense:<модель>  плотный поиск;
  hybrid          BM25 и плотный поиск, слитые через RRF;
  hybrid+rerank   то же плюс кросс-энкодер на первых 50.

Считаются Recall@1/5/10, MRR@10, NDCG@10 с бутстрэп-интервалами и парное
сравнение каждой конфигурации с BM25 на одних и тех же запросах.

Задержка меряется по методике «три прогона на прогрев, затем медиана
из десяти» на выборке запросов, с записью устройства: замеры времени
без указания железа сравнивать нельзя.

Выдача сохраняется на глубину 50, а не 10: разбору ошибок нужно знать,
нашёлся ли ответ хотя бы ниже десятого места.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin import metrics as M                                  # noqa: E402
from rufin.benchmark import read_qrels                          # noqa: E402
from rufin.retrieval.bm25 import BM25Index                      # noqa: E402
from rufin.retrieval.dense import MODELS, DenseIndex, Encoder   # noqa: E402
from rufin.retrieval.hybrid import rrf                          # noqa: E402
from rufin.retrieval.text import normalize_query                # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHUNKDIR = os.path.join(ROOT, "data", "chunks")
EMBDIR = os.path.join(ROOT, "data", "embeddings")
QDIR = os.path.join(ROOT, "data", "queries")
RUNDIR = os.path.join(ROOT, "data", "runs")
RESDIR = os.path.join(ROOT, "data", "results")

TOP_RETRIEVE = 50      # глубина выдачи: уходит в реранкер и в разбор ошибок
TOP_REPORT = 10        # глубина, на которой считаются метрики


def machine() -> str:
    return f"{platform.system()} {platform.machine()}"


def measure_latency(fn, keys: list[str], warmup: int = 3, runs: int = 10,
                    sample: int = 20, seed: int = 5) -> dict:
    """Медиана и 95-й перцентиль задержки одного запроса.

    Три прогона на прогрев, затем десять замеров на каждый запрос, из которых
    берётся медиана. Прогрев нужен, чтобы не измерять загрузку весов
    и прогрев кэшей вместо самого поиска.
    """
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
    idx95 = min(len(per_query) - 1, int(round(0.95 * (len(per_query) - 1))))
    return {"median_ms": round(statistics.median(per_query), 1),
            "p95_ms": round(per_query[idx95], 1),
            "queries": len(pick), "runs": runs, "warmup": warmup, "device": machine()}


def load_dense(chunks_cfg: str, model: str) -> tuple[DenseIndex, dict] | None:
    d = os.path.join(EMBDIR, chunks_cfg, model)
    if not os.path.exists(os.path.join(d, "vectors.npy")):
        return None
    index = DenseIndex.from_files(MODELS[model], os.path.join(d, "vectors.npy"),
                                  os.path.join(d, "ids.txt"))
    meta_path = os.path.join(d, "meta.json")
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
    return index, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="base")
    ap.add_argument("--queries", default=os.path.join(QDIR, "queries.jsonl"))
    ap.add_argument("--qrels", default=os.path.join(QDIR, "qrels.tsv"))
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--dense", nargs="*", default=["bge-m3", "e5-large"])
    ap.add_argument("--hybrid-dense", default="bge-m3")
    ap.add_argument("--rerank-model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--no-latency", action="store_true")
    ap.add_argument("--normalize-queries", action="store_true",
                    help="снять канцелярит и раскрыть сокращения (абляция)")
    ap.add_argument("--tag", default=None, help="имя набора результатов")
    args = ap.parse_args()

    for d in (RUNDIR, RESDIR, os.path.join(ROOT, "docs", "raw")):
        os.makedirs(d, exist_ok=True)

    chunks = [json.loads(l) for l in open(os.path.join(CHUNKDIR, f"{args.chunks}.jsonl"),
                                          encoding="utf-8")]
    texts = {c["chunk_id"]: c["text"] for c in chunks}
    ids = [c["chunk_id"] for c in chunks]

    queries = [json.loads(l) for l in open(args.queries, encoding="utf-8")]
    qrels = read_qrels(args.qrels)
    queries = [q for q in queries if q["query_id"] in qrels]
    qids = [q["query_id"] for q in queries]
    qtext = {q["query_id"]: (normalize_query(q["text"]) if args.normalize_queries else q["text"])
             for q in queries}

    tag = args.tag or (args.chunks + ("+norm" if args.normalize_queries else ""))
    report: list[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        report.append(s)

    say(f"нарезка: {args.chunks}, чанков {len(ids)}")
    say(f"запросов с разметкой: {len(queries)}, машина: {machine()}")
    if args.normalize_queries:
        say("запросы нормализованы: снят канцелярит, раскрыты сокращения")
    say()

    deep: dict[str, dict[str, list[str]]] = {}     # выдача на глубину 50
    latency: dict[str, dict] = {}
    index_info: dict[str, dict] = {}

    want = args.configs

    def enabled(name: str) -> bool:
        return want is None or name in want

    need_bm25 = enabled("bm25") or enabled("hybrid") or enabled("hybrid+rerank")
    need_hybrid = enabled("hybrid") or enabled("hybrid+rerank")

    # ---- BM25 ----
    bm25 = None
    if need_bm25:
        bm25 = BM25Index.build(ids, [texts[i] for i in ids])
        index_info["bm25"] = {"chunks": len(ids), "build_seconds": round(bm25.build_seconds, 1)}
        say(f"BM25: построен за {bm25.build_seconds:.0f} с")
        bm25_deep = {qid: [c for c, _ in bm25.search(qtext[qid], TOP_RETRIEVE)] for qid in qids}
        if enabled("bm25"):
            deep["bm25"] = bm25_deep
            if not args.no_latency:
                latency["bm25"] = measure_latency(
                    lambda k: bm25.search(qtext[k], TOP_REPORT), qids)

    # ---- плотный поиск ----
    dense_deep: dict[str, dict[str, list[tuple[str, float]]]] = {}
    for name in args.dense:
        cfg = f"dense:{name}"
        if not (enabled(cfg) or (name == args.hybrid_dense and need_hybrid)):
            continue
        loaded = load_dense(args.chunks, name)
        if loaded is None:
            say(f"{cfg}: матрицы эмбеддингов нет ({args.chunks}/{name}), пропуск")
            continue
        index, meta = loaded
        if len(index.ids) != len(ids):
            say(f"{cfg}: в матрице {len(index.ids)} векторов, в нарезке {len(ids)} чанков — "
                f"матрица посчитана на другой нарезке, пропуск")
            continue
        index_info[cfg] = {"size_mb": round(index.size_mb, 1), "dim": int(index.vectors.shape[1]),
                           "encode_seconds": meta.get("seconds"), "device": meta.get("device"),
                           "dtype": meta.get("dtype", "float32")}
        enc = Encoder(MODELS[name])
        qvec = enc.encode([qtext[q] for q in qids], is_query=True)
        found = index.search_vectors(qvec, TOP_RETRIEVE)
        dense_deep[name] = dict(zip(qids, found))
        if enabled(cfg):
            deep[cfg] = {qid: [c for c, _ in v] for qid, v in dense_deep[name].items()}
            if not args.no_latency:
                latency[cfg] = measure_latency(
                    lambda k, i=index, e=enc: i.search_vectors(
                        e.encode([qtext[k]], is_query=True), TOP_REPORT), qids)
        say(f"{cfg}: матрица {index.size_mb:.0f} МБ ({meta.get('dtype', 'float32')}), "
            f"размерность {index.vectors.shape[1]}, посчитана на {meta.get('device')} "
            f"за {meta.get('seconds')} с")
        if name == args.hybrid_dense:
            hybrid_index, hybrid_enc = index, enc
        else:
            del index, enc
            gc.collect()

    # ---- гибрид ----
    hyb_deep: dict[str, list[tuple[str, float]]] = {}
    if need_hybrid and args.hybrid_dense in dense_deep:
        for qid in qids:
            hyb_deep[qid] = rrf([[(c, 0.0) for c in bm25_deep[qid]],
                                 dense_deep[args.hybrid_dense][qid]], top=TOP_RETRIEVE)
        if enabled("hybrid"):
            deep["hybrid"] = {qid: [c for c, _ in v] for qid, v in hyb_deep.items()}
            if not args.no_latency:
                latency["hybrid"] = measure_latency(
                    lambda k: rrf([bm25.search(qtext[k], TOP_RETRIEVE),
                                   hybrid_index.search_vectors(
                                       hybrid_enc.encode([qtext[k]], is_query=True),
                                       TOP_RETRIEVE)[0]], top=TOP_RETRIEVE), qids)

    # ---- гибрид с реранкером ----
    if enabled("hybrid+rerank") and hyb_deep:
        from rufin.retrieval.rerank import Reranker
        rr = Reranker(model_path=args.rerank_model)
        say(f"реранкер: {rr.model_path} на {rr.device}")
        t0 = time.monotonic()
        out = {}
        for qid in qids:
            ranked, _ = rr.rerank(qtext[qid], hyb_deep[qid], texts, top=TOP_RETRIEVE)
            out[qid] = [c for c, _ in ranked]
        deep["hybrid+rerank"] = out
        say(f"переранжировано {len(qids)} запросов по {TOP_RETRIEVE} кандидатов "
            f"за {time.monotonic() - t0:.0f} с")
        index_info["hybrid+rerank"] = {"model": rr.model_path, "device": rr.device,
                                       "candidates": TOP_RETRIEVE}
        if not args.no_latency:
            latency["hybrid+rerank"] = measure_latency(
                lambda k: rr.rerank(qtext[k], hyb_deep[k], texts, top=TOP_REPORT),
                qids, sample=10)

    if not deep:
        say("ни одной конфигурации не прогнано")
        return

    # ---- метрики ----
    runs10 = {cfg: {qid: docs[:TOP_REPORT] for qid, docs in run.items()}
              for cfg, run in deep.items()}
    per_q = {cfg: M.per_query(run, qrels) for cfg, run in runs10.items()}
    names = list(M.METRICS)
    say()
    say(f"{'конфигурация':<16} " + " ".join(f"{n:>22}" for n in names))
    for cfg, pq in per_q.items():
        say(f"{cfg:<16} " + " ".join(f"{str(M.bootstrap_ci(pq[n])):>22}" for n in names))

    if "bm25" in per_q and len(per_q) > 1:
        say()
        say("разница с BM25, парный бутстрэп по запросам; p — перестановочный тест:")
        for cfg, pq in per_q.items():
            if cfg == "bm25":
                continue
            parts = []
            for n in ("Recall@5", "NDCG@10"):
                d, p = M.paired_diff_ci(pq[n], per_q["bm25"][n])
                mark = "" if d.lo <= 0 <= d.hi else " *"
                parts.append(f"{n} {d.mean:+.3f} [{d.lo:+.3f}; {d.hi:+.3f}] p={p:.3f}{mark}")
            say(f"   {cfg:<16} " + "   ".join(parts))
        say("   * интервал не накрывает ноль")

    if latency:
        say()
        say(f"{'конфигурация':<16} {'медиана, мс':>12} {'p95, мс':>10}   методика")
        for cfg, lat in latency.items():
            say(f"{cfg:<16} {lat['median_ms']:>12.1f} {lat['p95_ms']:>10.1f}   "
                f"{lat['queries']} запросов, {lat['warmup']} прогрева + {lat['runs']} замеров, "
                f"{lat['device']}")

    if index_info:
        say()
        say("индексы:")
        for cfg, info in index_info.items():
            say(f"   {cfg:<16} {info}")

    # ---- сохранение ----
    for cfg, run in deep.items():
        safe = cfg.replace(":", "-").replace("+", "-")
        with open(os.path.join(RUNDIR, f"{tag}__{safe}.jsonl"), "w", encoding="utf-8") as f:
            for qid, docs in run.items():
                f.write(json.dumps({"query_id": qid, "ranked": docs}, ensure_ascii=False) + "\n")

    result = {"chunks": args.chunks, "normalize_queries": args.normalize_queries,
              "queries": len(queries), "chunk_count": len(ids), "machine": machine(),
              "metrics": {cfg: {n: M.bootstrap_ci(pq[n]).__dict__ for n in names}
                          for cfg, pq in per_q.items()},
              "latency": latency, "index": index_info}
    with open(os.path.join(RESDIR, f"{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    with open(os.path.join(ROOT, "docs", "raw", f"eval_{tag}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    say(f"\nрезультаты: data/results/{tag}.json, сырой вывод: docs/raw/eval_{tag}.txt")


if __name__ == "__main__":
    main()
