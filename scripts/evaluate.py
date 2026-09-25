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
из десяти» на выборке запросов. Устройство пишется в отчёт: замеры
времени без указания железа сравнивать нельзя.
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
from rufin.retrieval.dense import MODELS, DenseIndex, Encoder, pick_device  # noqa: E402
from rufin.retrieval.hybrid import rrf                          # noqa: E402
from rufin.retrieval.text import normalize_query                # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHUNKDIR = os.path.join(ROOT, "data", "chunks")
EMBDIR = os.path.join(ROOT, "data", "embeddings")
QDIR = os.path.join(ROOT, "data", "queries")
RUNDIR = os.path.join(ROOT, "data", "runs")
RESDIR = os.path.join(ROOT, "data", "results")

TOP_RETRIEVE = 50      # глубина выдачи, которая уходит в реранкер
TOP_REPORT = 10        # глубина, на которой считаются метрики


def machine() -> str:
    return f"{platform.system()} {platform.machine()}"


def measure_latency(fn, queries: list[str], warmup: int = 3, runs: int = 10,
                    sample: int = 20, seed: int = 5) -> dict:
    """Медиана и 95-й перцентиль задержки одного запроса.

    Методика: три прогона на прогрев, затем десять замеров на каждый запрос,
    из которых берётся медиана. Прогрев нужен, чтобы не измерять первую
    загрузку весов и прогрев кэшей.
    """
    rnd = np.random.default_rng(seed)
    pick = [queries[i] for i in rnd.choice(len(queries), size=min(sample, len(queries)),
                                           replace=False)]
    for _ in range(warmup):
        for q in pick:
            fn(q)
    per_query = []
    for q in pick:
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            fn(q)
            times.append((time.perf_counter() - t0) * 1000)
        per_query.append(statistics.median(times))
    per_query.sort()
    return {"median_ms": round(statistics.median(per_query), 1),
            "p95_ms": round(per_query[min(len(per_query) - 1, int(0.95 * len(per_query)))], 1),
            "queries": len(pick), "runs": runs, "device": machine()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="base")
    ap.add_argument("--queries", default=os.path.join(QDIR, "queries.jsonl"))
    ap.add_argument("--qrels", default=os.path.join(QDIR, "qrels.tsv"))
    ap.add_argument("--configs", nargs="*", default=None,
                    help="какие конфигурации прогнать; по умолчанию все")
    ap.add_argument("--dense", nargs="*", default=["bge-m3", "e5-large"])
    ap.add_argument("--hybrid-dense", default="bge-m3", help="какая модель идёт в гибрид")
    ap.add_argument("--no-latency", action="store_true")
    ap.add_argument("--normalize-queries", action="store_true",
                    help="снять канцелярит и раскрыть сокращения (абляция)")
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
    qtext = {q["query_id"]: (normalize_query(q["text"]) if args.normalize_queries else q["text"])
             for q in queries}

    tag = args.chunks + ("+norm" if args.normalize_queries else "")
    report: list[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        report.append(s)

    say(f"нарезка: {args.chunks}, чанков {len(ids)}")
    say(f"запросов с разметкой: {len(queries)}, машина: {machine()}")
    if args.normalize_queries:
        say("запросы нормализованы (снят канцелярит, раскрыты сокращения)")
    say()

    runs: dict[str, dict[str, list[str]]] = {}
    latency: dict[str, dict] = {}
    index_info: dict[str, dict] = {}

    want = args.configs
    def enabled(name: str) -> bool:
        return want is None or name in want

    # ---- BM25 ----
    bm25 = None
    if enabled("bm25") or enabled("hybrid") or enabled("hybrid+rerank"):
        t0 = time.monotonic()
        bm25 = BM25Index.build(ids, [texts[i] for i in ids])
        index_info["bm25"] = {"build_seconds": round(bm25.build_seconds, 1), "chunks": len(ids)}
        say(f"BM25 построен за {bm25.build_seconds:.1f} с")
    if enabled("bm25"):
        runs["bm25"] = {q["query_id"]: [c for c, _ in bm25.search(qtext[q["query_id"]], TOP_REPORT)]
                        for q in queries}
        if not args.no_latency:
            latency["bm25"] = measure_latency(lambda q: bm25.search(q, TOP_REPORT),
                                              list(qtext.values()))

    # ---- плотный поиск ----
    dense_runs_full: dict[str, dict[str, list[tuple[str, float]]]] = {}
    for name in args.dense:
        cfg = f"dense:{name}"
        emb_dir = os.path.join(EMBDIR, args.chunks, name)
        if not os.path.exists(os.path.join(emb_dir, "vectors.npy")):
            say(f"{cfg}: нет матрицы эмбеддингов ({emb_dir}), пропуск")
            continue
        if not (enabled(cfg) or (name == args.hybrid_dense
                                 and (enabled("hybrid") or enabled("hybrid+rerank")))):
            continue
        index = DenseIndex.from_files(MODELS[name], os.path.join(emb_dir, "vectors.npy"),
                                      os.path.join(emb_dir, "ids.txt"))
        meta = json.load(open(os.path.join(emb_dir, "meta.json"), encoding="utf-8"))
        index_info[cfg] = {"size_mb": round(index.size_mb, 1), "dim": index.vectors.shape[1],
                           "encode_seconds": meta.get("seconds"), "device": meta.get("device")}
        enc = Encoder(MODELS[name])
        qvec = enc.encode([qtext[q["query_id"]] for q in queries], is_query=True)
        found = index.search_vectors(qvec, TOP_RETRIEVE)
        dense_runs_full[name] = {q["query_id"]: f for q, f in zip(queries, found)}
        if enabled(cfg):
            runs[cfg] = {qid: [c for c, _ in f[:TOP_REPORT]]
                         for qid, f in dense_runs_full[name].items()}
            if not args.no_latency:
                latency[cfg] = measure_latency(
                    lambda q: index.search_vectors(enc.encode([q], is_query=True), TOP_REPORT),
                    list(qtext.values()))
        say(f"{cfg}: матрица {index.size_mb:.0f} МБ, размерность {index.vectors.shape[1]}, "
            f"посчитана на {meta.get('device')} за {meta.get('seconds')} с")
        if name != args.hybrid_dense:
            del index, enc
            gc.collect()

    # ---- гибрид ----
    hyb_full: dict[str, list[tuple[str, float]]] = {}
    if (enabled("hybrid") or enabled("hybrid+rerank")) and args.hybrid_dense in dense_runs_full:
        for q in queries:
            qid = q["query_id"]
            hyb_full[qid] = rrf([bm25.search(qtext[qid], TOP_RETRIEVE),
                                 dense_runs_full[args.hybrid_dense][qid]], top=TOP_RETRIEVE)
        if enabled("hybrid"):
            runs["hybrid"] = {qid: [c for c, _ in v[:TOP_REPORT]] for qid, v in hyb_full.items()}
            if not args.no_latency:
                index = DenseIndex.from_files(
                    MODELS[args.hybrid_dense],
                    os.path.join(EMBDIR, args.chunks, args.hybrid_dense, "vectors.npy"),
                    os.path.join(EMBDIR, args.chunks, args.hybrid_dense, "ids.txt"))
                enc = Encoder(MODELS[args.hybrid_dense])
                latency["hybrid"] = measure_latency(
                    lambda q: rrf([bm25.search(q, TOP_RETRIEVE),
                                   index.search_vectors(enc.encode([q], is_query=True),
                                                        TOP_RETRIEVE)[0]], top=TOP_RETRIEVE))

    # ---- гибрид с реранкером ----
    if enabled("hybrid+rerank") and hyb_full:
        from rufin.retrieval.rerank import Reranker
        rr = Reranker()
        say(f"реранкер: {rr.model_path} на {rr.device}")
        out = {}
        t0 = time.monotonic()
        for q in queries:
            qid = q["query_id"]
            ranked, _ = rr.rerank(qtext[qid], hyb_full[qid], texts, top=TOP_REPORT)
            out[qid] = [c for c, _ in ranked]
        runs["hybrid+rerank"] = out
        say(f"переранжировано {len(queries)} запросов за {time.monotonic() - t0:.0f} с "
            f"(по {TOP_RETRIEVE} кандидатов)")
        if not args.no_latency:
            latency["hybrid+rerank"] = measure_latency(
                lambda q: rr.rerank(q, hyb_full[queries[0]["query_id"]], texts, top=TOP_REPORT),
                list(qtext.values()), sample=10)

    # ---- метрики ----
    per_q = {name: M.per_query(run, qrels) for name, run in runs.items()}
    say()
    names = list(M.METRICS)
    say(f"{'конфигурация':<16} " + " ".join(f"{n:>22}" for n in names))
    for cfg, pq in per_q.items():
        cells = [str(M.bootstrap_ci(pq[n])) for n in names]
        say(f"{cfg:<16} " + " ".join(f"{c:>22}" for c in cells))

    if "bm25" in per_q and len(per_q) > 1:
        say()
        say("разница с BM25 (парный бутстрэп, в скобках доверительный интервал; "
            "p — перестановочный тест):")
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
                f"{lat['queries']} запросов, 3 прогрева + {lat['runs']} замеров, {lat['device']}")

    if index_info:
        say()
        say("индексы:")
        for cfg, info in index_info.items():
            say(f"   {cfg:<16} {info}")

    # ---- сохранение ----
    for cfg, run in runs.items():
        safe = cfg.replace(":", "-").replace("+", "-")
        with open(os.path.join(RUNDIR, f"{tag}__{safe}.jsonl"), "w", encoding="utf-8") as f:
            for qid, docs in run.items():
                f.write(json.dumps({"query_id": qid, "ranked": docs}, ensure_ascii=False) + "\n")

    result = {
        "chunks": args.chunks, "normalize_queries": args.normalize_queries,
        "queries": len(queries), "chunk_count": len(ids), "machine": machine(),
        "metrics": {cfg: {n: M.bootstrap_ci(pq[n]).__dict__ for n in names}
                    for cfg, pq in per_q.items()},
        "latency": latency, "index": index_info,
    }
    with open(os.path.join(RESDIR, f"{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    with open(os.path.join(ROOT, "docs", "raw", f"eval_{tag}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    say(f"\nрезультаты: data/results/{tag}.json, сырой вывод: docs/raw/eval_{tag}.txt")


if __name__ == "__main__":
    main()
