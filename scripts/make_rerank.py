#!/usr/bin/env python3
"""Переранжирование гибридной выдачи кросс-энкодером — на этой машине.

Кросс-энкодер читает запрос и фрагмент вместе, поэтому качество выше
би-энкодера, но считать его можно только на коротком списке: пара
«запрос — фрагмент» прогоняется через модель целиком.

Считается локально намеренно. Модель уже лежит в кэше, скачивать нечего,
а значит отпадают и очередь за видеокартой, и ограничения HuggingFace,
на которых проект уже терял часы.

Результат пишется по мере готовности, и при повторном запуске уже
посчитанные запросы пропускаются: прерванный прогон не начинается заново.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNDIR = os.path.join(ROOT, "data", "runs")
CHUNKDIR = os.path.join(ROOT, "data", "chunks")

TOP = 50


def peak_gb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1073741824 if sys.platform == "darwin" else raw / 1048576


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base")
    ap.add_argument("--chunks", default="base")
    ap.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="для пробы: сколько запросов")
    args = ap.parse_args()

    hybrid_path = os.path.join(RUNDIR, f"{args.config}__hybrid.jsonl")
    if not os.path.exists(hybrid_path):
        raise SystemExit("нет гибридной выдачи: сначала scripts/make_hybrid.py")
    hybrid = {}
    with open(hybrid_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            hybrid[r["query_id"]] = r["ranked"][:TOP]

    queries = {}
    with open(os.path.join(ROOT, "data", "queries", "queries.jsonl"), encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries[q["query_id"]] = q["text"]

    texts = {}
    with open(os.path.join(CHUNKDIR, f"{args.chunks}.jsonl"), encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            texts[c["chunk_id"]] = c["text"]
    print(f"фрагментов в памяти: {len(texts)}, пик {peak_gb():.2f} ГБ", flush=True)

    out_path = os.path.join(RUNDIR, f"{args.config}__hybrid-rerank.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["query_id"])
                except Exception:  # noqa: BLE001
                    pass
        print(f"уже посчитано ранее: {len(done)} запросов", flush=True)

    todo = [q for q in hybrid if q not in done]
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        print("всё посчитано")
        return
    print(f"к переранжированию: {len(todo)} запросов по {TOP} кандидатов "
          f"= {len(todo) * TOP} пар", flush=True)

    from rufin.retrieval.rerank import Reranker
    rr = Reranker(model_path=args.model, batch_size=args.batch_size)
    print(f"модель поднята на {rr.device}, пик {peak_gb():.2f} ГБ", flush=True)

    t0 = time.monotonic()
    with open(out_path, "a", encoding="utf-8") as out:
        for n, qid in enumerate(todo, start=1):
            cand = [(c, 0.0) for c in hybrid[qid]]
            ranked, _ = rr.rerank(queries[qid], cand, texts, top=TOP)
            out.write(json.dumps({"query_id": qid, "ranked": [c for c, _ in ranked]},
                                 ensure_ascii=False) + "\n")
            out.flush()
            if n % 10 == 0 or n == len(todo):
                spent = time.monotonic() - t0
                left = spent / n * (len(todo) - n)
                print(f"   {n}/{len(todo)}, {spent:.0f} с, осталось ~{left / 60:.0f} мин, "
                      f"пик {peak_gb():.2f} ГБ", flush=True)

    print(f"\nготово: {os.path.relpath(out_path, ROOT)}")
    print(f"время {(time.monotonic() - t0) / 60:.1f} мин, пик памяти {peak_gb():.2f} ГБ")


if __name__ == "__main__":
    main()
