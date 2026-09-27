#!/usr/bin/env python3
"""Только переранжирование: кросс-энкодер поверх готовой гибридной выдачи.

Отдельный маленький прогон вместо повторения всего этапа B. BM25, плотный
поиск и гибрид уже посчитаны и лежат в проекте, поэтому здесь не нужны
ни нарезка корпуса, ни пять моделей — одна модель и десять тысяч пар.
Меньше шагов, меньше мест, где можно упасть.

На вход подаются готовая гибридная выдача и фрагменты базовой нарезки;
и то, и другое весит десятки мегабайт. Результат — переупорядоченная
выдача, файл в сотни килобайт.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gpu_common as common     # noqa: E402
import gpu_search as S          # noqa: E402


def read_jsonl(path: str) -> list[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hybrid", default=None, help="выдача гибрида; ищется сама")
    ap.add_argument("--chunks", default=None, help="фрагменты базовой нарезки; ищутся сами")
    ap.add_argument("--queries", default=None, help="набор запросов; ищется сам")
    ap.add_argument("--out", default="/kaggle/working/runs")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="для пробы")
    args = ap.parse_args()

    device = common.pick_device()
    print(f"устройство: {device}", flush=True)

    hybrid_path = args.hybrid or common.find_file("base__hybrid.jsonl")
    # find_file сам учитывает, что Kaggle распаковывает архивы при создании
    # датасета: искать оба имени руками больше не нужно
    chunks_path = (args.chunks or common.find_file("chunks_base.jsonl")
                   or common.find_file("base.jsonl"))
    queries_path = args.queries or common.find_file("queries.jsonl")
    for name, path in (("гибридная выдача", hybrid_path), ("фрагменты", chunks_path),
                       ("запросы", queries_path)):
        if path is None:
            raise SystemExit(f"не найдено: {name}")
        print(f"{name}: {path}", flush=True)

    hybrid = {r["query_id"]: r["ranked"][:S.TOP] for r in read_jsonl(hybrid_path)}
    queries = {q["query_id"]: q["text"] for q in read_jsonl(queries_path)}
    texts = {c["chunk_id"]: c["text"] for c in read_jsonl(chunks_path)}
    print(f"запросов {len(hybrid)}, фрагментов {len(texts)}", flush=True)

    todo = list(hybrid)
    if args.limit:
        todo = todo[:args.limit]
    print(f"пар к обсчёту: {len(todo) * S.TOP}", flush=True)

    t0 = time.time()
    ranked = S.rerank_runs([{"query_id": q, "text": queries[q]} for q in todo],
                           {q: [(c, 0.0) for c in hybrid[q]] for q in todo},
                           texts, batch_size=args.batch_size, device=device)
    os.makedirs(args.out, exist_ok=True)
    path = S.save_run(args.out, "base", "hybrid-rerank", ranked)
    print(f"\nготово за {(time.time() - t0) / 60:.1f} мин -> {path}", flush=True)
    print(f"размер {os.path.getsize(path) / 1024:.0f} КБ", flush=True)


if __name__ == "__main__":
    main()
