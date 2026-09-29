#!/usr/bin/env python3
"""Гибридная выдача из уже посчитанных: BM25 и плотный поиск через RRF.

Ни модели, ни видеокарты: RRF складывает обратные ранги, а ранги уже
известны из сохранённых выдач. Поэтому гибрид считается локально
за секунды, даже если сами выдачи получены на чужом железе.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin.retrieval.hybrid import rrf  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNDIR = os.path.join(ROOT, "data", "runs")


def load(path: str) -> dict[str, list[str]]:
    with open(path, encoding="utf-8") as f:
        return {r["query_id"]: r["ranked"] for r in (json.loads(l) for l in f if l.strip())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base")
    ap.add_argument("--dense", default="bge-m3")
    ap.add_argument("--top", type=int, default=50)
    args = ap.parse_args()

    sparse_path = os.path.join(RUNDIR, f"{args.config}__bm25.jsonl")
    dense_path = os.path.join(RUNDIR, f"{args.config}__dense-{args.dense}.jsonl")
    for p in (sparse_path, dense_path):
        if not os.path.exists(p):
            raise SystemExit(f"нет выдачи {os.path.relpath(p, ROOT)}")

    sparse, dense = load(sparse_path), load(dense_path)
    common = [q for q in sparse if q in dense]
    out_path = os.path.join(RUNDIR, f"{args.config}__hybrid.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for qid in common:
            merged = rrf([[(c, 0.0) for c in sparse[qid]],
                          [(c, 0.0) for c in dense[qid]]], top=args.top)
            f.write(json.dumps({"query_id": qid, "ranked": [c for c, _ in merged]},
                               ensure_ascii=False) + "\n")

    print(f"BM25: {len(sparse)} запросов, плотный ({args.dense}): {len(dense)}")
    print(f"гибрид посчитан для {len(common)} запросов -> "
          f"{os.path.relpath(out_path, ROOT)}")


if __name__ == "__main__":
    main()
