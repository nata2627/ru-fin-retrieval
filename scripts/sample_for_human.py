#!/usr/bin/env python3
"""Выборка пар для ручной проверки разметки судьи.

Судья — это прибор, и у прибора есть погрешность. Мерится она согласием
с человеком на выборке в 150 пар (`scripts/kappa.py`).

Выборка стратифицирована по оценкам судьи. Случайная состояла бы почти
из одних нулей: в пуле на четырнадцать кандидатов релевантны один-два,
и каппа померила бы согласие в том, что случайный фрагмент нерелевантен.
Доли берутся равными, а не пропорциональными, — иначе двоек в выборке
окажется две-три.

Оценок судьи в листе нет. Разметка, сделанная после подглядывания, мерит
внушаемость, а не согласие.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin.annotation import stratified_sample  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDIR = os.path.join(ROOT, "data", "queries")
CHUNKDIR = os.path.join(ROOT, "data", "chunks")
REPORT = os.path.join(ROOT, "docs", "raw", "sample_for_human.txt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default=os.path.join(QDIR, "judge.tsv"))
    ap.add_argument("--queries", default=os.path.join(QDIR, "queries.jsonl"))
    ap.add_argument("--chunks", default="base")
    ap.add_argument("--size", type=int, default=150)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default=os.path.join(QDIR, "human_sample.tsv"))
    args = ap.parse_args()

    if not os.path.exists(args.judge):
        raise SystemExit(f"нет {os.path.relpath(args.judge, ROOT)}: сначала разметка "
                         f"судьёй на Kaggle")

    with open(args.judge, encoding="utf-8", newline="") as f:
        pairs = [r for r in csv.DictReader(f, delimiter="\t")
                 if (r.get("otsenka_0_1_2") or "").strip() in ("0", "1", "2")]
    выбрано = stratified_sample(pairs, "otsenka_0_1_2", args.size, seed=args.seed)

    queries = {}
    with open(args.queries, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                queries[q["query_id"]] = q["text"]
    нужны = {r["chunk_id"] for r in выбрано}
    chunks = {}
    with open(os.path.join(CHUNKDIR, f"{args.chunks}.jsonl"), encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c["chunk_id"] in нужны:
                chunks[c["chunk_id"]] = c

    rows = []
    for n, r in enumerate(sorted(выбрано, key=lambda x: (x["query_id"], x["chunk_id"])), 1):
        c = chunks.get(r["chunk_id"], {})
        текст = c.get("text", "")
        rows.append({
            "nomer": n,
            "otsenka_0_1_2": "",
            "zapros": queries.get(r["query_id"], ""),
            "akt": f"{c.get('number', '?')} от {c.get('date', '?')}",
            "razdel": (c.get("section") or "")[:60],
            "punkty": c.get("units", ""),
            "fragment": (текст.split("\n\n", 1)[-1] if "\n\n" in текст else текст),
            "query_id": r["query_id"],
            "chunk_id": r["chunk_id"],
        })

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    было = collections.Counter(r["otsenka_0_1_2"] for r in pairs)
    стало = collections.Counter(r["otsenka_0_1_2"] for r in выбрано)
    lines = [
        f"пар в разметке судьи: {len(pairs)}, распределение оценок: {dict(было)}",
        f"в выборке: {len(rows)}, распределение оценок судьи: {dict(стало)}",
        "оценок судьи в листе нет — колонка otsenka_0_1_2 пустая",
        "инструкция: docs/ANNOTATION_GUIDE.md",
        f"файл: {os.path.relpath(args.out, ROOT)}",
        "дальше: make kappa",
    ]
    for s in lines:
        print(s)
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
