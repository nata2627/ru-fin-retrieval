#!/usr/bin/env python3
"""Сборка текстов запросов для прогона конфигураций.

Этапу B на видеокарте нужны только тексты вопросов: он строит выдачи,
а эталоны ему не нужны. Разметка делается позже — по объединённому пулу
выдач всех конфигураций.

Почему именно в таком порядке. Если подбирать эталон из того, что нашёл
один метод, этот метод получает незаслуженное преимущество: его recall
по построению близок к единице, а выигрыш остальных занижен. В поиске
это давно решено объединением выдач: кандидаты собираются из всех
сравниваемых систем, и разметка ни одной из них не подыгрывает.

Синтетические запросы — исключение: их эталон известен по построению,
вопрос писался по конкретному фрагменту.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDIR = os.path.join(ROOT, "data", "queries")


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def read_questions_tsv(path: str, origin: str) -> list[dict]:
    """Вопросы из листа кандидатов: берём только тексты, выбор эталона — позже."""
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("vopros"):
                out.append({"query_id": row["query_id"], "text": row["vopros"],
                            "origin": origin})
    return out


def read_plain(path: str, origin: str, prefix: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append({"query_id": f"{prefix}{len(out):04d}", "text": s, "origin": origin})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(QDIR, "queries.jsonl"))
    args = ap.parse_args()

    queries: list[dict] = []
    for q in read_jsonl(os.path.join(QDIR, "synthetic.jsonl")):
        queries.append({"query_id": q["query_id"], "text": q["text"],
                        "origin": "синтетический", "gold_chunk_id": q["gold_chunk_id"]})
    queries += read_questions_tsv(os.path.join(QDIR, "explan_candidates.tsv"),
                                  "вопрос из разъяснений Банка России")
    queries += read_plain(os.path.join(QDIR, "manual_questions.txt"), "ручной", "man")

    seen = set()
    unique = []
    for q in queries:
        if q["query_id"] in seen:
            continue
        seen.add(q["query_id"])
        unique.append(q)

    os.makedirs(QDIR, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for q in unique:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    by_origin = collections.Counter(q["origin"] for q in unique)
    with_gold = sum(1 for q in unique if q.get("gold_chunk_id"))
    print(f"запросов: {len(unique)}")
    for k, v in by_origin.most_common():
        print(f"  {k:<40} {v:>4}")
    print(f"доля синтетических: {100 * by_origin['синтетический'] / len(unique):.0f}%")
    print(f"эталон уже известен (по построению): {with_gold}")
    print("остальным эталон подбирается после прогона, по объединённым выдачам")
    print(f"\nфайл: {os.path.relpath(args.out, ROOT)}")


if __name__ == "__main__":
    main()
