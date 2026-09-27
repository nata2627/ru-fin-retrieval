#!/usr/bin/env python3
"""Превращение проверенного листа кандидатов в размеченные запросы.

Лист `explan_candidates.tsv` готовит prepare_explan_queries.py: на каждый
вопрос — до четырёх кандидатов в эталонный фрагмент. Человек проставляет
номер подходящего или 0, если ни один не подходит и вопрос надо выбросить.
Здесь выбор переносится в запросы.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDIR = os.path.join(ROOT, "data", "queries")

CHOICE = "vybor_1_2_3_4_ili_0"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default=os.path.join(QDIR, "explan_candidates.tsv"))
    ap.add_argument("--out", default=os.path.join(QDIR, "explan.jsonl"))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.tsv, encoding="utf-8"), delimiter="\t"))
    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        groups[r["query_id"]].append(r)

    out, rejected, unfilled = [], 0, 0
    for qid, group in groups.items():
        header = next((r for r in group if r["vopros"]), group[0])
        marks = {(r[CHOICE] or "").strip() for r in group} - {""}
        if not marks:
            unfilled += 1
            continue
        if marks == {"0"}:
            rejected += 1
            continue
        choice = next((m for m in marks if m != "0"), None)
        pick = next((r for r in group if r["variant"] == choice), None)
        if pick is None:
            unfilled += 1
            continue
        out.append({
            "query_id": qid,
            "origin": "вопрос из разъяснений Банка России",
            "text": header["vopros"],
            "gold_chunk_id": pick["chunk_id"],
            "source_url": header["istochnik"],
            "topic": header["tema_CB"],
        })

    with open(args.out, "w", encoding="utf-8") as f:
        for q in out:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"вопросов в листе: {len(groups)}")
    print(f"  принято:          {len(out)}")
    print(f"  отброшено (0):    {rejected}")
    print(f"  не заполнено:     {unfilled}")
    print(f"файл: {os.path.relpath(args.out, ROOT)}")


if __name__ == "__main__":
    main()
