#!/usr/bin/env python3
"""Разбор запросов, на которых поиск провалился.

Берутся запросы, где ни один размеченный фрагмент не попал в первую десятку,
и раскладываются по причинам. Признаки считаются автоматически, но причина
предлагается, а не назначается: итоговая таблица — заготовка для ручного
разбора, её колонка «причина» уточняется руками.

Причины из плана работ:
  синонимия          слова запроса и ответа не пересекаются;
  длинный документ   нужный акт найден, но выдан не тот его кусок;
  нужно несколько    ответ собирается из нескольких фрагментов;
  нарезка            ответ разорван границей чанка;
  нет в корпусе      размеченного ответа в корпусе фактически нет;
  почти нашли        ответ есть, но ниже десятого места.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin.benchmark import read_qrels  # noqa: E402
from rufin.retrieval.text import tokenize  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNDIR = os.path.join(ROOT, "data", "runs")
QDIR = os.path.join(ROOT, "data", "queries")
CHUNKDIR = os.path.join(ROOT, "data", "chunks")
OUT = os.path.join(ROOT, "docs", "raw")

# доля слов запроса, встречающихся в эталонном фрагменте, ниже которой
# промах объясняется расхождением терминологии
LOW_OVERLAP = 0.25
# длина акта, начиная с которой он считается длинным
LONG_ACT_CHARS = 60_000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="файл выдачи из data/runs")
    ap.add_argument("--chunks", default="base")
    ap.add_argument("--queries", default=os.path.join(QDIR, "queries.jsonl"))
    ap.add_argument("--qrels", default=os.path.join(QDIR, "qrels.tsv"))
    ap.add_argument("--deep-run", default=None,
                    help="та же конфигурация с глубиной 50: отличает «почти нашли»")
    ap.add_argument("--take", type=int, default=30)
    args = ap.parse_args()

    chunks = {c["chunk_id"]: c for c in
              (json.loads(l) for l in open(os.path.join(CHUNKDIR, f"{args.chunks}.jsonl"),
                                           encoding="utf-8"))}
    act_chars: dict[str, int] = collections.Counter()
    act_chunks: dict[str, int] = collections.Counter()
    for c in chunks.values():
        act_chars[c["act_id"]] += len(c["text"])
        act_chunks[c["act_id"]] += 1

    queries = {q["query_id"]: q for q in
               (json.loads(l) for l in open(args.queries, encoding="utf-8"))}
    qrels = read_qrels(args.qrels)
    run = {r["query_id"]: r["ranked"] for r in
           (json.loads(l) for l in open(args.run, encoding="utf-8"))}
    deep = {}
    if args.deep_run and os.path.exists(args.deep_run):
        deep = {r["query_id"]: r["ranked"] for r in
                (json.loads(l) for l in open(args.deep_run, encoding="utf-8"))}

    failures = []
    for qid, ranked in run.items():
        rel = {c for c, g in qrels.get(qid, {}).items() if g > 0}
        if not rel or rel & set(ranked[:10]):
            continue
        q = queries[qid]
        gold_ids = sorted(rel)
        gold = chunks.get(gold_ids[0])
        gold_act = gold["act_id"] if gold else ""
        found_acts = {chunks[c]["act_id"] for c in ranked[:10] if c in chunks}

        qt = set(tokenize(q["text"]))
        gt = set(tokenize(gold["text"])) if gold else set()
        overlap = len(qt & gt) / len(qt) if qt else 0.0
        deep_pos = None
        if qid in deep:
            for i, c in enumerate(deep[qid], start=1):
                if c in rel:
                    deep_pos = i
                    break

        if gold is None:
            cause = "нет в корпусе"
        elif deep_pos:
            cause = "почти нашли"
        elif gold_act in found_acts:
            cause = "длинный документ" if act_chars[gold_act] > LONG_ACT_CHARS else "нарезка"
        elif len(rel) > 1:
            cause = "нужно несколько"
        elif overlap < LOW_OVERLAP:
            cause = "синонимия"
        else:
            cause = "разобрать руками"

        failures.append({
            "query_id": qid,
            "origin": q.get("origin", ""),
            "query": q["text"],
            "predpolozhennaya_prichina": cause,
            "prichina_posle_razbora": "",
            "нужный_акт_в_выдаче": "да" if gold_act in found_acts else "нет",
            "позиция_в_топ50": deep_pos or "",
            "пересечение_слов": round(overlap, 2),
            "размеченных_фрагментов": len(rel),
            "акт": f"{gold['number']} от {gold['date']}" if gold else "",
            "размер_акта_симв": act_chars.get(gold_act, 0),
            "чанков_в_акте": act_chunks.get(gold_act, 0),
            "эталон_начало": (gold["text"].split("\n\n", 1)[-1][:180].replace("\n", " ")
                              if gold else ""),
            "найдено_первым": (chunks[ranked[0]]["text"].split("\n\n", 1)[-1][:180].replace("\n", " ")
                               if ranked and ranked[0] in chunks else ""),
        })

    failures.sort(key=lambda f: f["query_id"])
    take = failures[:args.take]

    os.makedirs(OUT, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.run))[0]
    tsv = os.path.join(OUT, f"errors_{base}.tsv")
    with open(tsv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(take[0].keys()) if take else ["query_id"],
                           delimiter="\t")
        w.writeheader()
        w.writerows(take)

    print(f"запросов в выдаче: {len(run)}")
    print(f"провалов (ни один размеченный фрагмент не попал в топ-10): {len(failures)} "
          f"= {100 * len(failures) / max(1, len(run)):.0f}%")
    print(f"разобрано: {len(take)}")
    print()
    print("предположенные причины:")
    for cause, n in collections.Counter(f["predpolozhennaya_prichina"] for f in take).most_common():
        print(f"   {cause:<20} {n:>3}")
    print(f"\nтаблица для ручного разбора: docs/raw/errors_{base}.tsv")
    print("колонка «prichina_posle_razbora» заполняется руками")


if __name__ == "__main__":
    main()
