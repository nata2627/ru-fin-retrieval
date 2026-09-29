#!/usr/bin/env python3
"""Объединение выдач всех конфигураций в пул кандидатов для разметки.

Эталон нельзя подбирать выдачей одного метода: тогда его полнота по
построению близка к единице, а выигрыш остальных занижен. Поэтому кандидаты
берутся по очереди из всех сравниваемых конфигураций — первое место каждой,
потом второе каждой, и так далее. Ни одна не получает преимущества.

Порядок кандидатов в листе перемешивается, а принадлежность к конфигурации
не показывается: иначе разметка невольно подстроится под то, что выдал
знакомый метод. Перемешивание детерминированное, от номера запроса,
чтобы лист воспроизводился.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import random
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNDIR = os.path.join(ROOT, "data", "runs")
QDIR = os.path.join(ROOT, "data", "queries")
CHUNKDIR = os.path.join(ROOT, "data", "chunks")


def body(chunk: dict) -> str:
    return chunk["text"].split("\n\n", 1)[-1] if "\n\n" in chunk["text"] else chunk["text"]


def load_runs(config: str) -> dict[str, dict[str, list[str]]]:
    runs: dict[str, dict[str, list[str]]] = {}
    for name in sorted(os.listdir(RUNDIR)):
        m = re.fullmatch(rf"{re.escape(config)}__(.+)\.jsonl", name)
        if not m:
            continue
        run: dict[str, list[str]] = {}
        with open(os.path.join(RUNDIR, name), encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                run[rec["query_id"]] = rec["ranked"]
        runs[m.group(1)] = run
    return runs


def pool(runs: dict[str, dict[str, list[str]]], qid: str, size: int,
         max_depth: int = 10) -> list[str]:
    """По очереди с каждой конфигурации: первые места, потом вторые, и так далее.

    Глубина ограничена: метрики считаются на первой десятке, и фрагмент,
    не попавший туда ни у одной конфигурации, на них не влияет. Зато всё,
    что хоть одна конфигурация ставит в первую десятку, обязано попасть
    в пул — иначе её результат будет занижен из-за неразмеченного ответа.
    """
    out: list[str] = []
    seen: set[str] = set()
    depth = 0
    lists = [r.get(qid, [])[:max_depth] for r in runs.values()]
    while len(out) < size and any(depth < len(l) for l in lists):
        for l in lists:
            if depth < len(l) and l[depth] not in seen:
                seen.add(l[depth])
                out.append(l[depth])
                if len(out) >= size:
                    break
        depth += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base")
    ap.add_argument("--queries", default=os.path.join(QDIR, "queries.jsonl"))
    ap.add_argument("--chunks", default="base")
    ap.add_argument("--size", type=int, default=14, help="кандидатов на запрос")
    ap.add_argument("--depth", type=int, default=10,
                    help="сколько верхних позиций брать с каждой выдачи")
    ap.add_argument("--out", default=os.path.join(QDIR, "pool_candidates.tsv"))
    ap.add_argument("--allow-partial", action="store_true",
                    help="собрать пул по неполным выдачам (занижает новые конфигурации)")
    args = ap.parse_args()

    runs = load_runs(args.config)
    if not runs:
        print(f"в data/runs нет выдач для нарезки «{args.config}» — их считает этап B")
        raise SystemExit(2)

    chunks = {}
    with open(os.path.join(CHUNKDIR, f"{args.chunks}.jsonl"), encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunks[c["chunk_id"]] = c

    queries = [json.loads(l) for l in open(args.queries, encoding="utf-8") if l.strip()]
    need = [q for q in queries
            if not (q.get("gold_chunk_ids") or q.get("gold_chunk_id"))]
    print(f"конфигураций в выдачах: {len(runs)} ({', '.join(runs)})")
    print(f"запросов всего {len(queries)}, эталон нужен для {len(need)}")

    # Пул, собранный по устаревшим выдачам, занижает всё, что появилось после
    # них: находок новых конфигураций в нём просто нет, и полнота этих
    # конфигураций упадёт не по заслугам. Заметить это по метрикам нельзя,
    # поэтому здесь отказ, а не предупреждение.
    covered = set.intersection(*(set(run) for run in runs.values()))
    отстали = [q["query_id"] for q in need if q["query_id"] not in covered]
    if отстали and not args.allow_partial:
        print(f"\nОТКАЗ: у {len(отстали)} запросов из {len(need)} нет выдачи "
              f"хотя бы у одной конфигурации.")
        print("Пул по неполным выдачам занижает конфигурации, которых в нём нет, "
              "и по метрикам этого не видно.")
        print("Нужен прогон этапа B по нынешнему queries.jsonl — см. kaggle/README.md.")
        print(f"Первые без выдачи: {', '.join(отстали[:8])}")
        print("Если пул нужен именно по тому, что посчитано, — `--allow-partial`.")
        raise SystemExit(2)
    if отстали:
        print(f"   --allow-partial: {len(отстали)} запросов без выдачи пропущены")
        need = [q for q in need if q["query_id"] in covered]

    rows = []
    for q in need:
        cands = pool(runs, q["query_id"], args.size, args.depth)
        if not cands:
            continue
        # перемешиваем, чтобы порядок не подсказывал ответ
        rnd = random.Random(q["query_id"])
        rnd.shuffle(cands)
        for n, cid in enumerate(cands, start=1):
            c = chunks.get(cid)
            if c is None:
                continue
            rows.append({
                "query_id": q["query_id"],
                "otsenka_0_1_2": "",
                "vopros": q["text"] if n == 1 else "",
                "variant": n,
                "chunk_id": cid,
                "akt": f"{c['number']} от {c['date']}",
                "punkty": c.get("units", ""),
                "razdel": c.get("section", "")[:60],
                "fragment": body(c)[:300].replace("\n", " "),
            })

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    per_q = collections.Counter(r["query_id"] for r in rows)
    print(f"\nлист разметки: {os.path.relpath(args.out, ROOT)}")
    print(f"  запросов: {len(per_q)}, строк: {len(rows)}, "
          f"кандидатов на запрос в среднем {len(rows) / max(1, len(per_q)):.1f}")
    # Шкала не пересказывается здесь: она одна на человека и на судью
    # и лежит в docs/ANNOTATION_GUIDE.md. Два пересказа разойдутся,
    # и каппа померит разницу в инструкциях, а не согласие.
    print("\nкак размечать — docs/ANNOTATION_GUIDE.md, колонка otsenka_0_1_2")
    print("если ни одному фрагменту не поставлена 2, запрос выбывает из набора")
    print("дальше: разметка судьёй на Kaggle (run_phase_c.py --judge), затем make judge")


if __name__ == "__main__":
    main()
