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


def pool(runs: dict[str, dict[str, list[str]]], qid: str, size: int) -> list[str]:
    """По очереди с каждой конфигурации: первые места, потом вторые, и так далее."""
    out: list[str] = []
    seen: set[str] = set()
    depth = 0
    lists = [r.get(qid, []) for r in runs.values()]
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
    ap.add_argument("--size", type=int, default=8, help="кандидатов на запрос")
    ap.add_argument("--out", default=os.path.join(QDIR, "pool_candidates.tsv"))
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
    need = [q for q in queries if not q.get("gold_chunk_id")]
    print(f"конфигураций в выдачах: {len(runs)} ({', '.join(runs)})")
    print(f"запросов всего {len(queries)}, эталон нужен для {len(need)}")

    rows = []
    for q in need:
        cands = pool(runs, q["query_id"], args.size)
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
    print("\nв колонке otsenka_0_1_2:")
    print("  2 — фрагмент прямо отвечает на вопрос")
    print("  1 — относится к делу, но ответа не содержит")
    print("  0 — не относится")
    print("если ни одному фрагменту не поставлена 2, запрос выбывает из набора")


if __name__ == "__main__":
    main()
