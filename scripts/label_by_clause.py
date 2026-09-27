#!/usr/bin/env python3
"""Разметка по указанию Банка России: эталон берётся из корпуса, а не из выдачи.

В заголовке темы раздела «Разъяснения» Банк России сам называет пункты акта,
к которым относится вопрос: «Ведение кредитного досье (3.1.3, 3.1.5)».
Это внешнее свидетельство, не зависящее от того, что нашёл поиск.

Почему это важнее, чем кажется. Пул кандидатов собирается из выдач, и если
нужный фрагмент не попал в первую десятку ни одной конфигурации, разметка
внутри пула объявит, что ответа нет. Провал поиска превратится в отсутствие
эталона и останется неизмеренным. Поэтому фрагмент ищется в корпусе
по номеру пункта, даже если ни одна конфигурация его не нашла.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDIR = os.path.join(ROOT, "data", "queries")
CHUNKDIR = os.path.join(ROOT, "data", "chunks")


def body(chunk: dict) -> str:
    return chunk["text"].split("\n\n", 1)[-1] if "\n\n" in chunk["text"] else chunk["text"]


def starts_clause(chunk: dict, clause: str) -> bool:
    """Фрагмент начинает пункт: строка вида «3.1.3. ...»."""
    return bool(re.search(r"(?:^|\n)\s*" + re.escape(clause) + r"\.\s", body(chunk)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="base")
    ap.add_argument("--out", default=os.path.join(QDIR, "qrels_by_clause.tsv"))
    args = ap.parse_args()

    by_act: dict[str, list[dict]] = collections.defaultdict(list)
    with open(os.path.join(CHUNKDIR, f"{args.chunks}.jsonl"), encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            by_act[c["number"]].append(c)

    hints = {}
    with open(os.path.join(QDIR, "explan_questions.jsonl"), encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            hints[q["text"]] = q

    queries = [json.loads(l) for l in open(os.path.join(QDIR, "queries.jsonl"), encoding="utf-8")]

    rows, stats = [], collections.Counter()
    for q in queries:
        if q.get("gold_chunk_id"):
            stats["эталон известен по построению"] += 1
            continue
        h = hints.get(q["text"])
        if not h or not h.get("clauses"):
            stats["без указания пунктов"] += 1
            continue
        act = next((a for a in h["acts"] if a in by_act), None)
        if act is None:
            stats["акт вне корпуса"] += 1
            continue
        found = []
        for clause in h["clauses"]:
            found += [c for c in by_act[act] if starts_clause(c, clause)]
        if not found:
            # пункта нет в этой редакции акта: «Вестник» печатает текст
            # на дату принятия, а разъяснения относятся к действующей
            stats["пункт отсутствует в этой редакции"] += 1
            continue
        uniq = {c["chunk_id"]: c for c in found}
        stats["размечено по указанию Банка России"] += 1
        for cid in uniq:
            rows.append({"query_id": q["query_id"], "chunk_id": cid, "score": 2,
                         "osnovanie": f"пункты {', '.join(h['clauses'])} акта {act}"})

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        f.write("query-id\tcorpus-id\tscore\tosnovanie\n")
        for r in rows:
            f.write(f"{r['query_id']}\t{r['chunk_id']}\t{r['score']}\t{r['osnovanie']}\n")

    for k, v in stats.most_common():
        print(f"  {k:<40} {v}")
    print(f"\nпар запрос-фрагмент размечено: {len(rows)}")
    print(f"файл: {os.path.relpath(args.out, ROOT)}")


if __name__ == "__main__":
    main()
