#!/usr/bin/env python3
"""Подготовка живых запросов из «Разъяснений» к разметке.

Вопросы собраны скриптом collect_explanations.py. Здесь для каждого
подбираются кандидаты в эталонный фрагмент и готовится лист для проверки
человеком: решение, какой фрагмент считать ответом, остаётся за ним.

Кандидаты берутся двумя путями. Основной: в заголовке темы у Банка России
указаны пункты акта — «Ведение кредитного досье (3.1.3, 3.1.5)», — и фрагмент
находится по началу пункта в тексте. Запасной, когда пунктов нет вовсе:
обычный поиск BM25 по корпусу.

Отдельный случай — пункты названы, но в корпусе их нет. Так бывает потому,
что «Вестник» печатает акт в редакции на дату принятия, а разъяснения Банка
России относятся к действующей: например, пункт 4.11 Положения 590-П добавлен
поздним Указанием, и в нашем тексте его не существует. Такой вопрос ответа
в корпусе не имеет, и подбирать ему кандидатов поиском нельзя: получатся
правдоподобные, но неверные. Вопрос отбрасывается.

Вопросы разносятся по темам и по актам: из одной темы берётся не больше
нескольких, из одного акта — не больше заданной доли. Без этого весь живой
набор оказывается про 590-П: по нему у Банка России разъяснений больше всего.

Отбирается с запасом: часть вопросов не имеет ответа в корпусе, и человек
их отбракует.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin.retrieval.bm25 import BM25Index     # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDIR = os.path.join(ROOT, "data", "queries")
CHUNKDIR = os.path.join(ROOT, "data", "chunks")

MAX_CANDIDATES = 4


def body(chunk: dict) -> str:
    return chunk["text"].split("\n\n", 1)[-1] if "\n\n" in chunk["text"] else chunk["text"]


def find_by_clause(chunks: list[dict], clause: str) -> list[dict]:
    """Фрагменты, где пункт начинается: строка вида «3.1.3. ...»."""
    pat = re.compile(r"(?:^|\n)\s*" + re.escape(clause) + r"\.\s")
    return [c for c in chunks if pat.search(body(c))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=35)
    ap.add_argument("--per-topic", type=int, default=2)
    ap.add_argument("--per-act", type=int, default=12,
                    help="не больше стольких вопросов на один акт")
    ap.add_argument("--chunks", default="base")
    ap.add_argument("--questions", default=os.path.join(QDIR, "explan_questions.jsonl"))
    args = ap.parse_args()

    chunks = [json.loads(l) for l in open(os.path.join(CHUNKDIR, f"{args.chunks}.jsonl"),
                                          encoding="utf-8")]
    by_id = {c["chunk_id"]: c for c in chunks}
    by_act: dict[str, list[dict]] = collections.defaultdict(list)
    for c in chunks:
        by_act[c["number"]].append(c)

    questions = [json.loads(l) for l in open(args.questions, encoding="utf-8") if l.strip()]
    print(f"вопросов собрано: {len(questions)}, фрагментов в корпусе: {len(chunks)}")

    # сперва те, где акт назван и есть пункты: у них кандидаты точные
    def priority(q: dict) -> int:
        act_known = any(a in by_act for a in q["acts"])
        return (0 if act_known and q["clauses"] else 1 if act_known else 2)

    questions.sort(key=lambda q: (priority(q), -len(q["text"])))

    bm25 = None
    skipped_edition = 0
    picked: list[dict] = []
    per_topic: collections.Counter = collections.Counter()
    per_act: collections.Counter = collections.Counter()
    for q in questions:
        if len(picked) >= args.target:
            break
        key = (q["section"], q["topic"])
        if per_topic[key] >= args.per_topic:
            continue

        act = next((a for a in q["acts"] if a in by_act), None)
        if act and per_act[act] >= args.per_act:
            continue
        cands: dict[str, dict] = {}
        how = ""
        if act and q["clauses"]:
            for cl in q["clauses"]:
                for c in find_by_clause(by_act[act], cl):
                    cands.setdefault(c["chunk_id"], c)
            how = f"по пунктам {', '.join(q['clauses'])} акта {act}"
        if act and q["clauses"] and not cands:
            # пункты названы, но в этой редакции акта их нет
            skipped_edition += 1
            continue
        if not cands:
            if bm25 is None:
                print("   пункты не указаны — строю индекс BM25 для запасного поиска")
                bm25 = BM25Index.build([c["chunk_id"] for c in chunks],
                                       [c["text"] for c in chunks])
            pool = by_act[act] if act else None
            for cid, _ in bm25.search(q["text"], 30):
                c = by_id[cid]
                if pool is not None and c["number"] != act:
                    continue
                cands.setdefault(cid, c)
                if len(cands) >= MAX_CANDIDATES:
                    break
            how = "поиском BM25" + (f" внутри акта {act}" if act else " по всему корпусу")
        if not cands:
            continue

        per_topic[key] += 1
        chosen_act = list(cands.values())[0]["number"]
        if per_act[chosen_act] >= args.per_act:
            continue
        per_act[chosen_act] += 1
        picked.append({"q": q, "candidates": list(cands.values())[:MAX_CANDIDATES], "how": how})

    # лист для проверки человеком
    rows = []
    for i, item in enumerate(picked):
        q = item["q"]
        for n, c in enumerate(item["candidates"], start=1):
            rows.append({
                "query_id": f"exp{i:04d}",
                "vybor_1_2_3_4_ili_0": "",
                "vopros": q["text"] if n == 1 else "",
                "variant": n,
                "chunk_id": c["chunk_id"],
                "akt": f"{c['number']} от {c['date']}",
                "punkty": c.get("units", ""),
                "razdel": c.get("section", "")[:60],
                "nachalo_fragmenta": body(c)[:220].replace("\n", " "),
                "kak_podobran": item["how"] if n == 1 else "",
                "tema_CB": q["topic"][:80] if n == 1 else "",
                "istochnik": q["url"] if n == 1 else "",
            })

    path = os.path.join(QDIR, "explan_candidates.tsv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    by_act_count = collections.Counter(item["candidates"][0]["number"] for item in picked)
    by_how = collections.Counter("по пунктам" if item["how"].startswith("по пунктам")
                                 else "поиском" for item in picked)
    print(f"\nотобрано вопросов: {len(picked)}")
    print(f"  отброшено: пункты названы, но в этой редакции акта их нет — {skipped_edition}")
    print(f"  как подобраны кандидаты: {dict(by_how)}")
    print(f"  по актам: {dict(by_act_count.most_common(6))}")
    print(f"  тем задействовано: {len(per_topic)}")
    print(f"  строк в листе проверки: {len(rows)}")
    print(f"\nлист: {os.path.relpath(path, ROOT)}")
    print("в колонке «vybor_1_2_3_4_ili_0» поставьте номер подходящего варианта")
    print("или 0, если ни один не подходит и вопрос надо выбросить")


if __name__ == "__main__":
    main()
