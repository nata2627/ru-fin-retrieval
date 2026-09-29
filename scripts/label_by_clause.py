#!/usr/bin/env python3
"""Живые вопросы: привязка к акту, разметка по названным пунктам, маршрутизация.

Банк России публикует вопросы поднадзорных организаций (`cbr.ru/explan/`).
Собрано 1621; большинство — про федеральные законы, формы отчётности и письма,
то есть про документы, которых в корпусе нет, и искать на них ответ
бессмысленно. Здесь из собранного отбирается то, что к корпусу относится,
и каждому отобранному вопросу назначается один из трёх маршрутов.

**По пунктам.** В заголовке темы Банк России сам называет пункты акта —
«Ведение кредитного досье (3.1.3, 3.1.5)». Фрагмент находится по началу
пункта в тексте, и это внешнее свидетельство: оно не зависит от того,
что нашёл поиск. Так размечать надёжнее, чем пулом, и дешевле.

Почему эталон берётся из корпуса, а не из пула: если нужный фрагмент
не попал в первую десятку ни одной конфигурации, разметка внутри пула
объявила бы, что ответа нет. Провал поиска превратился бы в отсутствие
эталона и остался бы неизмеренным.

**В пул.** Акт понятен, пунктов нет. Эталон подбирается по объединённым
выдачам всех конфигураций — это `make pool` и разметка судьёй.

**Отброшен.** Пункты названы, но в нашем тексте акта их нет: «Вестник»
печатает акт в редакции на дату принятия, а разъяснения относятся
к действующей. Пункт 4.11 Положения 590-П добавлен поздним Указанием,
и у нас его не существует. Отправлять такой вопрос в пул нельзя: кандидаты
получатся правдоподобные и заведомо неверные.

Правило привязки к акту и чистка номеров пунктов от дат — в
`rufin.explanations`, там же измерения, на которых они держатся.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin.explanations import (  # noqa: E402
    ORIGIN,
    act_of,
    assign_ids,
    body_of,
    load_questions,
    starts_clause,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDIR = os.path.join(ROOT, "data", "queries")
CHUNKDIR = os.path.join(ROOT, "data", "chunks")
REPORT = os.path.join(ROOT, "docs", "raw", "label_by_clause.txt")

ROUTE_CLAUSE = "по пунктам"
ROUTE_POOL = "в пул"
ROUTE_DROPPED = "отброшен: пункта нет в этой редакции"


def corpus_numbers(path: str) -> set[str]:
    """Номера актов корпуса. Читается по строке: файл нарезки крупный."""
    numbers = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            numbers.add(json.loads(line)["number"])
    return numbers


def bodies_of_acts(path: str, wanted: set[str]) -> dict[str, list[dict]]:
    """Фрагменты только нужных актов: держать в памяти весь корпус незачем."""
    out: dict[str, list[dict]] = collections.defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            if chunk["number"] in wanted:
                out[chunk["number"]].append(
                    {"chunk_id": chunk["chunk_id"], "act_id": chunk["act_id"],
                     "body": body_of(chunk)})
    return out


def known_ids(*paths: str) -> dict[str, str]:
    """Уже выданные номера: текст вопроса -> query_id."""
    known: dict[str, str] = {}
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                q = json.loads(line)
                if q["query_id"].startswith("exp"):
                    known.setdefault(q["text"], q["query_id"])
    return known


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="base")
    ap.add_argument("--questions", default=os.path.join(QDIR, "explan_questions.jsonl"))
    ap.add_argument("--out-live", default=os.path.join(QDIR, "explan_live.jsonl"))
    ap.add_argument("--out-qrels", default=os.path.join(QDIR, "qrels_by_clause.tsv"))
    args = ap.parse_args()

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    chunks_path = os.path.join(CHUNKDIR, f"{args.chunks}.jsonl")
    questions = load_questions(args.questions)
    numbers = corpus_numbers(chunks_path)
    say(f"собрано вопросов: {len(questions)}, актов в корпусе: {len(numbers)}")

    tied = [(q, act_of(q, numbers)) for q in questions]
    tied = [(q, a) for q, a in tied if a]
    say(f"привязано к актам корпуса: {len(tied)}"
        f"  (остальные — про законы, формы отчётности и письма, их в корпусе нет)")

    by_act = bodies_of_acts(chunks_path, {a for _, a in tied})

    # Номера сохраняются за прежними текстами: под ними уже посчитаны выдачи
    # и проставлена разметка.
    known = known_ids(os.path.join(QDIR, "queries.jsonl"), args.out_live)
    ids = assign_ids([q["text"] for q, _ in tied], known)

    live: list[dict] = []
    rows: list[dict] = []
    stats: collections.Counter = collections.Counter()
    dropped_by_act: collections.Counter = collections.Counter()
    seen: set[str] = set()
    for q, act in tied:
        qid = ids[q["text"]]
        if qid in seen:                      # один и тот же текст в двух разделах
            stats["повтор текста"] += 1
            continue
        seen.add(qid)
        record = {"query_id": qid, "origin": ORIGIN, "text": q["text"], "act": act,
                  "clauses": q["clauses"], "topic": q["topic"], "section": q["section"],
                  "url": q["url"]}
        if not q["clauses"]:
            record["route"] = ROUTE_POOL
            stats[ROUTE_POOL] += 1
            live.append(record)
            continue
        found = {c["chunk_id"]: c for cl in q["clauses"]
                 for c in by_act[act] if starts_clause(c["body"], cl)}
        if not found:
            record["route"] = ROUTE_DROPPED
            stats[ROUTE_DROPPED] += 1
            dropped_by_act[act] += 1
            live.append(record)
            continue
        record["route"] = ROUTE_CLAUSE
        record["gold_chunk_ids"] = sorted(found)
        record["act_id"] = next(iter(found.values()))["act_id"]
        stats[ROUTE_CLAUSE] += 1
        live.append(record)
        osnovanie = f"пункты {', '.join(q['clauses'])} акта {act}"
        for cid in sorted(found):
            rows.append((qid, cid, 2, osnovanie))

    # Вопросы, отобранные прежде без привязки к акту, остаются в наборе:
    # разметку им подберёт пул, а выдачи по ним уже посчитаны.
    tied_texts = {q["text"] for q, _ in tied}
    carried = 0
    for text, qid in known.items():
        if text in tied_texts:
            continue
        live.append({"query_id": qid, "origin": ORIGIN, "text": text, "act": None,
                     "clauses": [], "topic": "", "section": "", "url": "",
                     "route": ROUTE_POOL})
        carried += 1

    with open(args.out_live, "w", encoding="utf-8") as f:
        for q in sorted(live, key=lambda r: r["query_id"]):
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    with open(args.out_qrels, "w", encoding="utf-8", newline="") as f:
        f.write("query-id\tcorpus-id\tscore\tosnovanie\n")
        for qid, cid, score, osnovanie in sorted(rows):
            f.write(f"{qid}\t{cid}\t{score}\t{osnovanie}\n")

    with_clauses = stats[ROUTE_CLAUSE] + stats[ROUTE_DROPPED]
    say()
    say(f"  {'вопросов с названными пунктами':<44} {with_clauses}")
    say(f"  {'из них размечено по пунктам':<44} {stats[ROUTE_CLAUSE]}")
    say(f"  {'из них отброшено (пункта нет в корпусе)':<44} {stats[ROUTE_DROPPED]}"
        f"  ({100 * stats[ROUTE_DROPPED] / max(1, with_clauses):.0f}%)")
    say(f"  {'вопросов без пунктов — в пул':<44} {stats[ROUTE_POOL]}")
    say(f"  {'вопросов без привязки к акту — в пул':<44} {carried}")
    if stats["повтор текста"]:
        say(f"  {'повторов текста пропущено':<44} {stats['повтор текста']}")
    say()

    usable = [q for q in live if q["route"] != ROUTE_DROPPED]
    acts = collections.Counter(q["act"] for q in usable if q["act"])
    top = acts.most_common(1)[0] if acts else ("—", 0)
    say(f"живых вопросов в наборе: {len(usable)} по {len(acts)} актам")
    say(f"доля самого частого акта ({top[0]}): "
        f"{100 * top[1] / max(1, len(usable)):.0f}%  — {top[1]} вопросов")
    say("распределение по актам: " +
        ", ".join(f"{a} {n}" for a, n in acts.most_common(8)) +
        (f", ещё {len(acts) - 8} актов" if len(acts) > 8 else ""))
    if dropped_by_act:
        say("отброшено по актам: " +
            ", ".join(f"{a} {n}" for a, n in dropped_by_act.most_common()))
    say()
    say(f"пар запрос-фрагмент размечено по пунктам: {len(rows)}")
    say(f"эталонных фрагментов на вопрос: "
        f"{len(rows) / max(1, stats[ROUTE_CLAUSE]):.1f} в среднем")
    say(f"файлы: {os.path.relpath(args.out_live, ROOT)}, "
        f"{os.path.relpath(args.out_qrels, ROOT)}")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
