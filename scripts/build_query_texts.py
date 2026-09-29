#!/usr/bin/env python3
"""Сборка текстов запросов для прогона конфигураций.

Этапу B на видеокарте нужны только тексты вопросов: он строит выдачи,
а эталоны ему не нужны. Разметка делается позже — по объединённому пулу
выдач всех конфигураций.

Почему именно в таком порядке. Если подбирать эталон из того, что нашёл
один метод, этот метод получает незаслуженное преимущество: его полнота
по построению близка к единице, а выигрыш остальных занижен. В поиске
это давно решено объединением выдач: кандидаты собираются из всех
сравниваемых систем, и разметка ни одной из них не подыгрывает.

Два исключения, у которых эталон известен до всякого прогона:

* синтетические запросы — вопрос писался по конкретному фрагменту;
* живые вопросы с названными пунктами — фрагмент найден по началу пункта
  в тексте акта, и это внешнее свидетельство, не зависящее от выдачи.

Вопросы, у которых пункт назван, но в нашей редакции акта его нет, сюда
не попадают вовсе: подбирать им кандидатов поиском нельзя, получатся
правдоподобные и заведомо неверные.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDIR = os.path.join(ROOT, "data", "queries")
REPORT = os.path.join(ROOT, "docs", "raw", "build_query_texts.txt")

DROPPED = "отброшен"


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(QDIR, "queries.jsonl"))
    args = ap.parse_args()

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    queries: list[dict] = []
    # источники: файл, происхождение по умолчанию, подвыборка
    sources = [
        ("synthetic.jsonl", "синтетический", "синтетика"),
        ("synthetic_dev.jsonl", "синтетический dev", "dev"),
        ("synthetic_test.jsonl", "синтетический тест", "невиданные акты"),
        ("explan_live.jsonl", "вопрос из разъяснений Банка России", "живые"),
        ("manual.jsonl", "ручной", "ручные"),
    ]
    for name, origin, subset in sources:
        got = 0
        for q in read_jsonl(os.path.join(QDIR, name)):
            if str(q.get("route", "")).startswith(DROPPED):
                continue
            rec = {"query_id": q["query_id"], "text": q["text"],
                   "origin": q.get("origin", origin), "podvyborka": q.get("podvyborka", subset)}
            gold = q.get("gold_chunk_ids") or ([q["gold_chunk_id"]]
                                               if q.get("gold_chunk_id") else [])
            if gold:
                rec["gold_chunk_ids"] = gold
                rec["gold_chunk_id"] = gold[0]     # прежнее поле: его читают старые шаги
            if q.get("act"):
                rec["act"] = q["act"]
            queries.append(rec)
            got += 1
        if got or os.path.exists(os.path.join(QDIR, name)):
            say(f"   {name:<24} {got:>5}")

    seen: set[str] = set()
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

    by_subset = collections.Counter(q["podvyborka"] for q in unique)
    with_gold = sum(1 for q in unique if q.get("gold_chunk_ids"))
    say()
    say(f"запросов: {len(unique)}")
    for k, v in by_subset.most_common():
        длины = [len(q["text"]) for q in unique if q["podvyborka"] == k]
        say(f"  {k:<20} {v:>5}   медиана длины {statistics.median(длины):.0f} знаков")
    головная = sum(v for k, v in by_subset.items() if k in ("живые", "ручные",
                                                            "невиданные акты"))
    say(f"в заголовочную цифру (живые + ручные + невиданные): {головная}")
    say(f"доля синтетики среди собранных: "
        f"{100 * by_subset['синтетика'] / max(1, len(unique)):.0f}%")
    say(f"эталон уже известен (по построению или по пунктам): {with_gold}")
    say("остальным эталон подбирается после прогона, по объединённым выдачам")
    say(f"\nфайл: {os.path.relpath(args.out, ROOT)}")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
