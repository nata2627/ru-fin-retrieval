#!/usr/bin/env python3
"""Живые вопросы в трёх вариантах: с номерами актов, без номеров, без ссылок.

Зачем. В живых вопросах из «Разъяснений» автор обычно сам называет акт
и пункт. Сравнение «вопросы с номером против вопросов без номера» идёт
по двум разным группам: в одной 112 вопросов, в другой 30, и они отличаются
не только номером, а темой, длиной и тем, насколько вообще находимы.
Поэтому номер убирается из тех же самых вопросов, и сравнение становится
парным: один и тот же вопрос до и после правки.

Два варианта правки, потому что «без номера» можно понять двояко; разбор
обоих и их границы — в src/rufin/references.py.

На выходе один файл со всеми вариантами. Идентификатор запроса составной:
«bez-nomerov:exp0133». Так весь прогон на видеокарте делается одним
проходом: индекс BM25 строится один раз на все 426 запросов, а не трижды.
Эталоны у вариантов те же самые, поэтому в файл они не пишутся: разбирать
выдачи будем локально, по исходным эталонам.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from rufin.references import has_act_number, strip_numbers, strip_references  # noqa: E402

QDIR = os.path.join(ROOT, "data", "queries")
REPORT = os.path.join(ROOT, "docs", "raw", "live_variants.txt")

# название варианта -> как получить текст
VARIANTS = {
    "orig": lambda t: t,
    "bez-nomerov": strip_numbers,
    "bez-ssylok": strip_references,
}
ОПИСАНИЕ = {
    "orig": "как задан на сайте, без правок",
    "bez-nomerov": "номера актов и пунктов убраны, слова оставлены",
    "bez-ssylok": "ссылка убрана целиком вместе со словами",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=os.path.join(QDIR, "queries.jsonl"))
    ap.add_argument("--subsets", default=os.path.join(QDIR, "podvyborki.json"))
    ap.add_argument("--subset", default="живые")
    ap.add_argument("--out", default=os.path.join(QDIR, "live_variants.jsonl"))
    ap.add_argument("--examples", type=int, default=6)
    args = ap.parse_args()

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    with open(args.queries, encoding="utf-8") as f:
        rows = {r["query_id"]: r for r in (json.loads(l) for l in f if l.strip())}
    with open(args.subsets, encoding="utf-8") as f:
        ids = json.load(f)[args.subset]

    out: list[dict] = []
    for name, rule in VARIANTS.items():
        for qid in ids:
            text = rule(rows[qid]["text"])
            out.append({"query_id": f"{name}:{qid}", "base_id": qid,
                        "variant": name, "text": text})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with_num = [q for q in ids if has_act_number(rows[q]["text"])]
    say(f"подвыборка «{args.subset}»: {len(ids)} вопросов")
    say(f"из них с номером акта: {len(with_num)}, без номера: {len(ids) - len(with_num)}")
    say()
    say(f"{'вариант':<14} {'запросов':>9} {'медиана знаков':>15} {'номер остался':>14}")
    for name in VARIANTS:
        часть = [r for r in out if r["variant"] == name]
        длины = statistics.median(len(r["text"]) for r in часть)
        осталось = sum(1 for r in часть if has_act_number(r["text"]))
        say(f"{name:<14} {len(часть):>9} {длины:>15.0f} {осталось:>14}")
    say()
    say("Правка задевает только те вопросы, где ссылка есть: остальные "
        "переписаны тождественно.")
    for name in ("bez-nomerov", "bez-ssylok"):
        тронуто = sum(1 for r in out if r["variant"] == name
                      and r["text"] != rows[r["base_id"]]["text"])
        say(f"  {name:<14} изменено вопросов: {тронуто} из {len(ids)}")

    say()
    say(f"Примеры правки ({args.examples} вопросов с номером акта):")
    for qid in with_num[:args.examples]:
        say()
        say(f"  {qid}")
        for name in VARIANTS:
            text = VARIANTS[name](rows[qid]["text"])
            say(f"    {name:<12} {text[:300]}")

    say()
    say(f"файл: {os.path.relpath(args.out, ROOT)}")
    for name, note in ОПИСАНИЕ.items():
        say(f"  {name:<14} {note}")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
