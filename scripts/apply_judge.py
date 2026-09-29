#!/usr/bin/env python3
"""Перенос разметки судьи в эталоны.

Судья размечает пул на Kaggle (`run_phase_c.py --judge`), обратно едет
`judge.tsv` — три колонки и сотни килобайт. Здесь оценки превращаются
в разметку.

Два правила.

**Запрос, у которого ни одному кандидату не поставлена двойка, из набора
выбывает.** Единицы без двойки — это «рядом, но ответа нет»; такой запрос
мерил бы не поиск, а полноту корпуса. Выбывших надо считать: их доля —
характеристика корпуса, и она идёт в отчёт.

**Разметка по названным пунктам сильнее разметки судьи.** Пункт, названный
Банком России, — внешнее свидетельство; оценка судьи — измерение
с погрешностью. Где они спорят, остаётся первое, а расхождения считаются:
это дешёвая проверка судьи на тех парах, где ответ известен независимо.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin.benchmark import read_qrels  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDIR = os.path.join(ROOT, "data", "queries")
REPORT = os.path.join(ROOT, "docs", "raw", "apply_judge.txt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default=os.path.join(QDIR, "judge.tsv"))
    ap.add_argument("--by-clause", default=os.path.join(QDIR, "qrels_by_clause.tsv"))
    ap.add_argument("--out", default=os.path.join(QDIR, "qrels_judge.tsv"))
    args = ap.parse_args()

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    if not os.path.exists(args.judge):
        raise SystemExit(f"нет {os.path.relpath(args.judge, ROOT)}: разметка судьи "
                         f"считается на Kaggle (run_phase_c.py --judge)")

    marks: dict[str, dict[str, int]] = collections.defaultdict(dict)
    with open(args.judge, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            mark = (row.get("otsenka_0_1_2") or "").strip()
            if mark in ("0", "1", "2"):
                marks[row["query_id"]][row["chunk_id"]] = int(mark)
    say(f"пар в разметке судьи: {sum(len(v) for v in marks.values())} "
        f"по {len(marks)} запросам")

    by_clause = read_qrels(args.by_clause) if os.path.exists(args.by_clause) else {}
    спор = совпало = 0
    for qid, rel in by_clause.items():
        for cid, score in rel.items():
            if cid in marks.get(qid, {}):
                if marks[qid][cid] == score:
                    совпало += 1
                else:
                    спор += 1
                marks[qid][cid] = score
    if совпало or спор:
        say(f"пар, где пункт назван и судья тоже высказался: {совпало + спор}; "
            f"совпало {совпало}, разошлось {спор} "
            f"({100 * спор / max(1, совпало + спор):.0f}%)")
        say("   при расхождении остаётся разметка по пунктам: она внешняя")

    kept, dropped = {}, []
    for qid, rel in marks.items():
        relevant = {cid: s for cid, s in rel.items() if s > 0}
        if 2 in relevant.values():
            kept[qid] = relevant
        else:
            dropped.append(qid)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for qid in sorted(kept):
            for cid, score in sorted(kept[qid].items()):
                f.write(f"{qid}\t{cid}\t{score}\n")

    def префикс(qid: str) -> str:
        return {"exp": "живые", "man": "ручные"}.get(qid[:3], qid[:3])

    say()
    say(f"запросов размечено: {len(kept)}")
    say(f"выбыло (ни одной двойки): {len(dropped)} "
        f"({100 * len(dropped) / max(1, len(marks)):.0f}%)")
    by_kind = collections.Counter(префикс(q) for q in kept)
    say("по происхождению: " + ", ".join(f"{k} {v}" for k, v in by_kind.most_common()))
    выбыло = collections.Counter(префикс(q) for q in dropped)
    if выбыло:
        say("выбыло по происхождению: " +
            ", ".join(f"{k} {v}" for k, v in выбыло.most_common()))
    двоек = sum(1 for rel in kept.values() for s in rel.values() if s == 2)
    единиц = sum(1 for rel in kept.values() for s in rel.values() if s == 1)
    say(f"пар с оценкой 2: {двоек}, с оценкой 1: {единиц}")
    say(f"файл: {os.path.relpath(args.out, ROOT)}")
    say("дальше: make bench — сборка набора и выгрузка в формате MTEB")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
