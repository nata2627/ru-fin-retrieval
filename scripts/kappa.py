#!/usr/bin/env python3
"""Согласие судьи с человеком: каппа Коэна.

Судья размечает пул целиком, человек — стратифицированную выборку в 150 пар.
Каппа — единственное число, которым в проекте измеряется качество самой
разметки, и без него «размечено языковой моделью» означает «неизвестно, что
размечено».

Считаются две каппы. Простая не различает, насколько велик промах.
Взвешенная (квадратичные веса) различает: оценки упорядочены, и «2 против 0»
хуже, чем «2 против 1». Публикуются обе.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin.annotation import cohen_kappa, kappa_verdict  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDIR = os.path.join(ROOT, "data", "queries")
REPORT = os.path.join(ROOT, "docs", "raw", "kappa.txt")


def read_marks(path: str) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            mark = (row.get("otsenka_0_1_2") or "").strip()
            if mark in ("0", "1", "2"):
                out[(row["query_id"], row["chunk_id"])] = int(mark)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default=os.path.join(QDIR, "judge.tsv"))
    ap.add_argument("--human", default=os.path.join(QDIR, "human_sample.tsv"))
    args = ap.parse_args()

    for path in (args.judge, args.human):
        if not os.path.exists(path):
            raise SystemExit(f"нет {os.path.relpath(path, ROOT)}")

    судья = read_marks(args.judge)
    человек = read_marks(args.human)
    общие = sorted(set(судья) & set(человек))
    if not общие:
        raise SystemExit("нет ни одной пары, размеченной обоими: "
                         "лист человека не заполнен?")

    a = [судья[k] for k in общие]
    b = [человек[k] for k in общие]
    простая = cohen_kappa(a, b)
    взвешенная = cohen_kappa(a, b, weighted=True)
    совпало = sum(1 for x, y in zip(a, b) if x == y)

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    say(f"пар размечено человеком: {len(человек)}, из них есть у судьи: {len(общие)}")
    say(f"совпало дословно: {совпало} из {len(общие)} "
        f"({100 * совпало / len(общие):.0f}%)")
    say()
    say(f"каппа Коэна простая:     {простая:.3f}  — {kappa_verdict(простая)}")
    say(f"каппа Коэна взвешенная:  {взвешенная:.3f}  — {kappa_verdict(взвешенная)}")
    say()
    say("матрица расхождений (строки — судья, столбцы — человек):")
    матрица: collections.Counter = collections.Counter(zip(a, b))
    say("        " + "".join(f"{j:>8}" for j in (0, 1, 2)))
    for i in (0, 1, 2):
        say(f"   {i:<4}" + "".join(f"{матрица[(i, j)]:>8}" for j in (0, 1, 2)))
    расхождения = [(k, x, y) for k, x, y in zip(общие, a, b) if abs(x - y) >= 2]
    say()
    say(f"грубых расхождений (на две ступени): {len(расхождения)}")
    for k, x, y in расхождения[:10]:
        say(f"   {k[0]} / {k[1]}: судья {x}, человек {y}")
    if простая < 0.40:
        say()
        say("каппа ниже 0,40: разметку судьи в набор брать нельзя, она меряет "
            "что-то своё. Уточнить инструкцию и разметить заново — всё, "
            "а не только спорное.")
    say()
    say("число идёт в README.md и DATASET_CARD.md")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
