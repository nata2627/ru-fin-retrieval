#!/usr/bin/env python3
"""Таблица рецепта из журнала обучения, с пересчётом вердиктов на месте.

Журнал пишется на видеокарте, а таблица в отчёт собирается здесь. Вердикты
при этом **считаются заново** из поквериных значений, а не переписываются
из файла. Это не недоверие к прогону, а дешёвая проверка: правило приёмки
одно, код один, и если пересчёт даёт другой вердикт, значит журнал приехал
не тем, чем уехал, — например, собран другой версией кода.

Печатается таблица целиком, включая отклонённые этапы. Отклонённый этап —
такой же результат, как принятый: он говорит, что в этой задаче не работает,
и из отчёта его не выбрасывают.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin.training.journal import Journal  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(ROOT, "docs", "raw")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", default=os.path.join(RAW, "journal.json"),
                    help="журнал, приехавший с видеокарты")
    args = ap.parse_args()

    if not os.path.exists(args.journal):
        raise SystemExit(
            f"нет журнала ({os.path.relpath(args.journal, ROOT)}). Его пишет "
            f"`run_phase_d.py` на Kaggle; привезти и положить в docs/raw/.")

    приехавший = Journal.load(args.journal)
    if not приехавший.entries:
        raise SystemExit("журнал пуст: ни один этап не посчитан")

    # пересчёт: записи добавляются в том же порядке, вердикты выводятся заново
    свежий = Journal()
    расхождения = []
    for запись in приехавший.entries:
        было = (запись.verdict, запись.compared_with)
        копия = Journal.load(args.journal).by_tag(запись.tag)
        копия.verdict, копия.compared_with, копия.diff = "", "", None
        стало = свежий.add(копия)
        if (стало.verdict, стало.compared_with) != было and было != ("", ""):
            расхождения.append((запись.tag, было, (стало.verdict, стало.compared_with)))

    строки = [свежий.table(), ""]
    лучшее = свежий.best()
    базовая = свежий.baseline()
    if лучшее and базовая and лучшее.tag != базовая.tag:
        строки.append(f"Лучшее принятое: `{лучшее.tag}`, dev NDCG@10 "
                      f"{лучшее.ndcg:.3f} против {базовая.ndcg:.3f} "
                      f"у необученного ученика, то есть "
                      f"{лучшее.ndcg - базовая.ndcg:+.3f}.")
    elif лучшее:
        строки.append(f"Ни один этап не принят. Нулевая точка: "
                      f"dev NDCG@10 {базовая.ndcg:.3f}.")
    принято = sum(1 for e in свежий.entries if e.verdict == "принят")
    строки.append(f"Этапов посчитано {len(свежий.entries) - 1}, принято {принято}.")
    if расхождения:
        строки.append("")
        строки.append("**Пересчёт вердиктов разошёлся с журналом**, и это надо "
                      "разобрать до публикации отчёта:")
        for метка, было, стало in расхождения:
            строки.append(f"* `{метка}`: в журнале {было}, пересчёт даёт {стало}")

    текст = "\n".join(строки)
    print(текст)
    os.makedirs(RAW, exist_ok=True)
    with open(os.path.join(RAW, "recipe.txt"), "w", encoding="utf-8") as f:
        f.write(текст + "\n")
    print("\nтаблица: docs/raw/recipe.txt")
    if расхождения:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
