#!/usr/bin/env python3
"""Перенос заполненного листа ручных запросов в набор.

Лист готовит `manual_task.py`: строка — акт, колонка «zapros» — пустая.
Здесь заполненные строки превращаются в запросы с устойчивыми номерами.

Эталон тут не подбирается. Ручной запрос идёт тем же путём, что живой
вопрос без названных пунктов: прогон всех конфигураций, объединённый пул,
разметка. Подбирать эталон выдачей одного метода нельзя — его полнота
по построению окажется близка к единице.

Акт, назначенный листом, сохраняется рядом с запросом: по нему потом
проверяется, что подобранный эталон относится к тому акту, про который
запрос и писался. Расхождение — не обязательно ошибка (запрос может
подойти и другому акту), но знать о нём надо.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import statistics

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDIR = os.path.join(ROOT, "data", "queries")
REPORT = os.path.join(ROOT, "docs", "raw", "apply_manual_task.txt")

PREFIX = "man"
ORIGIN = "ручной"


def known_ids(*paths: str) -> dict[str, str]:
    """Уже выданные номера ручных запросов: текст -> query_id."""
    known: dict[str, str] = {}
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                q = json.loads(line)
                if q["query_id"].startswith(PREFIX):
                    known.setdefault(q["text"], q["query_id"])
    return known


def read_plain(path: str) -> list[str]:
    """Запросы, написанные до появления листа, — простым текстом по строке.

    Номер такому запросу даётся по порядку строки, а не по тексту: под этими
    номерами уже посчитаны выдачи, а текст в файле правился — в одном запросе
    исправлена опечатка. Привязка по тексту после такой правки выдала бы
    новый номер и сдвинула бы все остальные.

    Отсюда правило: строки в этом файле не переставляются и не удаляются.
    Новые ручные запросы пишутся не здесь, а в листе `manual_task.tsv`.
    """
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [s.strip() for s in f if s.strip() and not s.startswith("#")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default=os.path.join(QDIR, "manual_task.tsv"))
    ap.add_argument("--plain", default=os.path.join(QDIR, "manual_questions.txt"))
    ap.add_argument("--out", default=os.path.join(QDIR, "manual.jsonl"))
    args = ap.parse_args()

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    known = known_ids(os.path.join(QDIR, "queries.jsonl"), args.out)

    плоские = read_plain(args.plain)
    # номер по порядку строки: выдачи уже посчитаны под ними
    по_порядку = {text: f"{PREFIX}{n:04d}" for n, text in enumerate(плоские)}
    known = {**known, **по_порядку}

    items: list[tuple[str, str, str]] = []          # текст, акт, группа сплита
    for text in плоские:
        items.append((text, "", "до листа"))
    filled = skipped = 0
    if os.path.exists(args.tsv):
        with open(args.tsv, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                text = (row.get("zapros") or "").strip()
                if not text:
                    skipped += 1
                    continue
                filled += 1
                items.append((text, row["akt"], row["gruppa"]))

    seen: set[str] = set()
    out: list[dict] = []
    used = {int(q[len(PREFIX):]) for q in known.values() if q[len(PREFIX):].isdigit()}
    nxt = max(used) + 1 if used else 0
    for text, act, group in items:
        if text in seen:
            continue
        seen.add(text)
        qid = known.get(text)
        if qid is None:
            while nxt in used:
                nxt += 1
            qid = f"{PREFIX}{nxt:04d}"
            used.add(nxt)
        out.append({"query_id": qid, "origin": ORIGIN, "text": text,
                    "act": act or None, "gruppa_splita": group})

    with open(args.out, "w", encoding="utf-8") as f:
        for q in sorted(out, key=lambda r: r["query_id"]):
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    say(f"строк в листе заполнено: {filled}, пропущено: {skipped}")
    say(f"запросов, написанных до листа: {len(read_plain(args.plain))}")
    say(f"ручных запросов всего: {len(out)}, актов назначено: "
        f"{len({q['act'] for q in out if q['act']})}")
    if out:
        длины = [len(q["text"]) for q in out]
        say(f"длина запроса: медиана {statistics.median(длины):.0f} знаков, "
            f"от {min(длины)} до {max(длины)}")
        by_group = collections.Counter(q["gruppa_splita"] for q in out)
        say("по группам сплита: " + ", ".join(f"{k} {v}" for k, v in by_group.most_common()))
    if filled < 75:
        say(f"\nдо цели не хватает {max(0, 75 - filled)} строк: ручные запросы — "
            f"единственное место, где разнообразие актов задаётся нарочно")
    say(f"файл: {os.path.relpath(args.out, ROOT)}")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
