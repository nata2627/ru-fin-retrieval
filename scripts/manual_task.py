#!/usr/bin/env python3
"""Лист для ручных запросов: список актов, по которым их писать.

Ручные запросы — единственная часть набора, где **разнообразие актов
задаётся нарочно**. Живые вопросы его не дают: 52% из них про одно
Положение 590-П, потому что по нему у Банка России разъяснений больше
всего. Синтетика не даёт третьего речевого режима: вопрос, написанный
моделью по фрагменту, наследует язык акта, а в строку поиска набирают
иначе — короче, без канцелярита, с сокращениями.

Поэтому лист, а не «напишите запросов». Запрос по памяти — это запрос
про то, чем занимался на неделе, и сотня таких снова окажется про резервы.
Лист заранее назначает акт каждому запросу и покрывает и виды документов,
и темы.

Акты берутся только из отложенных групп сплита. Ручной запрос попадает
в тест, а значит, его акт не может участвовать в обучении: иначе прирост
окажется отчасти узнаванием того же текста.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin import split as S  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDIR = os.path.join(ROOT, "data", "queries")
CHUNKDIR = os.path.join(ROOT, "data", "chunks")
REPORT = os.path.join(ROOT, "docs", "raw", "manual_task.txt")

KIND_NAMES = {"У": "Указание", "П": "Положение", "И": "Инструкция",
              "МР": "Методические рекомендации", "ОР": "Официальное разъяснение"}

# Откуда и сколько актов брать. Больше всего — из невиданных: они и нужны
# для разнообразия. Инструкции берутся отдельной долей, потому что это
# отложенный класс документов и переносить на него надо проверенно.
PLAN = {S.TEST_UNSEEN: 55, S.TEST_KIND: 25, S.TEST_LIVE: 20}

MIN_CHUNKS = 4          # акт из одного фрагмента — это справка, спрашивать не о чем


def read_acts(chunks_path: str) -> dict[str, dict]:
    acts: dict[str, dict] = {}
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            a = acts.setdefault(c["act_id"], {
                "act_id": c["act_id"], "number": c["number"], "type": c["type"],
                "title": c["title"], "date": c["date"], "chunks": 0, "sections": []})
            a["chunks"] += 1
            section = (c.get("section") or "").strip()
            if section and section not in a["sections"] and len(a["sections"]) < 3:
                a["sections"].append(section)
    return acts


def pick(pool: list[dict], count: int, rnd: random.Random,
         used: set[str]) -> list[dict]:
    """Отобрать акты, распределив их по видам документа поровну от наличия.

    Один номер — один запрос: акт, изданный дважды (разные редакции под тем же
    номером), дал бы две строки про одно и то же, а лист существует ради
    разнообразия.
    """
    by_kind: dict[str, list[dict]] = collections.defaultdict(list)
    for a in pool:
        if a["number"] not in used:
            by_kind[a["type"]].append(a)
    for group in by_kind.values():
        group.sort(key=lambda a: a["act_id"])
        rnd.shuffle(group)
    out: list[dict] = []
    kinds = sorted(by_kind, key=lambda k: -len(by_kind[k]))
    while len(out) < count and any(by_kind[k] for k in kinds):
        for k in kinds:
            if by_kind[k] and len(out) < count:
                a = by_kind[k].pop()
                if a["number"] in used:
                    continue
                used.add(a["number"])
                out.append(a)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="base")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(QDIR, "manual_task.tsv"))
    args = ap.parse_args()

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    split = S.load(os.path.join(QDIR, "split.json"))
    acts = read_acts(os.path.join(CHUNKDIR, f"{args.chunks}.jsonl"))
    rnd = random.Random(args.seed)

    chosen: list[tuple[str, dict]] = []
    used: set[str] = set()
    for group, count in PLAN.items():
        pool = [acts[i] for i in split["gruppy"][group]
                if i in acts and acts[i]["chunks"] >= MIN_CHUNKS]
        got = pick(pool, count, rnd, used)
        chosen += [(group, a) for a in got]
        say(f"{group:<18} актов доступно {len(pool):>4}, взято {len(got):>3}")

    rows = []
    for n, (group, a) in enumerate(chosen, start=1):
        rows.append({
            "nomer": n,
            "zapros": "",
            "akt": a["number"],
            "vid": KIND_NAMES.get(a["type"], a["type"]),
            "data": a["date"],
            "nazvanie": a["title"][:150],
            "razdely": " | ".join(s[:50] for s in a["sections"]),
            "gruppa": group,
        })

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    kinds = collections.Counter(r["vid"] for r in rows)
    say()
    say(f"строк в листе: {len(rows)}, актов: {len({r['akt'] for r in rows})}")
    say("по видам документа: " + ", ".join(f"{k} {v}" for k, v in kinds.most_common()))
    say(f"файл: {os.path.relpath(args.out, ROOT)}")
    say()
    say("Что делать с листом (колонка «zapros»):")
    say("  — писать так, как набирают в строку поиска: коротко, без канцелярита;")
    say("    ориентир по длине — 40–80 знаков, это медиана уже написанных;")
    say("  — про то, что в этом акте действительно есть: колонки «nazvanie»")
    say("    и «razdely» подсказывают тему, открывать сам акт не нужно;")
    say("  — не переписывать название акта и не называть его номер: запрос")
    say("    с номером акта решает задачу поиска по номеру, а не по смыслу;")
    say("  — строку можно пропустить, если про акт нечего спросить, — лучше")
    say("    пропустить, чем написать вопрос ни о чём;")
    say("  — цель 75–100 заполненных строк.")
    say()
    say("дальше: python3 scripts/apply_manual_task.py")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
