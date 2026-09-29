#!/usr/bin/env python3
"""Сбор корпуса: выкачать выпуски «Вестника» и разложить их на акты.

Состав корпуса — самостоятельные Положения, Инструкции, Указания,
Методические рекомендации и Официальные разъяснения начиная с 2013 года.
Правки («О внесении изменений в пункт 3.1…») не берутся: самостоятельного
смысла в них нет, а как ответ на вопрос они бесполезны.

Выпуски кладутся в data/raw и повторно не выкачиваются, поэтому прогон
можно прервать и продолжить. Готовые акты — data/corpus/acts.jsonl.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin.http import Client  # noqa: E402
from rufin.sources import cbr_vestnik as vb  # noqa: E402
from rufin.vestnik_split import normalize_number, split_issue  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INDEX = os.path.join(ROOT, "data", "index", "cbr_vestnik_acts.jsonl")
RAWDIR = os.path.join(ROOT, "data", "raw", "vestnik")
OUT = os.path.join(ROOT, "data", "corpus", "acts.jsonl")
REPORT = os.path.join(ROOT, "docs", "raw", "build_corpus.txt")

CYR = re.compile(r"[а-яёА-ЯЁ]")
LAT = re.compile(r"[a-zA-Z]")
# ниже этой доли кириллицы текст считается испорченным и в корпус не идёт
MIN_CYRILLIC = 0.90
# слишком короткий «акт» означает сбой нарезки, а не короткий документ
MIN_CHARS = 500


def issue_code(url: str) -> str:
    return url.rstrip("/").rsplit("/", 1)[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=int, default=2013)
    ap.add_argument("--pause", type=float, default=1.5)
    ap.add_argument("--limit", type=int, default=0, help="ограничить число выпусков (для пробы)")
    args = ap.parse_args()

    os.makedirs(RAWDIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)

    all_rows = [json.loads(l) for l in open(INDEX, encoding="utf-8")]
    wanted = vb.select_corpus(all_rows, since=args.since)
    # ключ акта — номер вместе с датой: у методических рекомендаций
    # нумерация начинается заново каждый год, номер сам по себе не уникален
    want_by_key = {(normalize_number(a["number"]), a["date"]): a for a in
                   {(a["number"], a["date"]): a for a in wanted}.values()}

    # какие выпуски нужно выкачать и что в каждом искать
    issues: dict[str, dict] = {}
    for a in wanted:
        code = issue_code(a["issue_url"])
        issues.setdefault(code, {"issue": a["issue"], "url": a["issue_url"], "want": set()})
        issues[code]["want"].add((normalize_number(a["number"]), a["date"]))
    # границы актов считаются по всем публикациям выпуска, а не только по нужным:
    # неопознанный акт не закрывает предыдущий, и его страницы прилипают к чужому тексту
    for a in all_rows:
        code = issue_code(a["issue_url"])
        if code in issues:
            issues[code].setdefault("bounds", set()).add(normalize_number(a["number"]))

    codes = sorted(issues, key=lambda c: issues[c]["issue"])
    if args.limit:
        codes = codes[:args.limit]

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    say(f"нужно актов: {len(want_by_key)}, выпусков к выкачке: {len(codes)}")
    client = Client(pause=args.pause)
    collected: dict[tuple[str, str], dict] = {}
    t0 = time.monotonic()
    downloaded = cached = failed = 0

    for n, code in enumerate(codes, 1):
        info = issues[code]
        path = os.path.join(RAWDIR, f"{code}.pdf")
        if not os.path.exists(path):
            try:
                r = client.get(info["url"])
            except Exception as e:  # noqa: BLE001
                say(f"   выпуск {info['issue']}: не скачался ({e})")
                failed += 1
                continue
            with open(path, "wb") as f:
                f.write(r.content)
            downloaded += 1
        else:
            cached += 1

        try:
            acts = split_issue(path, expected=info.get("bounds"))
        except Exception as e:  # noqa: BLE001
            say(f"   выпуск {info['issue']}: не разобрался ({e})")
            failed += 1
            continue

        by_num = {a.number: a for a in acts}
        for key in info["want"]:
            if key in collected:
                continue                      # уже взят из другого выпуска
            act = by_num.get(key[0])
            if act is None:
                continue
            c, l = len(CYR.findall(act.text)), len(LAT.findall(act.text))
            meta = want_by_key[key]
            collected[key] = {
                "act_id": f"{key[0]}_{meta['date'].replace('.', '')}",
                "number": key[0],
                "date": meta["date"],
                "kind": act.kind,
                "type": vb.act_type(meta["number"]),
                "title": meta["title"],
                "issue": info["issue"],
                "issue_url": info["url"],
                "first_page": act.first_page,
                "last_page": act.last_page,
                "chars": len(act.text),
                "cyrillic": round(c / (c + l), 3) if c + l else 0.0,
                "text": act.text,
            }
        if n % 25 == 0 or n == len(codes):
            say(f"   выпуск {n}/{len(codes)}, собрано актов {len(collected)}/{len(want_by_key)}, "
                f"{time.monotonic() - t0:.0f} с")

    # отсев заведомо испорченного
    good, bad = [], collections.Counter()
    for rec in collected.values():
        if rec["cyrillic"] < MIN_CYRILLIC:
            bad["мало кириллицы"] += 1
        elif rec["chars"] < MIN_CHARS:
            bad["слишком короткий"] += 1
        else:
            good.append(rec)

    good.sort(key=lambda r: (r["date"][-4:], r["date"][3:5], r["date"][:2], r["number"]))
    with open(OUT, "w", encoding="utf-8") as f:
        for rec in good:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    say()
    say(f"выпусков: скачано {downloaded}, из кэша {cached}, сбоев {failed}")
    say(f"актов найдено {len(collected)} из {len(want_by_key)} "
        f"= {100 * len(collected) / len(want_by_key):.0f}%")
    if bad:
        say(f"отсеяно: {dict(bad)}")
    say(f"в корпусе: {len(good)} актов")
    if good:
        ch = [r["chars"] for r in good]
        say(f"символов: всего {sum(ch) / 1e6:.1f} млн, медиана {statistics.median(ch):.0f}, "
            f"максимум {max(ch)}")
        say(f"по видам: {dict(collections.Counter(r['type'] for r in good).most_common())}")
        say(f"по годам: {dict(sorted(collections.Counter(r['date'][-4:] for r in good).items()))}")
        say(f"файл: data/corpus/acts.jsonl ({os.path.getsize(OUT) / 1048576:.0f} МБ)")

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
