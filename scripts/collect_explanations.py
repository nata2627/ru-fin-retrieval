#!/usr/bin/env python3
"""Сбор настоящих вопросов практиков из раздела «Разъяснения» Банка России.

Половина набора запросов генерируется моделью по фрагментам корпуса и потому
неизбежно наследует язык самих актов. Чтобы измерить поиск на живых
формулировках, нужен источник вопросов, к корпусу не привязанный.

Банк России публикует ответы на вопросы поднадзорных организаций
(cbr.ru/explan/). Это вопросы, которые реально задавали банки: с жаргоном,
сокращениями и подразумеваемым контекстом. Заголовок темы у части разделов
прямо называет пункты акта — «Ведение кредитного досье (3.1.3, 3.1.5)», —
и это позволяет предложить эталонный фрагмент, а не искать его вслепую.

Важно понимать, чем эти вопросы отличаются от поисковых запросов: это письма
в надзорный орган, они длиннее и развёрнутее. В наборе они идут отдельным
происхождением, и в отчёте это указывается.

Собираются только вопросы; ответы Банка России не используются.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin.http import Client        # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "data", "queries", "explan_questions.jsonl")
REPORT = os.path.join(ROOT, "docs", "raw", "collect_explanations.txt")

INDEX = "https://www.cbr.ru/explan/"
SITE = "https://www.cbr.ru"

_SECTION = re.compile(r'href="(/explan/[^"#?]+/)"')
_H1 = re.compile(r"<h1[^>]*>(.*?)</h1>", re.S)
_BLOCK = re.compile(
    r'<div class="dropdown dropdown_container general-question".*?<h2[^>]*>(.*?)</h2>(.*?)'
    r'(?=<div class="dropdown dropdown_container general-question"|\Z)', re.S)
_QUESTION = re.compile(r'<div class="question_title[^"]*"[^>]*>(.*?)</div>', re.S)
# номер акта: «590-П», «4212-У», «180-И»
_ACT = re.compile(r"№\s*([\d/]+-[А-ЯЁ]{1,3})")
# ссылка на пункт: «3.1.3», «3.7.2.2»
_CLAUSE = re.compile(r"\b\d+(?:\.\d+)+\b")

MIN_LEN = 40


def clean(x: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", x))).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pause", type=float, default=1.2)
    ap.add_argument("--limit", type=int, default=0, help="ограничить число разделов")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    client = Client(pause=args.pause)

    sections = sorted(set(_SECTION.findall(client.get(INDEX).text)))
    if args.limit:
        sections = sections[:args.limit]

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    say(f"разделов «Разъяснения»: {len(sections)}")
    say()

    records: list[dict] = []
    for n, path in enumerate(sections, 1):
        url = SITE + path
        try:
            page = client.get(url).text
        except Exception as e:  # noqa: BLE001
            say(f"   {path}: не открылся ({e})")
            continue
        h1 = _H1.search(page)
        title = clean(h1.group(1)) if h1 else path
        acts = _ACT.findall(title)
        found = 0
        for head_raw, body in _BLOCK.findall(page):
            topic = clean(head_raw)
            clauses = _CLAUSE.findall(topic)
            for q_raw in _QUESTION.findall(body):
                text = clean(q_raw)
                if len(text) < MIN_LEN:
                    continue
                records.append({
                    "text": text, "topic": topic, "clauses": clauses,
                    "section": title, "acts": acts, "url": url,
                })
                found += 1
        if found:
            say(f"   {path:<52} {found:>4} вопросов"
                + (f", акт {', '.join(acts)}" if acts else ""))
        if n % 20 == 0:
            say(f"   ... пройдено {n}/{len(sections)}, собрано {len(records)}")

    with open(OUT, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with_act = [r for r in records if r["acts"]]
    with_clause = [r for r in records if r["acts"] and r["clauses"]]
    say()
    say(f"собрано вопросов: {len(records)}")
    say(f"  из них с указанным в разделе актом: {len(with_act)}")
    say(f"  из них ещё и с пунктами в теме:     {len(with_clause)}")
    if records:
        import statistics
        say(f"длина вопроса: медиана {statistics.median(len(r['text']) for r in records):.0f} "
            f"символов")
    say(f"файл: {os.path.relpath(OUT, ROOT)}")

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
