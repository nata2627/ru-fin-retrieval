#!/usr/bin/env python3
"""Проверка нарезки выпусков на акты.

Берёт выборку выпусков, режет их и сверяет найденные акты с перечнем:
сколько обещано, сколько найдено, не появилось ли лишних. Лишние опаснее
пропусков — посторонний «акт» забирает себе приложения настоящего.

Акт считается найденным, если он обнаружен хотя бы в одном из выпусков,
за которыми числится: один и тот же акт нередко значится сразу за
несколькими, и текст лежит не всегда в первом из них.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin.http import Client  # noqa: E402
from rufin.sources import cbr_vestnik as vb  # noqa: E402
from rufin.vestnik_split import normalize_number, split_issue  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INDEX = os.path.join(ROOT, "data", "index", "cbr_vestnik_acts.jsonl")
RAW = os.path.join(ROOT, "docs", "raw", "validate_split.txt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--issues", type=int, default=25, help="сколько выпусков проверить")
    ap.add_argument("--since", type=int, default=2013)
    ap.add_argument("--pause", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    acts = [json.loads(l) for l in open(INDEX, encoding="utf-8")]
    acts = vb.select_corpus(acts, since=args.since)
    by_issue: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for a in acts:
        by_issue[(a["issue"], a["issue_url"])].add(normalize_number(a["number"]))

    items = list(by_issue.items())
    random.Random(args.seed).shuffle(items)
    # половина выборки — самые насыщенные выпуски: там больше шансов на ошибку границ
    half = args.issues // 2
    pick = sorted(items, key=lambda x: -len(x[1]))[:half] + items[:args.issues - half]

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    client = Client(pause=args.pause)
    exp_total = found_total = extra_total = 0
    misses: dict[str, list[str]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for (issue, url), expected in pick:
            try:
                r = client.get(url)
            except Exception as e:  # noqa: BLE001
                say(f"   {issue}: не скачался ({e})")
                continue
            p = os.path.join(tmp, "issue.pdf")
            with open(p, "wb") as f:
                f.write(r.content)
            got = {a.number for a in split_issue(p, expected=expected)}
            hit, miss, extra = expected & got, expected - got, got - expected
            exp_total += len(expected)
            found_total += len(hit)
            extra_total += len(extra)
            if miss:
                misses[issue] = sorted(miss)
            say(f"{'OK ' if not miss and not extra else '!! '}{issue:<30} "
                f"обещано {len(expected):>2}, найдено {len(hit):>2}"
                + (f"  не найдено: {', '.join(sorted(miss))}" if miss else "")
                + (f"  лишние: {', '.join(sorted(extra))}" if extra else ""))
            os.remove(p)

    say()
    say(f"итого: найдено {found_total}/{exp_total} = {100 * found_total / exp_total:.0f}%, "
        f"лишних {extra_total}")
    if misses:
        say("\nне найдено по выпускам (акт может быть опубликован в соседнем выпуске,")
        say("тогда при сборе он подберётся оттуда):")
        for issue, nums in misses.items():
            say(f"   {issue}: {', '.join(nums)}")

    os.makedirs(os.path.dirname(RAW), exist_ok=True)
    with open(RAW, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\nсырой вывод: docs/raw/validate_split.txt")


if __name__ == "__main__":
    main()
