#!/usr/bin/env python3
"""Сборка набора запросов с разметкой и выгрузка в формате MTEB.

Оценки релевантности:
  2 — фрагмент, по которому вопрос и составлен: прямой ответ;
  1 — соседний фрагмент того же акта, физически содержащий часть того же
      текста. Это не догадка: чанки нарезаны с перекрытием, и соседние
      действительно делят кусок текста, поэтому часть ответа в них есть;
  0 — всё остальное.

Кандидаты на оценку 1, которые перекрытием не объясняются (тот же раздел
акта, но другой текст), выносятся отдельным списком на ручную проверку,
а не проставляются сами: догадка, записанная в разметку, потом выглядит
как измерение.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin.benchmark import write_corpus, write_qrels, write_queries  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDIR = os.path.join(ROOT, "data", "queries")
CHUNKDIR = os.path.join(ROOT, "data", "chunks")
BENCH = os.path.join(ROOT, "data", "benchmark")
REPORT = os.path.join(ROOT, "docs", "raw", "build_benchmark.txt")

# Какую долю эталонного фрагмента должен перекрывать сосед, чтобы получить
# оценку 1 без участия человека. Замер по корпусу: половина соседних пар
# вообще не пересекается, у пересекающихся медиана перекрытия 0,15.
# Сосед с перекрытием 0,15 содержит седьмую часть эталона, и вероятность,
# что именно в неё попал ответ, невелика — записывать такое в разметку
# значит выдать догадку за измерение. Порог 0,30 срабатывает на 3% пар,
# где перекрытие действительно существенное; остальное идёт человеку.
MIN_SPAN_OVERLAP = 0.30
# сколько соседей показывать на ручную проверку: список должен быть обозримым
MAX_REVIEW_PER_QUERY = 4
# насколько далеко от эталона по порядку следования имеет смысл смотреть
NEIGHBOUR_WINDOW = 2


def span_overlap(a: dict, b: dict) -> float:
    lo, hi = max(a["char_start"], b["char_start"]), min(a["char_end"], b["char_end"])
    span = a["char_end"] - a["char_start"]
    return max(0, hi - lo) / span if span else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="base")
    ap.add_argument("--synthetic", default=os.path.join(QDIR, "synthetic.jsonl"))
    ap.add_argument("--manual", default=os.path.join(QDIR, "manual.jsonl"))
    ap.add_argument("--grades", default=os.path.join(QDIR, "grades.tsv"),
                    help="проверенные вручную градации: тот же файл кандидатов "
                         "с заполненной колонкой otsenka_0_1_2")
    args = ap.parse_args()

    chunks = [json.loads(l) for l in open(os.path.join(CHUNKDIR, f"{args.chunks}.jsonl"),
                                          encoding="utf-8")]
    by_id = {c["chunk_id"]: c for c in chunks}
    by_act: dict[str, list[dict]] = collections.defaultdict(list)
    for c in chunks:
        by_act[c["act_id"]].append(c)

    queries: list[dict] = []
    for path, origin in ((args.synthetic, "синтетический"), (args.manual, "ручной")):
        if not os.path.exists(path):
            continue
        for line in open(path, encoding="utf-8"):
            q = json.loads(line)
            q.setdefault("origin", origin)
            queries.append(q)

    qrels: dict[str, dict[str, int]] = {}
    review_rows: list[dict] = []
    for q in queries:
        gold_id = q["gold_chunk_id"]
        gold = by_id.get(gold_id)
        if gold is None:
            continue
        rel = {gold_id: 2}
        candidates: list[tuple[int, dict]] = []
        for other in by_act[gold["act_id"]]:
            if other["chunk_id"] == gold_id:
                continue
            if span_overlap(gold, other) >= MIN_SPAN_OVERLAP:
                rel[other["chunk_id"]] = 1
                continue
            distance = abs(other["position"] - gold["position"])
            same_section = bool(other.get("section")) and other["section"] == gold.get("section")
            if distance <= NEIGHBOUR_WINDOW or same_section:
                # ближние соседи важнее однораздельных: сортируем по расстоянию
                candidates.append((distance if distance <= NEIGHBOUR_WINDOW else 100 + distance,
                                   other))
        for _, other in sorted(candidates, key=lambda x: x[0])[:MAX_REVIEW_PER_QUERY]:
            review_rows.append({
                "query_id": q["query_id"], "query": q["text"],
                "kandidat_chunk_id": other["chunk_id"],
                "otsenka_0_1_2": "",
                "razdel": other.get("section", ""),
                "punkty": other.get("units", ""),
                "nachalo": other["text"].split("\n\n", 1)[-1][:200].replace("\n", " "),
            })
        qrels[q["query_id"]] = rel

    # проверенные человеком градации имеют приоритет над автоматическими
    graded_by_hand = 0
    if os.path.exists(args.grades):
        with open(args.grades, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                mark = (row.get("otsenka_0_1_2") or "").strip()
                qid, cid = row.get("query_id"), row.get("kandidat_chunk_id")
                if mark not in ("0", "1", "2") or qid not in qrels or not cid:
                    continue
                if mark == "0":
                    qrels[qid].pop(cid, None)
                else:
                    qrels[qid][cid] = int(mark)
                graded_by_hand += 1

    os.makedirs(QDIR, exist_ok=True)
    with open(os.path.join(QDIR, "queries.jsonl"), "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps({"query_id": q["query_id"], "text": q["text"],
                                "origin": q["origin"], "gold_chunk_id": q["gold_chunk_id"]},
                               ensure_ascii=False) + "\n")
    write_qrels(os.path.join(QDIR, "qrels.tsv"), qrels)

    # выгрузка в формате MTEB: корпус, запросы, разметка
    write_corpus(os.path.join(BENCH, "corpus.jsonl"), chunks)
    write_queries(os.path.join(BENCH, "queries.jsonl"), queries)
    write_qrels(os.path.join(BENCH, "qrels", "test.tsv"), qrels)

    # заготовка для ручной проверки градаций
    if review_rows:
        path = os.path.join(QDIR, "grade_candidates.tsv")
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(review_rows[0]), delimiter="\t")
            w.writeheader()
            w.writerows(review_rows[:1000])

    lines = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    origins = collections.Counter(q["origin"] for q in queries)
    graded = sum(1 for r in qrels.values() if len(r) > 1)
    say(f"запросов: {len(queries)}  ({', '.join(f'{k}: {v}' for k, v in origins.items())})")
    say(f"доля синтетических: {100 * origins['синтетический'] / max(1, len(queries)):.0f}%")
    say(f"размеченных пар запрос-фрагмент: {sum(len(r) for r in qrels.values())}")
    say(f"запросов с более чем одним релевантным фрагментом: {graded}")
    say(f"кандидатов на ручную проверку градаций: {len(review_rows)}")
    say(f"градаций проставлено вручную: {graded_by_hand}"
        + ("" if graded_by_hand else "  (файл data/queries/grades.tsv ещё не заполнен)"))
    say(f"корпус бенчмарка: {len(chunks)} фрагментов")
    say()
    say("выгружено в формате MTEB:")
    say("   data/benchmark/corpus.jsonl")
    say("   data/benchmark/queries.jsonl")
    say("   data/benchmark/qrels/test.tsv")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
