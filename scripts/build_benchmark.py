#!/usr/bin/env python3
"""Сборка размеченного набора и выгрузка в формате MTEB.

Разметка стекается из трёх источников, и приоритет у них разный.

1. **Эталон по построению** — синтетический вопрос писался по конкретному
   фрагменту. Оценка 2.
2. **Эталон по названным пунктам** — Банк России сам назвал пункты акта,
   фрагмент найден по началу пункта. Оценка 2. Внешнее свидетельство,
   от выдачи не зависит.
3. **Разметка судьи по пулу** (`qrels_judge.tsv`) — измерение с погрешностью,
   и погрешность измерена каппой. Где судья спорит с первыми двумя,
   остаются первые два.

Плюс автоматическая оценка 1 для соседа, который физически содержит
существенную часть текста эталона: чанки нарезаны с перекрытием, и часть
ответа в таком соседе действительно есть. Порог 0,30 — не догадка,
а замер по корпусу: половина соседних пар не пересекается вовсе,
у пересекающихся медиана перекрытия 0,15, то есть сосед содержит седьмую
часть эталона. Записывать такое в релевантные значило бы выдать догадку
за измерение, поэтому остальные соседи уходят человеку списком.

**Разметка только дополняется.** Пара «запрос — фрагмент», однажды
опубликованная в `qrels.tsv`, обязана остаться: прежние прогоны посчитаны
по ней, и молча исчезнувший эталон выглядел бы как провал поиска. Если пара
пропала, сборка отказывается писать файл и говорит, какая именно.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin import split as S  # noqa: E402
from rufin.benchmark import read_qrels, write_corpus, write_qrels, write_queries  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDIR = os.path.join(ROOT, "data", "queries")
CHUNKDIR = os.path.join(ROOT, "data", "chunks")
BENCH = os.path.join(ROOT, "data", "benchmark")
REPORT = os.path.join(ROOT, "docs", "raw", "build_benchmark.txt")

MIN_SPAN_OVERLAP = 0.30
MAX_REVIEW_PER_QUERY = 4
NEIGHBOUR_WINDOW = 2

# Подвыборки, из которых складывается заголовочная цифра. Синтетика в неё
# не входит: вопрос, написанный моделью по фрагменту, меряет сопоставление
# канцелярита с канцеляритом, а не ответ на живой вопрос.
HEADLINE = ("живые", "ручные", "невиданные акты")
# Акт, доля которого в живых вопросах велика настолько, что цифру по живым
# надо печатать дважды: со ним и без него.
DOMINANT_ACT = "590-П"


def span_overlap(a: dict, b: dict) -> float:
    lo, hi = max(a["char_start"], b["char_start"]), min(a["char_end"], b["char_end"])
    span = a["char_end"] - a["char_start"]
    return max(0, hi - lo) / span if span else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="base")
    ap.add_argument("--queries", default=os.path.join(QDIR, "queries.jsonl"))
    ap.add_argument("--judge", default=os.path.join(QDIR, "qrels_judge.tsv"))
    ap.add_argument("--by-clause", default=os.path.join(QDIR, "qrels_by_clause.tsv"))
    ap.add_argument("--grades", default=os.path.join(QDIR, "grades.tsv"))
    ap.add_argument("--allow-shrink", action="store_true",
                    help="разрешить исчезновение прежних эталонов (не надо)")
    args = ap.parse_args()

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    chunks = [json.loads(l) for l in open(os.path.join(CHUNKDIR, f"{args.chunks}.jsonl"),
                                          encoding="utf-8")]
    by_id = {c["chunk_id"]: c for c in chunks}
    by_act: dict[str, list[dict]] = collections.defaultdict(list)
    for c in chunks:
        by_act[c["act_id"]].append(c)

    queries = [json.loads(l) for l in open(args.queries, encoding="utf-8") if l.strip()]
    say(f"запросов в наборе: {len(queries)}, фрагментов в корпусе: {len(chunks)}")

    qrels: dict[str, dict[str, int]] = collections.defaultdict(dict)
    review_rows: list[dict] = []

    # ---- 1-2. эталоны, известные до прогона ----
    for q in queries:
        gold_ids = q.get("gold_chunk_ids") or ([q["gold_chunk_id"]]
                                               if q.get("gold_chunk_id") else [])
        for gold_id in gold_ids:
            gold = by_id.get(gold_id)
            if gold is None:
                continue
            qrels[q["query_id"]][gold_id] = 2
            candidates: list[tuple[int, dict]] = []
            for other in by_act[gold["act_id"]]:
                if other["chunk_id"] in qrels[q["query_id"]]:
                    continue
                if span_overlap(gold, other) >= MIN_SPAN_OVERLAP:
                    qrels[q["query_id"]].setdefault(other["chunk_id"], 1)
                    continue
                distance = abs(other["position"] - gold["position"])
                same_section = (bool(other.get("section"))
                                and other["section"] == gold.get("section"))
                if distance <= NEIGHBOUR_WINDOW or same_section:
                    candidates.append((distance if distance <= NEIGHBOUR_WINDOW
                                       else 100 + distance, other))
            for _, other in sorted(candidates, key=lambda x: x[0])[:MAX_REVIEW_PER_QUERY]:
                review_rows.append({
                    "query_id": q["query_id"], "query": q["text"],
                    "kandidat_chunk_id": other["chunk_id"], "otsenka_0_1_2": "",
                    "razdel": other.get("section", ""), "punkty": other.get("units", ""),
                    "nachalo": other["text"].split("\n\n", 1)[-1][:200].replace("\n", " "),
                })

    по_пунктам = 0
    if os.path.exists(args.by_clause):
        for qid, rel in read_qrels(args.by_clause).items():
            for cid, score in rel.items():
                if cid in by_id:
                    qrels[qid][cid] = score
                    по_пунктам += 1

    # ---- 3. разметка судьи: не перебивает первые два источника ----
    судьёй = 0
    if os.path.exists(args.judge):
        известно = {qid: set(rel) for qid, rel in qrels.items()}
        for qid, rel in read_qrels(args.judge).items():
            for cid, score in rel.items():
                if cid not in by_id or cid in известно.get(qid, ()):
                    continue
                qrels[qid][cid] = score
                судьёй += 1

    # ---- проверенные человеком градации имеют приоритет над всем ----
    руками = 0
    if os.path.exists(args.grades):
        with open(args.grades, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                mark = (row.get("otsenka_0_1_2") or "").strip()
                qid = row.get("query_id")
                cid = row.get("kandidat_chunk_id") or row.get("chunk_id")
                if mark not in ("0", "1", "2") or qid not in qrels or not cid:
                    continue
                if mark == "0":
                    qrels[qid].pop(cid, None)
                else:
                    qrels[qid][cid] = int(mark)
                руками += 1

    qrels = {qid: rel for qid, rel in qrels.items() if rel}

    # ---- разметка только дополняется ----
    prev_path = os.path.join(QDIR, "qrels.tsv")
    пропало: list[tuple[str, str]] = []
    if os.path.exists(prev_path):
        for qid, rel in read_qrels(prev_path).items():
            for cid, score in rel.items():
                if score > 0 and qrels.get(qid, {}).get(cid, 0) <= 0:
                    пропало.append((qid, cid))
    if пропало and not args.allow_shrink:
        say(f"\nОТКАЗ: из разметки пропало {len(пропало)} пар, которые в ней были.")
        for qid, cid in пропало[:10]:
            say(f"   {qid} / {cid}")
        say("Прежние прогоны посчитаны по этим эталонам, и молча исчезнувший "
            "эталон выглядит как провал поиска.")
        say("Если пара удалена сознательно — `--allow-shrink`.")
        raise SystemExit(1)

    # ---- подвыборки ----
    подвыборки: dict[str, list[str]] = collections.defaultdict(list)
    for q in queries:
        if q["query_id"] not in qrels:
            continue
        подвыборки[q["podvyborka"]].append(q["query_id"])
        if q["podvyborka"] in HEADLINE:
            подвыборки["заголовочная"].append(q["query_id"])
        if q["podvyborka"] == "живые" and q.get("act") != DOMINANT_ACT:
            подвыборки[f"живые без {DOMINANT_ACT}"].append(q["query_id"])

    os.makedirs(QDIR, exist_ok=True)
    write_qrels(prev_path, qrels)
    with open(os.path.join(QDIR, "podvyborki.json"), "w", encoding="utf-8") as f:
        json.dump({k: sorted(v) for k, v in подвыборки.items()}, f,
                  ensure_ascii=False, indent=1)

    write_corpus(os.path.join(BENCH, "corpus.jsonl"), chunks)
    write_queries(os.path.join(BENCH, "queries.jsonl"),
                  [q for q in queries if q["query_id"] in qrels])
    write_qrels(os.path.join(BENCH, "qrels", "test.tsv"),
                {q: qrels[q] for q in подвыборки.get("заголовочная", [])
                 + подвыборки.get("синтетика", []) if q in qrels})
    if подвыборки.get("dev"):
        write_qrels(os.path.join(BENCH, "qrels", "dev.tsv"),
                    {q: qrels[q] for q in подвыборки["dev"]})

    if review_rows:
        path = os.path.join(QDIR, "grade_candidates.tsv")
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(review_rows[0]), delimiter="\t")
            w.writeheader()
            w.writerows(review_rows[:1000])

    # ---- отчёт ----
    say(f"размеченных запросов: {len(qrels)}, пар: {sum(len(r) for r in qrels.values())}")
    say(f"   по построению и по пунктам: {по_пунктам} пар из разметки по пунктам")
    say(f"   добавлено судьёй: {судьёй}")
    say(f"   поправлено человеком: {руками}")
    say()
    say(f"{'подвыборка':<24} {'запросов':>9} {'пар':>7} {'релевантных на запрос':>22}")
    for name in ("заголовочная", "живые", f"живые без {DOMINANT_ACT}", "ручные",
                 "невиданные акты", "синтетика", "dev"):
        ids = подвыборки.get(name, [])
        if not ids:
            continue
        пар = sum(len(qrels[q]) for q in ids)
        say(f"{name:<24} {len(ids):>9} {пар:>7} {пар / len(ids):>22.1f}")
    головная = len(подвыборки.get("заголовочная", []))
    синтетика = len(подвыборки.get("синтетика", []))
    say()
    say(f"ЗАГОЛОВОЧНАЯ ЦИФРА: {головная} размеченных запросов, "
        f"доля синтетики в ней {0:.0f}%")
    say(f"синтетика печатается отдельной строкой: {синтетика} запросов")

    # у синтетики свой риск: эталон может лежать в обучающем акте
    split_path = os.path.join(QDIR, "split.json")
    if os.path.exists(split_path):
        группа = S.group_by_act(S.load(split_path))
        for name in ("синтетика", "живые", "ручные", "невиданные акты"):
            ids = подвыборки.get(name, [])
            if not ids:
                continue
            в_обучении = sum(1 for q in ids
                             if any(группа.get(S.act_of_chunk(c)) == S.TRAIN
                                    for c in qrels[q]))
            if в_обучении:
                say(f"   ВНИМАНИЕ: в подвыборке «{name}» у {в_обучении} запросов "
                    f"эталон лежит в обучающем акте")
    say()
    say("выгружено в формате MTEB:")
    say("   data/benchmark/corpus.jsonl")
    say("   data/benchmark/queries.jsonl")
    say("   data/benchmark/qrels/test.tsv")
    say(f"кандидатов на ручную проверку градаций: {len(review_rows)}")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
