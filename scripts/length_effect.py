#!/usr/bin/env python3
"""Зависит ли провал плотного поиска от длины эталонного фрагмента.

Гипотеза этапа: нарезка сделана под размер окна, а не под смысловую единицу.
Один вектор отвечает за 466 токенов канцелярита, сигнал в нём размывается,
а BM25 с нормировкой по длине такое переносит легче.

У гипотезы есть следствие, которое проверяется на уже посчитанных выдачах,
без единой новой модели: если дело в длине, то разрыв между BM25 и плотным
поиском обязан расти с длиной эталонного фрагмента. На коротких эталонах
плотный поиск должен держаться, на длинных — проваливаться.

Проверка бесплатная и потому первая: если следствие не подтвердится,
объяснять провал длиной фрагмента нельзя, сколько бы ни было абляций.

Ограничение, которое надо держать в голове: 93% запросов синтетические,
и вопрос писался по эталонному фрагменту. Длинный фрагмент даёт генератору
больше материала, и связь «длина — качество вопроса» тут есть независимо
от поиска. Поэтому смотрим не на уровень метрики в группе, а на **разницу**
между BM25 и плотным поиском внутри одной и той же группы: обоим методам
достаётся один и тот же вопрос.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin import metrics as M  # noqa: E402
from rufin.benchmark import read_qrels  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNDIR = os.path.join(ROOT, "data", "runs")
QDIR = os.path.join(ROOT, "data", "queries")
CHUNKDIR = os.path.join(ROOT, "data", "chunks")
RAW = os.path.join(ROOT, "docs", "raw")

TOP = 10
# символов на токен, замерено на корпусе (docs/raw/build_chunks.txt)
CHARS_PER_TOKEN = 4.64


def load_run(path: str) -> dict[str, list[str]]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            out[rec["query_id"]] = rec["ranked"][:TOP]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base")
    ap.add_argument("--baseline", default="bm25")
    ap.add_argument("--dense", default="dense-bge-m3")
    ap.add_argument("--metric", default="NDCG@10")
    ap.add_argument("--groups", type=int, default=3, help="на сколько групп по длине делить")
    ap.add_argument("--seed", type=int, default=0, help="зерно бутстрэпа")
    args = ap.parse_args()

    qrels = read_qrels(os.path.join(QDIR, "qrels.tsv"))
    queries = {json.loads(l)["query_id"]: json.loads(l)
               for l in open(os.path.join(QDIR, "queries.jsonl"), encoding="utf-8") if l.strip()}

    # длина берётся у того фрагмента, по которому писался вопрос, и считается
    # по тексту целиком — вместе с шапкой, потому что кодируется именно он
    lengths: dict[str, int] = {}
    with open(os.path.join(CHUNKDIR, f"{args.config}.jsonl"), encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            lengths[c["chunk_id"]] = len(c["text"])

    runs = {}
    for tag in (args.baseline, args.dense):
        path = os.path.join(RUNDIR, f"{args.config}__{tag}.jsonl")
        if not os.path.exists(path):
            raise SystemExit(f"нет выдачи {os.path.relpath(path, ROOT)}")
        runs[tag] = load_run(path)

    known = [qid for qid in qrels if qid in queries and qid in runs[args.baseline]]
    per_q = {tag: M.per_query({q: runs[tag][q] for q in known},
                              {q: qrels[q] for q in known}) for tag in runs}

    def gold_length(qid: str) -> int | None:
        gold = queries[qid].get("gold_chunk_id")
        if gold in lengths:
            return lengths[gold]
        свои = [lengths[c] for c, s in qrels[qid].items() if s == 2 and c in lengths]
        return max(свои) if свои else None

    # порядок строк в per_query задаётся им самим, а не нами: берём его
    место = {qid: i for i, qid in enumerate(per_q[args.baseline]["_qids"].tolist())}
    rows = []
    for qid in known:
        ln = gold_length(qid)
        if ln is None or qid not in место:
            continue
        rows.append((ln, qid, место[qid]))
    rows.sort()

    report: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        report.append(s)

    say(f"нарезка: {args.config}; метрика: {args.metric}; запросов: {len(rows)}")
    say(f"длина эталона в символах: медиана {statistics.median(r[0] for r in rows):.0f}, "
        f"это примерно {statistics.median(r[0] for r in rows) / CHARS_PER_TOKEN:.0f} токенов")
    say()
    say(f"{'группа по длине эталона':<26} {'запросов':>9} {'медиана, ток.':>14} "
        f"{args.baseline:>14} {args.dense:>18} {'разрыв':>10}   {'интервал 95%':>20}")

    size = len(rows) // args.groups
    разрывы: list[tuple[str, np.ndarray]] = []
    for g in range(args.groups):
        lo = g * size
        hi = len(rows) if g == args.groups - 1 else (g + 1) * size
        часть = rows[lo:hi]
        if not часть:
            continue
        idx = [i for _, _, i in часть]
        bv = per_q[args.baseline][args.metric][idx]
        dv = per_q[args.dense][args.metric][idx]
        # разрыв берётся по каждому запросу: без поквериных значений
        # к нему не посчитать интервал, а без интервала клетку читать нельзя
        gap = dv - bv
        ci = M.bootstrap_ci(gap, seed=args.seed)
        label = (f"{часть[0][0] / CHARS_PER_TOKEN:.0f}–{часть[-1][0] / CHARS_PER_TOKEN:.0f} "
                 f"токенов")
        разрывы.append((label, gap))
        say(f"{label:<26} {len(часть):>9} "
            f"{statistics.median(x[0] for x in часть) / CHARS_PER_TOKEN:>14.0f} "
            f"{float(bv.mean()):>14.3f} {float(dv.mean()):>18.3f} {ci.mean:>+10.3f}   "
            f"[{ci.lo:+.3f}; {ci.hi:+.3f}]")

    say()
    if len(разрывы) >= 2:
        кор, длин = разрывы[0][1], разрывы[-1][1]
        # Группы состоят из разных запросов, поэтому сравнение непарное:
        # пары нет, и каждая выборка пересобирается независимо.
        ci = M.unpaired_diff_ci(длин, кор, seed=args.seed)
        вывод = ("установлено" if ci.lo * ci.hi > 0
                 else "НЕ установлено: интервал накрывает ноль")
        say(f"изменение разрыва, длинные минус короткие: {ci.mean:+.3f} "
            f"[{ci.lo:+.3f}; {ci.hi:+.3f}] — {вывод}")
        say()

    say("Читать надо предпоследний столбец: он показывает, как меняется отставание")
    say("плотного поиска от лексического при переходе к более длинным эталонам.")
    say("Уровень метрики внутри группы сравнивать между группами нельзя —")
    say("вопросы к длинным и коротким фрагментам писались по разному материалу.")
    say()
    say("Отдельная клетка опирается на полсотни запросов и потому шумная. Судить")
    say("следует по строке с изменением разрыва: она и говорит, установлен ли")
    say("эффект длины у этой модели, — а не по тому, как выглядит столбик глазами.")

    os.makedirs(RAW, exist_ok=True)
    out = os.path.join(RAW, f"length_effect_{args.config}_{args.dense}.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print(f"\nсырой вывод: {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
