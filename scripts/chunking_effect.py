#!/usr/bin/env python3
"""Меняется ли разрыв с базовой линией при смене нарезки.

Сравнивать уровни метрики между нарезками нельзя: при дроблении цель
уменьшается — эталоном становится один фрагмент из ста тридцати шести
тысяч вместо одного из шестидесяти двух, — и попасть в него труднее обоим
методам. Вдобавок ответ, разорванный границей, засчитывается только
в одной из половин, и это свойство переноса эталонов, а не поиска.

Читать из такой абляции можно **только изменение разрыва** между плотным
поиском и BM25. Но и его нельзя брать вычитанием двух средних: у каждого
своя неопределённость, и две оценки с перекрывающимися интервалами могут
дать «разрыв сократился втрое» там, где не изменилось ничего.

Поэтому здесь считается разность разностей на одних и тех же запросах:
для каждого запроса берётся разрыв в одной нарезке и в другой, и парным
бутстрэпом оценивается их разница. Сравнение парное, потому что запрос
один и тот же — меняется только нарезка под ним.

Ничего не грузит и не считает на видеокарте: арифметика по готовым спискам.

    python3 scripts/chunking_effect.py --from base --to size-256
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin import metrics as M  # noqa: E402
from rufin.benchmark import read_qrels  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNDIR = os.path.join(ROOT, "data", "runs")
QDIR = os.path.join(ROOT, "data", "queries")
RAW = os.path.join(ROOT, "docs", "raw")


def load_run(path: str) -> dict[str, list[str]]:
    with open(path, encoding="utf-8") as f:
        return {r["query_id"]: r["ranked"] for r in map(json.loads, f)}


def qrels_path(config: str) -> str:
    """Разметка нарезки: у базовой своя, у остальных — перенесённая."""
    if config == "base":
        return os.path.join(QDIR, "qrels.tsv")
    return os.path.join(QDIR, f"qrels_{config}.tsv")


def gaps(config: str, dense: str, baseline: str, metric: str,
         qids: list[str]) -> np.ndarray:
    """Разрыв «плотный минус лексический» по каждому запросу."""
    qrels = read_qrels(qrels_path(config))
    fn = M.METRICS[metric]
    d = load_run(os.path.join(RUNDIR, f"{config}__{dense}.jsonl"))
    b = load_run(os.path.join(RUNDIR, f"{config}__{baseline}.jsonl"))
    return np.array([fn(d[q], qrels[q]) - fn(b[q], qrels[q]) for q in qids])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", default="base")
    ap.add_argument("--to", dest="dst", default="size-256")
    ap.add_argument("--dense", default="dense-e5-small")
    ap.add_argument("--baseline", default="bm25")
    ap.add_argument("--metric", default="NDCG@10")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    for config in (args.src, args.dst):
        path = qrels_path(config)
        if not os.path.exists(path):
            raise SystemExit(
                f"нет разметки {os.path.relpath(path, ROOT)}.\n"
                f"Эталоны переносятся командой `make remap CHUNKS={config}`.")

    # только запросы, размеченные в обеих нарезках: сравнение парное
    общие = sorted(set(read_qrels(qrels_path(args.src)))
                   & set(read_qrels(qrels_path(args.dst))))
    src = gaps(args.src, args.dense, args.baseline, args.metric, общие)
    dst = gaps(args.dst, args.dense, args.baseline, args.metric, общие)
    ci, p = M.paired_diff_ci(dst, src, seed=args.seed)

    report = [
        f"изменение разрыва при смене нарезки: {args.src} -> {args.dst}",
        f"метрика: {args.metric}; {args.dense} против {args.baseline}",
        f"запросов, размеченных в обеих нарезках: {len(общие)}",
        "",
        f"разрыв в нарезке {args.src:<12} {src.mean():+.3f}",
        f"разрыв в нарезке {args.dst:<12} {dst.mean():+.3f}",
        "",
        f"изменение разрыва: {ci.mean:+.3f} [{ci.lo:+.3f}; {ci.hi:+.3f}]  p = {p:.3f}",
        ("вывод: изменение установлено — интервал не накрывает ноль"
         if ci.lo * ci.hi > 0 else
         "вывод: изменение НЕ установлено — интервал накрывает ноль"),
        "",
        "Уровни метрики между нарезками не сравниваются: при дроблении цель",
        "уменьшается, и попасть в неё труднее обоим методам. Читается только",
        "изменение разрыва, и только с интервалом: два средних с перекрывающимися",
        "интервалами могут показать «сократился втрое» там, где не изменилось ничего.",
    ]
    print("\n".join(report))

    os.makedirs(RAW, exist_ok=True)
    out = os.path.join(RAW, f"chunking_effect_{args.src}_{args.dst}_{args.dense}.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print(f"\nсырой вывод: {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
