#!/usr/bin/env python3
"""Метрики по выдачам, посчитанным на видеокарте.

Тяжёлое — индексы, эмбеддинги, переранжирование — считается на Kaggle
(см. kaggle/README.md), а обратно едут только списки идентификаторов.
Этот скрипт читает их и считает то, что не требует ни памяти, ни видеокарты:
Recall@1/5/10, MRR@10, NDCG@10 с бутстрэп-интервалами и парное сравнение
каждой конфигурации с BM25 на одних и тех же запросах.

Сравнение парное, потому что конфигурации меряются на одном наборе запросов
и их ошибки скоррелированы: независимые интервалы завысили бы
неопределённость и скрыли реальную разницу.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin import metrics as M  # noqa: E402
from rufin.benchmark import read_qrels  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNDIR = os.path.join(ROOT, "data", "runs")
QDIR = os.path.join(ROOT, "data", "queries")
RESDIR = os.path.join(ROOT, "data", "results")
RAW = os.path.join(ROOT, "docs", "raw")

TOP_REPORT = 10
# порядок конфигураций в таблице: от базовой линии к самой тяжёлой
ORDER = ["bm25", "dense", "hybrid", "hybrid-rerank"]


def sort_key(name: str) -> tuple:
    for i, prefix in enumerate(ORDER):
        if name == prefix or name.startswith(prefix + "-"):
            return (i, name)
    return (len(ORDER), name)


def resolve_qrels(config: str) -> str:
    """Какую разметку брать для этой нарезки.

    Эталон записан идентификатором фрагмента, а нумерация у каждой нарезки
    своя: с разметкой базовой нарезки любая другая честно покажет нули, и ноль
    этот будет означать «эталона тут нет», а не «нарезка плохая». Поэтому для
    неосновных нарезок нужен перенос, и без него скрипт не считает вовсе —
    молчаливый ноль здесь опаснее отказа.
    """
    if config == "base":
        return os.path.join(QDIR, "qrels.tsv")
    moved = os.path.join(QDIR, f"qrels_{config}.tsv")
    if os.path.exists(moved):
        return moved
    raise SystemExit(
        f"для нарезки «{config}» нет перенесённой разметки "
        f"({os.path.relpath(moved, ROOT)}).\n"
        f"Считать по data/queries/qrels.tsv нельзя: там эталоны базовой нарезки, "
        f"и метрики выйдут нулевыми не из-за качества поиска.\n"
        f"Сделать перенос: python3 scripts/remap_qrels.py --to {config}")


def load_runs(config: str) -> dict[str, dict[str, list[str]]]:
    """Прочитать выдачи одной нарезки: файлы вида <нарезка>__<конфигурация>.jsonl."""
    out: dict[str, dict[str, list[str]]] = {}
    if not os.path.isdir(RUNDIR):
        return out
    for name in sorted(os.listdir(RUNDIR)):
        m = re.fullmatch(rf"{re.escape(config)}__(.+)\.jsonl", name)
        if not m:
            continue
        run = {}
        with open(os.path.join(RUNDIR, name), encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                run[rec["query_id"]] = rec["ranked"]
        out[m.group(1)] = run
    return dict(sorted(out.items(), key=lambda kv: sort_key(kv[0])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base", help="какая нарезка")
    ap.add_argument("--queries", default=os.path.join(QDIR, "queries.jsonl"))
    ap.add_argument("--qrels", default=None,
                    help="разметка; по умолчанию qrels.tsv для базовой нарезки "
                         "и qrels_<нарезка>.tsv для остальных")
    ap.add_argument("--baseline", default="bm25")
    ap.add_argument("--phase-b-report", default=os.path.join(RUNDIR, "report_phase_b.json"))
    args = ap.parse_args()

    os.makedirs(RESDIR, exist_ok=True)
    os.makedirs(RAW, exist_ok=True)

    qrels_path = args.qrels or resolve_qrels(args.config)

    runs = load_runs(args.config)
    if not runs:
        print(f"в {os.path.relpath(RUNDIR, ROOT)} нет выдач для нарезки «{args.config}».")
        print("Их считает этап B на Kaggle — см. kaggle/README.md.")
        return

    queries = [json.loads(l) for l in open(args.queries, encoding="utf-8") if l.strip()]
    qrels = read_qrels(qrels_path)
    known = {q["query_id"] for q in queries} & set(qrels)
    origins: dict[str, int] = {}
    for q in queries:
        if q["query_id"] in known:
            origins[q.get("origin", "не указано")] = origins.get(q.get("origin", "не указано"), 0) + 1

    report: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        report.append(s)

    say(f"нарезка: {args.config}")
    say(f"разметка: {os.path.relpath(qrels_path, ROOT)}")
    say(f"запросов с разметкой: {len(known)}  "
        f"({', '.join(f'{k}: {v}' for k, v in sorted(origins.items()))})")
    syn = origins.get("синтетический", 0)
    if len(known):
        say(f"доля синтетических: {100 * syn / len(known):.0f}%")
    say()

    trimmed = {cfg: {qid: docs[:TOP_REPORT] for qid, docs in run.items() if qid in known}
               for cfg, run in runs.items()}
    per_q = {cfg: M.per_query(run, {k: v for k, v in qrels.items() if k in known})
             for cfg, run in trimmed.items()}
    names = list(M.METRICS)

    say(f"{'конфигурация':<18} " + " ".join(f"{n:>22}" for n in names))
    for cfg, pq in per_q.items():
        say(f"{cfg:<18} " + " ".join(f"{str(M.bootstrap_ci(pq[n])):>22}" for n in names))

    base = args.baseline
    if base in per_q and len(per_q) > 1:
        say()
        say(f"разница с {base}, парный бутстрэп по запросам; p — перестановочный тест:")
        for cfg, pq in per_q.items():
            if cfg == base:
                continue
            parts = []
            for n in ("Recall@5", "NDCG@10"):
                d, p = M.paired_diff_ci(pq[n], per_q[base][n])
                mark = "" if d.lo <= 0 <= d.hi else " *"
                parts.append(f"{n} {d.mean:+.3f} [{d.lo:+.3f}; {d.hi:+.3f}] p={p:.3f}{mark}")
            say(f"   {cfg:<18} " + "   ".join(parts))
        say("   * интервал не накрывает ноль — разницу можно считать установленной")

    index_info = {}
    if os.path.exists(args.phase_b_report):
        index_info = json.load(open(args.phase_b_report, encoding="utf-8")).get("index", {})
        rows = {k: v for k, v in index_info.items() if k.startswith(args.config + "/")
                or k == "reranker"}
        if rows:
            say()
            say("индексы (посчитано на видеокарте Kaggle):")
            for k, v in rows.items():
                say(f"   {k:<26} {v}")

    result = {"config": args.config, "qrels": os.path.relpath(qrels_path, ROOT),
              "queries": len(known), "origins": origins,
              "metrics": {cfg: {n: M.bootstrap_ci(pq[n]).__dict__ for n in names}
                          for cfg, pq in per_q.items()},
              "index": index_info}
    with open(os.path.join(RESDIR, f"{args.config}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    with open(os.path.join(RAW, f"metrics_{args.config}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    say(f"\nрезультаты: data/results/{args.config}.json, "
        f"сырой вывод: docs/raw/metrics_{args.config}.txt")


if __name__ == "__main__":
    main()
