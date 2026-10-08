#!/usr/bin/env python3
"""Сколько качества на живых вопросах держится на номере акта.

Опыт парный: один и тот же вопрос в трёх видах — как задан, без номеров,
без ссылок (см. scripts/live_variants.py). Сравнивать «вопросы с номером»
против «вопросов без номера» как две группы нельзя: группы отличаются
не только номером, их 112 и 30, и вторая вообще труднее — по объединённым
выдачам всех конфигураций в ней находимы далеко не все эталоны. Правка
же меняет ровно одно, и запросы остаются те же.

Что здесь считается:

* метрика каждой конфигурации в каждом варианте;
* парная разница «вариант минус исходный вид» внутри конфигурации —
  это и есть вклад номера;
* та же разница отдельно по вопросам, где номер акта был назван,
  и где не был;
* проверка самой правки: на вопросах, текст которых она не изменила,
  разница обязана быть ровно нулевой. Группа «номер не назван» для этой
  проверки не годится: номера акта там нет, а указание на пункт
  («согласно п. 3.12») есть, и правка его убирает;
* парное сравнение конфигураций внутри варианта без номеров — там видно,
  проигрывает ли дообученная модель, когда лексической подсказки нет.
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
from rufin.references import has_act_number  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNDIR = os.path.join(ROOT, "data", "runs")
QDIR = os.path.join(ROOT, "data", "queries")
REPORT = os.path.join(ROOT, "docs", "raw", "act_number_effect.txt")

TOP = 10
METRIC = "NDCG@10"
ORDER = ["orig", "bez-nomerov", "bez-ssylok"]
CFG_ORDER = ["bm25", "dense-e5-small", "dense-rufin", "dense-user-bge-m3",
             "hybrid", "hybrid-rufin", "hybrid-rerank"]


def cfg_key(name: str) -> tuple:
    return (CFG_ORDER.index(name) if name in CFG_ORDER else len(CFG_ORDER), name)


def load_variant_runs(prefix: str) -> dict[str, dict[str, dict[str, list[str]]]]:
    """Выдачи по вариантам: конфигурация -> вариант -> запрос -> список.

    Идентификатор в файле составной — «bez-nomerov:exp0133», — потому что
    на видеокарте все варианты считаются одним проходом: индекс BM25
    строится один раз, а не трижды.
    """
    out: dict[str, dict[str, dict[str, list[str]]]] = {}
    for name in sorted(os.listdir(RUNDIR)):
        m = re.fullmatch(rf"{re.escape(prefix)}__(.+)\.jsonl", name)
        if not m:
            continue
        by_variant: dict[str, dict[str, list[str]]] = {}
        with open(os.path.join(RUNDIR, name), encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                variant, _, qid = rec["query_id"].partition(":")
                by_variant.setdefault(variant, {})[qid] = rec["ranked"][:TOP]
        out[m.group(1)] = by_variant
    return dict(sorted(out.items(), key=lambda kv: cfg_key(kv[0])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="live-variants")
    ap.add_argument("--queries", default=os.path.join(QDIR, "queries.jsonl"))
    ap.add_argument("--qrels", default=os.path.join(QDIR, "qrels.tsv"))
    ap.add_argument("--subsets", default=os.path.join(QDIR, "podvyborki.json"))
    ap.add_argument("--variants", default=os.path.join(QDIR, "live_variants.jsonl"),
                    help="тексты вариантов: нужны, чтобы отделить вопросы, "
                         "которых правка не коснулась")
    ap.add_argument("--subset", default="живые")
    ap.add_argument("--metric", default=METRIC, choices=sorted(M.METRICS))
    args = ap.parse_args()

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    runs = load_variant_runs(args.prefix)
    if not runs:
        raise SystemExit(
            f"нет файлов {args.prefix}__*.jsonl в {os.path.relpath(RUNDIR, ROOT)}.\n"
            f"Варианты вопросов собирает scripts/live_variants.py, выдачи "
            f"плотного поиска — scripts/dense_run.py, остальные — Kaggle.")

    qrels_all = read_qrels(args.qrels)
    with open(args.queries, encoding="utf-8") as f:
        texts = {r["query_id"]: r["text"]
                 for r in (json.loads(l) for l in f if l.strip())}
    with open(args.subsets, encoding="utf-8") as f:
        ids = json.load(f)[args.subset]
    variant_text: dict[tuple[str, str], str] = {}
    with open(args.variants, encoding="utf-8") as f:
        for rec in (json.loads(l) for l in f if l.strip()):
            variant_text[(rec["variant"], rec["base_id"])] = rec["text"]

    # Запросы, по которым считаем: из подвыборки, с разметкой и с выдачей
    # во всех вариантах всех конфигураций. Разные знаменатели сделали бы
    # парное сравнение бессмысленным.
    covered = set.intersection(*(set(v) for cfg in runs.values()
                                 for v in cfg.values()))
    known = [q for q in ids if q in qrels_all and q in covered]
    missing = [q for q in ids if q not in covered]
    qrels = {q: qrels_all[q] for q in known}
    with_num = [q for q in known if has_act_number(texts[q])]
    without = [q for q in known if q not in set(with_num)]

    say(f"подвыборка «{args.subset}»: {len(known)} вопросов с разметкой и выдачами")
    if missing:
        say(f"без выдачи хотя бы в одном варианте: {len(missing)} — не считаются")
    say(f"из них номер акта назван: {len(with_num)}, не назван: {len(without)}")
    say(f"метрика: {args.metric}, глубина {TOP}")
    say(f"конфигураций: {len(runs)} ({', '.join(runs)})")
    say()

    variants = [v for v in ORDER if all(v in cfg for cfg in runs.values())]

    def per_q(cfg: str, variant: str, subset: list[str]) -> dict:
        run = {q: runs[cfg][variant][q] for q in subset}
        return M.per_query(run, {q: qrels[q] for q in subset})

    say("Метрика по вариантам вопроса")
    say(f"{'конфигурация':<20}" + "".join(f"{v:>24}" for v in variants))
    cache: dict[tuple[str, str], dict] = {}
    for cfg in runs:
        row = []
        for v in variants:
            cache[(cfg, v)] = pq = per_q(cfg, v, known)
            row.append(str(M.bootstrap_ci(pq[args.metric])))
        say(f"{cfg:<20}" + "".join(f"{c:>24}" for c in row))

    for v in variants[1:]:
        say()
        say(f"Вклад правки «{v}»: парная разница с исходным видом вопроса, "
            f"p — перестановочный тест")
        say(f"{'конфигурация':<20}{'все вопросы':>26}{'p':>7}"
            f"{'номер назван':>26}{'p':>7}{'номер не назван':>26}{'p':>7}")
        for cfg in runs:
            parts = []
            for subset in (known, with_num, without):
                a = per_q(cfg, v, subset)[args.metric]
                b = per_q(cfg, "orig", subset)[args.metric]
                d, p = M.paired_diff_ci(a, b)
                parts.append(f"{str(d):>26}{p:>7.3f}")
            say(f"{cfg:<20}" + "".join(parts))
        # Контроль: там, где правка текст не изменила, выдача обязана
        # совпасть с исходной до последнего места.
        same = [q for q in known
                if variant_text[(v, q)] == variant_text[("orig", q)]]
        say(f"  правка не изменила текст у {len(same)} вопросов из {len(known)}; "
            f"разница на них:")
        for cfg in runs:
            d, _ = M.paired_diff_ci(per_q(cfg, v, same)[args.metric],
                                    per_q(cfg, "orig", same)[args.metric])
            say(f"    {cfg:<20} {d.mean:+.3f}")

    if len(runs) > 1:
        baseline = "dense-rufin" if "dense-rufin" in runs else list(runs)[0]
        for v in variants:
            say()
            say(f"Вариант «{v}»: {baseline} против остальных, парно")
            for cfg in runs:
                if cfg == baseline:
                    continue
                d, p = M.paired_diff_ci(cache[(baseline, v)][args.metric],
                                        cache[(cfg, v)][args.metric])
                mark = " *" if (d.lo > 0) == (d.hi > 0) else ""
                say(f"  против {cfg:<20} {str(d):>26} p={p:.3f}{mark}")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    say()
    say(f"отчёт: {os.path.relpath(REPORT, ROOT)}")


if __name__ == "__main__":
    main()
