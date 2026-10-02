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

# Порядок подвыборок в разбивке. Заголовочная идёт первой, прежняя
# синтетика последней.
#
# Осторожно с названиями. «Заголовочная» это живые плюс ручные плюс
# невиданные акты, и невиданные акты сгенерированы языковой моделью
# по отложенным актам. То есть заголовочная цифра синтетики НЕ лишена,
# и доля печатается под таблицей числом, а не словом. Строка «синтетика»
# это только прежние 150 вопросов, оставленные для сопоставимости
# с ранними прогонами.
SUBSET_ORDER = ["заголовочная", "живые", "живые без 590-П", "ручные",
                "невиданные акты", "dev", "синтетика"]
# Минимум запросов, при котором подвыборку имеет смысл печатать: на десяти
# запросах доверительный интервал шире самой метрики.
MIN_SUBSET = 20


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
    ap.add_argument("--subsets", default=os.path.join(QDIR, "podvyborki.json"),
                    help="состав подвыборок; считает `make bench`")
    ap.add_argument("--only", default=None,
                    help="считать ТОЛЬКО по этой подвыборке, например dev. "
                         "Нужно на время подбора рецепта обучения: тест "
                         "трогается один раз, в самом конце, и цель, которая "
                         "считает по всему размеченному, слишком легко "
                         "запустить по привычке")
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
    subsets: dict[str, list[str]] = {}
    if os.path.exists(args.subsets):
        subsets = json.load(open(args.subsets, encoding="utf-8"))
    if args.only and args.only not in subsets:
        raise SystemExit(
            f"нет подвыборки «{args.only}». Есть: {', '.join(subsets) or 'никаких'}.\n"
            f"Состав подвыборок считает `make bench`.")
    # Метрика считается только по запросам, для которых выдача есть у всех
    # конфигураций: сравнение парное, и разные знаменатели сделали бы его
    # бессмысленным. Размеченные запросы без выдачи — это не ноль качества,
    # а непосчитанный прогон, и о них надо сказать вслух.
    covered = set.intersection(*(set(run) for run in runs.values()))
    labelled = {q["query_id"] for q in queries} & set(qrels)
    known = labelled & covered
    без_выдачи = sorted(labelled - covered)
    if args.only:
        known &= set(subsets[args.only])
        без_выдачи = sorted(set(без_выдачи) & set(subsets[args.only]))
        if not known:
            raise SystemExit(f"в подвыборке «{args.only}» нет ни одного запроса "
                             f"с разметкой и выдачей у всех конфигураций")
    origins: dict[str, int] = {}
    for q in queries:
        if q["query_id"] in known:
            origins[q.get("origin", "не указано")] = origins.get(q.get("origin", "не указано"), 0) + 1

    report: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        report.append(s)

    say(f"нарезка: {args.config}")
    if args.only:
        say(f"ТОЛЬКО подвыборка «{args.only}»: остальные запросы в метрику "
            f"не входят вовсе")
    say(f"разметка: {os.path.relpath(qrels_path, ROOT)}")
    say(f"запросов с разметкой и выдачей: {len(known)}  "
        f"({', '.join(f'{k}: {v}' for k, v in sorted(origins.items()))})")
    if без_выдачи:
        say(f"размечено, но выдачи нет: {len(без_выдачи)} — эти запросы в метрику "
            f"не входят вовсе")
        say(f"   нужен новый прогон этапа B по нынешнему queries.jsonl "
            f"({len(queries)} запросов)")
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

    # ---- разбивка по подвыборкам ----
    # Усреднять живые вопросы с синтетическими нельзя: это три разных речевых
    # режима, и одна цифра по ним говорит о смеси, которой не существует.
    # Заголовочная цифра считается по живым, ручным и невиданным актам;
    # синтетика печатается отдельной строкой для сопоставимости с прежними
    # прогонами.
    подвыборки: dict[str, dict] = {}
    if subsets and not args.only:
        say()
        say("по подвыборкам (NDCG@10 с интервалом):")
        имена = [n for n in SUBSET_ORDER if n in subsets] + \
                [n for n in subsets if n not in SUBSET_ORDER]
        шапка = [n for n in имена if len(set(subsets[n]) & known) >= MIN_SUBSET]
        малые = [(n, len(set(subsets[n]) & known)) for n in имена if n not in шапка]
        if not шапка:
            say("   ни одна подвыборка не набрала "
                f"{MIN_SUBSET} размеченных запросов")
        else:
            say(f"{'конфигурация':<18} " + " ".join(f"{n[:20]:>22}" for n in шапка))
            for cfg, run in trimmed.items():
                клетки = []
                for name in шапка:
                    ids = set(subsets[name]) & known
                    pq = M.per_query({k: v for k, v in run.items() if k in ids},
                                     {k: v for k, v in qrels.items() if k in ids})
                    ci = M.bootstrap_ci(pq["NDCG@10"])
                    подвыборки.setdefault(name, {})[cfg] = ci.__dict__
                    клетки.append(f"{str(ci):>22}")
                say(f"{cfg:<18} " + " ".join(клетки))
            say(f"{'запросов':<18} " + " ".join(
                f"{len(set(subsets[n]) & known):>22}" for n in шапка))
        if малые:
            say("   не напечатаны (меньше "
                f"{MIN_SUBSET} размеченных запросов): "
                + ", ".join(f"{n} — {k}" for n, k in малые))
        head = set(subsets.get("заголовочная", ())) & known
        if head:
            по_id = {q["query_id"]: q for q in queries}
            син = sum(1 for qid in head
                      if по_id.get(qid, {}).get("origin") == "синтетический")
            say("   заголовочная = живые + ручные + невиданные акты, "
                f"из них сгенерировано моделью {син} из {len(head)} "
                f"({100 * син / len(head):.0f}%)")
            say("   невиданные акты тоже написаны моделью, но по актам, "
                "которых не было в обучении")

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

    хвост = f"_{args.only}" if args.only else ""
    result = {"config": args.config, "qrels": os.path.relpath(qrels_path, ROOT),
              "only": args.only,
              "queries": len(known), "origins": origins,
              "podvyborki": подвыборки,
              "metrics": {cfg: {n: M.bootstrap_ci(pq[n]).__dict__ for n in names}
                          for cfg, pq in per_q.items()},
              "index": index_info}
    with open(os.path.join(RESDIR, f"{args.config}{хвост}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    with open(os.path.join(RAW, f"metrics_{args.config}{хвост}.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    say(f"\nрезультаты: data/results/{args.config}{хвост}.json, "
        f"сырой вывод: docs/raw/metrics_{args.config}{хвост}.txt")


if __name__ == "__main__":
    main()
