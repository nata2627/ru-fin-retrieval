#!/usr/bin/env python3
"""Перенос эталонов с одной нарезки на другую.

Зачем это нужно, показывает простой опыт: `make metrics CHUNKS=size-256`
честно посчитает метрики по нарезке на 256 токенов с исходной разметкой
и покажет нули. Не потому, что нарезка плохая, а потому, что эталон
записан идентификатором вида «590-П_28062017#0034», а такого фрагмента
в нарезке на 256 токенов не существует: нумерация там своя.

Скрипт переносит разметку по символьным границам — правило и его причины
описаны в `rufin.qrels_remap`. Исходный `qrels.tsv` не трогается никогда:
он остаётся истиной, а перенос кладётся рядом отдельным файлом.

    python3 scripts/remap_qrels.py --to size-256

Сам перенос ничего не считает и не грузит: арифметика над парами чисел.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin.benchmark import read_qrels, write_qrels  # noqa: E402
from rufin.qrels_remap import RemapStats, Span, remap  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDIR = os.path.join(ROOT, "data", "queries")
CHUNKDIR = os.path.join(ROOT, "data", "chunks")
RAW = os.path.join(ROOT, "docs", "raw")

# Какую долю эталонного текста должен содержать кандидат, чтобы получить
# оценку 1 без участия человека.
#
# Порог 0,30, которым сборка разметки отбирает соседей внутри одной нарезки,
# здесь не годится. При дроблении 512 -> 256 эталон делится примерно пополам,
# и каждая половина покрывает около 0,45–0,50 эталона — порог 0,30 записал бы
# в релевантные обе, а с учётом перекрытия нарезки и третий кусок. Ответ при
# этом лежит в одном из них, и остальное было бы догадкой; заодно у нарезки
# на 256 токенов оказалось бы вдвое больше правильных ответов на запрос,
# чем у базовой, и сравнивать их стало бы нельзя.
#
# Порог 0,50 означает «кандидат содержит больше половины эталонного текста».
# При дроблении такое бывает только за счёт перекрытия нарезки, то есть
# когда эталон почти целиком попал в два куска; при укрупнении (256 -> 512)
# кандидат содержит эталон целиком и проходит с запасом.
DEFAULT_THRESHOLD = 0.50


def load_spans(path: str, keep: set[str] | None = None,
               acts: set[str] | None = None) -> dict[str, Span]:
    """Прочитать границы фрагментов, не держа в памяти тексты.

    Нарезка — это четверть гигабайта текста, а нужны из неё четыре поля.
    Поэтому файл читается построчно, и от строки остаётся отрезок.
    """
    out: dict[str, Span] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            if keep is not None and c["chunk_id"] not in keep:
                continue
            if acts is not None and c["act_id"] not in acts:
                continue
            out[c["chunk_id"]] = Span.from_chunk(c)
    return out


def histogram(values: list[float], edges: tuple[float, ...]) -> list[tuple[str, int]]:
    rows = []
    for lo, hi in zip((0.0,) + edges, edges + (1.01,)):
        rows.append((f"[{lo:.2f}; {hi:.2f})",
                     sum(1 for v in values if lo <= v < hi)))
    return rows


def describe(stats: RemapStats, threshold: float, src: str, dst: str,
             src_chunks: int, dst_chunks: int) -> list[str]:
    say: list[str] = []
    say.append(f"перенос эталонов: {src} -> {dst}")
    say.append(f"фрагментов в нарезках: {src_chunks} -> {dst_chunks}")
    say.append(f"порог для оценки 1: {threshold:.2f} доли эталона")
    say.append("")
    say.append(f"эталонов на входе:            {stats.entries}")
    say.append(f"   перенесено точь-в-точь:    {stats.exact}")
    say.append(f"   не нашлось пересечений:    {stats.lost}")
    say.append(f"   нет в исходной нарезке:    {stats.unknown}")
    say.append(f"запросов: {stats.queries_in} -> {stats.queries_out}")
    say.append(f"пар на выходе: {stats.pairs_out} "
               f"(оценка 2: {stats.grade_two}, оценка 1: {stats.grade_one})")
    if stats.best_shares:
        say.append("")
        say.append("доля эталона в главном кандидате (тот, что получил оценку 2):")
        say.append(f"   медиана {statistics.median(stats.best_shares):.2f}, "
                   f"минимум {min(stats.best_shares):.2f}, "
                   f"максимум {max(stats.best_shares):.2f}")
        for label, n in histogram(stats.best_shares, (0.25, 0.5, 0.75, 0.95)):
            say.append(f"   {label} {n}")
    if stats.extra_shares:
        say.append("")
        say.append("доля эталона в остальных пересекающихся кандидатах "
                   f"(всего {len(stats.extra_shares)}); "
                   f"порог {threshold:.2f} пропускает "
                   f"{sum(1 for v in stats.extra_shares if v >= threshold)}:")
        for label, n in histogram(stats.extra_shares, (0.25, 0.5, 0.75, 0.95)):
            say.append(f"   {label} {n}")
    if stats.lost_ids:
        say.append("")
        say.append("эталоны без пересечений (первые 20):")
        say += ["   " + s for s in stats.lost_ids[:20]]
    if stats.unknown_ids:
        say.append("")
        say.append("эталоны, которых нет в исходной нарезке (первые 20):")
        say += ["   " + s for s in stats.unknown_ids[:20]]
    return say


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", default="base", help="исходная нарезка")
    ap.add_argument("--to", dest="dst", required=True, help="целевая нарезка")
    ap.add_argument("--qrels", default=os.path.join(QDIR, "qrels.tsv"),
                    help="разметка, привязанная к исходной нарезке")
    ap.add_argument("--out", default=None,
                    help="куда писать; по умолчанию data/queries/qrels_<нарезка>.tsv")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help="доля эталона, ниже которой кандидат не получает оценку 1")
    ap.add_argument("--chunks-dir", default=CHUNKDIR)
    ap.add_argument("--report", default=None,
                    help="куда писать сырой вывод; по умолчанию docs/raw/remap_<нарезка>.txt")
    args = ap.parse_args()

    src_path = os.path.join(args.chunks_dir, f"{args.src}.jsonl")
    dst_path = os.path.join(args.chunks_dir, f"{args.dst}.jsonl")
    for path, config in ((src_path, args.src), (dst_path, args.dst)):
        if not os.path.exists(path):
            raise SystemExit(
                f"нет нарезки «{config}»: {os.path.relpath(path, ROOT)}.\n"
                "Нарезки не пересобираются локально — версия токенизатора другая "
                "и границы сдвинутся. Выгрузить с Kaggle "
                f"(export_chunks.py --config {config}) и поставить через "
                f"make use-chunks CHUNKS={config} FILE=...")

    qrels = read_qrels(args.qrels)
    gold_ids = {cid for d in qrels.values() for cid in d}
    source = load_spans(src_path, keep=gold_ids)
    acts = {s.act_id for s in source.values()}
    target = load_spans(dst_path, acts=acts)

    moved, stats = remap(qrels, source, list(target.values()), args.threshold)

    out_path = args.out or os.path.join(QDIR, f"qrels_{args.dst}.tsv")
    write_qrels(out_path, moved)

    report = describe(stats, args.threshold, args.src, args.dst,
                      sum(1 for _ in open(src_path, encoding="utf-8")),
                      sum(1 for _ in open(dst_path, encoding="utf-8")))
    report.append("")
    report.append(f"записано: {os.path.relpath(out_path, ROOT)}")
    print("\n".join(report))

    os.makedirs(RAW, exist_ok=True)
    raw_path = args.report or os.path.join(RAW, f"remap_{args.dst}.txt")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print(f"сырой вывод: {os.path.relpath(raw_path, ROOT)}")


if __name__ == "__main__":
    main()
