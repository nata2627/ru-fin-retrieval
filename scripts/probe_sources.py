#!/usr/bin/env python3
"""Разведка источников: доступность, объём, пригодность текстового слоя.

Запускается до сбора корпуса и отвечает на три вопроса:
  1. отвечают ли источники и как устроен перебор их каталогов;
  2. сколько там документов и какого вида;
  3. какая доля публикаций пригодна для поиска, а какая — сканы.

Результат печатается в stdout и складывается в docs/raw/, перечни актов —
в data/index/. Числа в отчётах взяты только отсюда.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import statistics
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin.http import Client  # noqa: E402
from rufin.pdftext import extract  # noqa: E402
from rufin.sources import cbr_lawacts as la  # noqa: E402
from rufin.sources import cbr_vestnik as vb

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(ROOT, "docs", "raw")
IDX = os.path.join(ROOT, "data", "index")

SOURCES = {
    "ЦБ, правовые акты":        "https://www.cbr.ru/na/",
    "ЦБ, Вестник":              "https://www.cbr.ru/about_br/publ/vestnik/",
    "ЦБ, поиск актов Вестника": "https://www.cbr.ru/about_br/publ/vestnik-search/",
    "Минфин, документы":        "https://minfin.gov.ru/ru/document/",
}


def rel(path: str) -> str:
    """Путь относительно корня проекта.

    Сырой вывод попадает в репозиторий, и абсолютные пути раскрывали бы
    расположение проекта на конкретной машине, ничего не добавляя к смыслу.
    """
    return os.path.relpath(path, ROOT)


class Tee:
    """Печать одновременно на экран и в файл: отчёт и сырой вывод не должны разойтись."""

    def __init__(self, path: str):
        self.f = open(path, "w", encoding="utf-8")

    def __call__(self, *parts):
        line = " ".join(str(p) for p in parts)
        print(line)
        self.f.write(line + "\n")
        self.f.flush()


def check_availability(client: Client, out: Tee) -> None:
    out("== доступность источников ==")
    for name, url in SOURCES.items():
        try:
            r = client.get(url)
            out(f"  {name:<26} HTTP {r.status_code}  {len(r.content) / 1024:.0f} КБ")
        except Exception as e:  # noqa: BLE001
            out(f"  {name:<26} НЕДОСТУПЕН: {e}")


def collect_lawacts(client: Client, out: Tee, cache: bool) -> list[dict]:
    path = os.path.join(IDX, "cbr_lawacts_catalog.jsonl")
    if cache and os.path.exists(path):
        docs = [json.loads(l) for l in open(path, encoding="utf-8")]
        out(f"\n== каталог «Правовые акты»: {len(docs)} записей (из data/index) ==")
    else:
        out("\n== обход каталога «Правовые акты» ==")
        docs = list(la.iter_catalog(client))
        with open(path, "w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        out(f"  собрано {len(docs)} записей -> {rel(path)}")
    for vid, n in collections.Counter(d["vid"] for d in docs).most_common():
        out(f"  {vid:<48} {n:>5}")
    return docs


def collect_vestnik(client: Client, out: Tee, cache: bool) -> list[dict]:
    path = os.path.join(IDX, "cbr_vestnik_acts.jsonl")
    if cache and os.path.exists(path):
        acts = [json.loads(l) for l in open(path, encoding="utf-8")]
        out(f"\n== перечень актов «Вестника»: {len(acts)} записей (из data/index) ==")
    else:
        out("\n== сбор перечня актов «Вестника» ==")
        acts = vb.fetch_acts_index(client)
        with open(path, "w", encoding="utf-8") as f:
            for a in acts:
                f.write(json.dumps(a, ensure_ascii=False) + "\n")
        out(f"  собрано {len(acts)} актов -> {rel(path)}")

    by_type = collections.Counter(vb.act_type(a["number"]) for a in acts)
    names = {"П": "Положение", "У": "Указание", "И": "Инструкция", "Т": "Письмо",
             "МР": "Методические рекомендации", "ОР": "Официальное разъяснение",
             "ФЗ": "Федеральный закон"}
    out("  по видам:")
    for k, n in by_type.most_common(8):
        out(f"    {k:<4} {n:>5}  {names.get(k, '')}")
    base = [a for a in acts if not vb.is_amendment(a)]
    out(f"  самостоятельных (не правки): {len(base)} из {len(acts)}")
    out(f"  выпусков задействовано: {len({a['issue'] for a in acts})}")
    years = [vb.year(a) for a in acts if vb.year(a) > 1990]
    out(f"  годы публикации: {min(years)}–{max(years)}")
    return acts


def probe_quality(client: Client, out: Tee, title: str, urls: list[tuple[str, str]]) -> list[dict]:
    """Скачать выборку и оценить текстовый слой. Файлы не сохраняются."""
    out(f"\n== качество текстового слоя: {title} (выборка {len(urls)}) ==")
    out(f"  {'стр':>4} {'симв':>8} {'кирилл':>7}  вердикт")
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (url, label) in enumerate(urls):
            try:
                r = client.get(url)
            except Exception as e:  # noqa: BLE001
                out(f"  ошибка {label[:40]}: {e}")
                continue
            p = os.path.join(tmp, f"{i}.pdf")
            with open(p, "wb") as f:
                f.write(r.content)
            t = extract(p)
            rows.append({"label": label, "pages": t.pages, "chars": t.chars,
                         "cyrillic": round(t.cyrillic, 3), "ocr": t.has_ocr_layer,
                         "usable": t.usable, "verdict": t.verdict, "bytes": len(r.content)})
            out(f"  {t.pages:>4} {t.chars:>8} {t.cyrillic:>6.0%}  {t.verdict:<26} {label[:52]}")
            os.remove(p)
    ok = [r for r in rows if r["usable"]]
    out(f"  пригодных: {len(ok)}/{len(rows)}" + (f" = {100 * len(ok) / len(rows):.0f}%" if rows else ""))
    if ok:
        ch = [r["chars"] for r in ok]
        out(f"  символов: медиана {statistics.median(ch):.0f}, сумма {sum(ch)}")
    bad = collections.Counter(r["verdict"] for r in rows if not r["usable"])
    if bad:
        out(f"  причины брака: {dict(bad)}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=12, help="документов в каждой выборке качества")
    ap.add_argument("--pause", type=float, default=1.5, help="пауза между запросами, с")
    ap.add_argument("--refresh", action="store_true", help="перекачать перечни, не брать из data/index")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    os.makedirs(RAW, exist_ok=True)
    os.makedirs(IDX, exist_ok=True)
    out = Tee(os.path.join(RAW, "probe_sources.txt"))
    client = Client(pause=args.pause)
    rnd = random.Random(args.seed)

    check_availability(client, out)
    docs = collect_lawacts(client, out, cache=not args.refresh)
    acts = collect_vestnik(client, out, cache=not args.refresh)

    # выборка 1: отдельные файлы содержательных актов в разделе «Правовые акты»
    heavy = [d for d in docs if d["vid"] in
             ("Положение Банка России", "Указание Банка России", "Инструкция Банка России")]
    q1 = probe_quality(client, out, "отдельные файлы Положений/Указаний/Инструкций",
                       [(d["url"], f"{d['number']} {d['title']}") for d in rnd.sample(heavy, args.sample)])

    # выборка 2: те же виды актов, но взятые из выпусков «Вестника»
    corpus = vb.select_corpus(acts)
    pick = rnd.sample([a for a in corpus if vb.act_type(a["number"]) in ("П", "И")], args.sample)
    q2 = probe_quality(client, out, "выпуски «Вестника» с Положениями/Инструкциями",
                       [(a["issue_url"], f"{a['number']} вып. {a['issue']}") for a in pick])

    out("\n== состав предлагаемого корпуса ==")
    iss = {a["issue"] for a in corpus}
    out(f"  самостоятельные П/И/У/МР/ОР с 2013 года: {len(corpus)} актов в {len(iss)} выпусках")
    out(f"  по видам: {dict(collections.Counter(vb.act_type(a['number']) for a in corpus).most_common())}")
    if q2:
        mb = statistics.mean([r["bytes"] for r in q2]) / 1048576
        out(f"  выкачать выпуски: ≈ {len(iss) * mb:.0f} МБ, "
            f"время при паузе {args.pause} с ≈ {len(iss) * (args.pause + 0.5) / 60:.0f} мин")

    json.dump({"lawacts_sample": q1, "vestnik_sample": q2},
              open(os.path.join(RAW, "probe_quality.json"), "w"), ensure_ascii=False, indent=1)
    out(f"\nсырой вывод: {os.path.join('docs', 'raw', 'probe_sources.txt')}")


if __name__ == "__main__":
    main()
