#!/usr/bin/env python3
"""Проверка обучающих данных до того, как включена видеокарта.

Всё, что здесь считается, не требует ни памяти, ни карты, и каждая проверка
отвечает на вопрос, который дорого задать позже. Утечка по актам обесценивает
dev, а значит и весь подбор рецепта. Съехавший эталон учит модель находить
не то. Отсечка негативов, пропускающая всё, превращает трудные негативы
в ложные.

Запускается перед прогоном и после него — второй раз уже с файлом
подготовки D0, по которому видно, сколько негативов осталось после отсечки
и не выбросила ли она всё.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin import split as S  # noqa: E402
from rufin.training.negatives import NegativeRules, detect_scale, pick_all  # noqa: E402
from rufin.training.pairs import (  # noqa: E402
    acts_of,
    build_pairs,
    check_group,
    leaked_acts,
    read_jsonl,
    train_acts,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDIR = os.path.join(ROOT, "data", "queries")
CHUNKS = os.path.join(ROOT, "data", "chunks")
RAW = os.path.join(ROOT, "docs", "raw")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="base")
    ap.add_argument("--prepared", default=os.path.join(QDIR, "train_prepared.jsonl"),
                    help="файл подготовки D0; если его нет, проверки по негативам "
                         "пропускаются")
    ap.add_argument("--per-query", type=int, default=2)
    args = ap.parse_args()

    отчёт: list[str] = []
    беда = 0

    def say(s: str = "") -> None:
        print(s)
        отчёт.append(s)

    def плохо(s: str) -> None:
        nonlocal беда
        беда += 1
        say(f"   ПЛОХО: {s}")

    train = read_jsonl(os.path.join(QDIR, "synthetic_train.jsonl"))
    dev = read_jsonl(os.path.join(QDIR, "synthetic_dev.jsonl"))
    test = read_jsonl(os.path.join(QDIR, "synthetic_test.jsonl"))
    split = S.load(os.path.join(QDIR, "split.json"))

    say("== состав ==")
    for имя, набор in (("обучение", train), ("dev", dev), ("тест", test)):
        say(f"   {имя:<10} вопросов {len(набор):>5}, актов {len(acts_of(набор)):>5}")
    say(f"   обучающих актов в сплите: {len(train_acts(split))}")

    say()
    say("== утечка по актам ==")
    say("   фрагменты одного акта нарезаны с перекрытием, поэтому общий акт "
        "означает, что отложенный набор мерит узнавание, а не поиск")
    for имя, набор in (("dev", dev), ("тест", test)):
        общие = leaked_acts(train, набор)
        if общие:
            плохо(f"обучение и {имя} делят {len(общие)} актов: {sorted(общие)[:5]}")
        else:
            say(f"   обучение и {имя}: общих актов нет")

    say()
    say("== группы сплита ==")
    for имя, набор, группа in (("обучение", train, S.TRAIN), ("dev", dev, S.DEV)):
        чужие = check_group(набор, split, группа)
        if чужие:
            плохо(f"{имя}: {len(чужие)} вопросов не в группе {группа}, "
                  f"например {list(чужие.items())[:3]}")
        else:
            say(f"   {имя}: все вопросы в группе {группа}")
    вне_теста = {qid: г for qid, г in check_group(test, split, S.TEST_UNSEEN).items()
                 if г != S.TEST_KIND}
    if вне_теста:
        плохо(f"тест: {len(вне_теста)} вопросов не в тестовых группах")
    else:
        say(f"   тест: все вопросы в {S.TEST_UNSEEN} или {S.TEST_KIND}")

    say()
    say("== пары ==")
    путь = os.path.join(CHUNKS, f"{args.chunks}.jsonl")
    if not os.path.exists(путь):
        say(f"   нарезки {args.chunks} нет локально — пары не собираются. "
            f"Её отдаёт этап B с видеокарты (chunks_base.jsonl.gz), ставится "
            f"через `make use-chunks FILE=...`")
    else:
        тексты = {}
        with open(путь, encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                тексты[c["chunk_id"]] = c["text"]
        pairs, stats = build_pairs(train, тексты)
        for k, v in stats.items():
            say(f"   {k:<26} {v}")
        if stats["эталона нет в нарезке"]:
            плохо("часть эталонов отсутствует в нарезке: идентификатор фрагмента "
                  "позиционный, значит нарезка разошлась с той, по которой "
                  "генерировались вопросы")
        if len(pairs) != len(train):
            say(f"   в обучение пойдёт {len(pairs)} пар из {len(train)} вопросов")

    say()
    say("== негативы ==")
    if not os.path.exists(args.prepared):
        say(f"   файла подготовки нет ({os.path.relpath(args.prepared, ROOT)}): "
            f"его считает D0 на видеокарте. Проверки по негативам пропущены")
    else:
        подготовка = read_jsonl(args.prepared)
        scale = detect_scale([r["gold_score"] for r in подготовка])
        правила = NegativeRules(per_query=args.per_query, train_acts=train_acts(split))
        _, stats = pick_all(подготовка, правила, scale)
        for k, v in stats.items():
            if isinstance(v, dict):
                say(f"   {k}:")
                for k2, v2 in v.items():
                    say(f"      {k2:<34} {v2}")
            else:
                say(f"   {k:<26} {v}")
        доля = stats["хватило негативов"] / max(stats["вопросов"], 1)
        if доля < 0.5:
            плохо(f"негативов хватило только у {100 * доля:.0f}% вопросов: "
                  f"этап B будет учиться на половине выборки. Проверьте глубину "
                  f"подготовки и пороги отсечки")
        отброшено = stats["отброшено кандидатов"]
        если_бы = sum(отброшено.get(k, 0) for k in
                      ("учитель считает релевантным", "оценка близка к эталону"))
        if не_сработала := (если_бы == 0):
            плохо("отсечка по оценке учителя не отбросила ни одного кандидата: "
                  "скорее всего шкала оценок определена неверно, и негативами "
                  "служат неразмеченные позитивы")
        if not не_сработала:
            say(f"   отсечка по оценке учителя убрала {если_бы} кандидатов")

    say()
    if беда:
        say(f"ИТОГ: {беда} проблем. Обучать нельзя, пока они не разобраны.")
    else:
        say("ИТОГ: обучающие данные в порядке.")

    os.makedirs(RAW, exist_ok=True)
    with open(os.path.join(RAW, "check_train.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(отчёт) + "\n")
    print("\nсырой вывод: docs/raw/check_train.txt")
    raise SystemExit(1 if беда else 0)


if __name__ == "__main__":
    main()
