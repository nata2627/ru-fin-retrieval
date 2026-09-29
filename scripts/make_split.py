#!/usr/bin/env python3
"""Сплит корпуса по актам: какие акты идут в обучение, какие отложены.

Сплит режется по актам, а не по фрагментам: фрагменты одного акта нарезаны
с перекрытием и делят друг с другом текст, так что «отложенный фрагмент»
обучающего акта модель уже видела — просто с другой границей.

Что откладывается и почему.

**Акты живых вопросов.** Главная цифра проекта считается по вопросам
из «Разъяснений», и все акты, про которые в них спрашивают, обязаны быть
вне обучения. Иначе прирост от обучения будет отчасти узнаванием текста,
на котором оно шло, и отделить одно от другого будет нечем. Это дорого:
590-П — самый обсуждаемый акт корпуса, и он уходит целиком.

**Целый класс документов.** Все Инструкции: их 44, и они отличаются
от Указаний и по языку, и по структуре. Перенос на новый акт и перенос
на новый жанр — разные вещи, и мерить их надо порознь.

**Невиданные акты** — случайная доля остальных, стратифицированная по виду
документа: на них проверяется обобщение против запоминания.

**dev** — тоже отложенные целиком акты, а не отложенные фрагменты обучающих.
По dev принимается или отклоняется каждый этап рецепта обучения, и dev,
собранный из фрагментов обучающего акта, принял бы рецепт, который просто
запомнил этот акт.

Сплит фиксируется файлом и проверяется тестами `tests/test_split.py`.
Пересобирать его нужно после каждой новой разметки и обязательно **до**
генерации обучающей выборки: проверка «ни один акт с эталоном теста
не лежит в train» иначе окажется нарушенной задним числом.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin import split as S  # noqa: E402
from rufin.benchmark import read_qrels  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDIR = os.path.join(ROOT, "data", "queries")
CHUNKDIR = os.path.join(ROOT, "data", "chunks")
REPORT = os.path.join(ROOT, "docs", "raw", "make_split.txt")

# Вид документа, откладываемый целиком: Инструкции.
HELD_KIND = "И"
KIND_NAMES = {"У": "Указания", "П": "Положения", "И": "Инструкции",
              "МР": "Методические рекомендации", "ОР": "Официальные разъяснения"}


def read_acts(chunks_path: str) -> dict[str, dict]:
    """Акты и их размер в фрагментах — прямо из нарезки, по строке."""
    acts: dict[str, dict] = {}
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            a = acts.setdefault(c["act_id"], {"act_id": c["act_id"], "number": c["number"],
                                              "type": c["type"], "title": c["title"],
                                              "date": c["date"], "chunks": 0})
            a["chunks"] += 1
    return acts


def live_act_ids(acts: dict[str, dict], qrels_paths: list[str]) -> set[str]:
    """Акты, на которые указывают живые вопросы.

    Два источника, и оба нужны. Первый — привязка вопроса к акту по номеру
    (`explan_live.jsonl`): она известна до всякой разметки. Второй — сама
    разметка: эталон, подобранный пулом, может указать на акт, который
    в привязке не назывался, и такой акт тоже обязан быть вне обучения.
    """
    by_number: dict[str, str] = {}
    for a in acts.values():
        by_number.setdefault(a["number"], a["act_id"])

    out: set[str] = set()
    path = os.path.join(QDIR, "explan_live.jsonl")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                q = json.loads(line)
                if q.get("act") and q["act"] in by_number:
                    out.add(by_number[q["act"]])
    for qrels_path in qrels_paths:
        if not os.path.exists(qrels_path):
            continue
        for qid, rel in read_qrels(qrels_path).items():
            if not qid.startswith(("exp", "man")):
                continue
            for chunk_id in rel:
                act_id = S.act_of_chunk(chunk_id)
                if act_id in acts:
                    out.add(act_id)
    return out


def take_share(pool: list[dict], share: float, rnd: random.Random) -> list[str]:
    """Отобрать акты на заданную долю фрагментов, по виду документа отдельно.

    Стратификация по виду нужна, чтобы отложенная часть не оказалась
    случайно из одних Указаний: виды различаются и по языку, и по длине,
    и по тому, насколько их текст похож на вопрос практика.
    """
    chosen: list[str] = []
    by_kind: dict[str, list[dict]] = collections.defaultdict(list)
    for a in pool:
        by_kind[a["type"]].append(a)
    for kind in sorted(by_kind):
        group = sorted(by_kind[kind], key=lambda a: a["act_id"])
        rnd.shuffle(group)
        need = share * sum(a["chunks"] for a in group)
        got = 0
        for a in group:
            if got >= need:
                break
            chosen.append(a["act_id"])
            got += a["chunks"]
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="base")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--unseen-share", type=float, default=0.08,
                    help="доля фрагментов в «невиданные акты»")
    ap.add_argument("--dev-share", type=float, default=0.04,
                    help="доля фрагментов в dev")
    ap.add_argument("--out", default=os.path.join(QDIR, "split.json"))
    args = ap.parse_args()

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    acts = read_acts(os.path.join(CHUNKDIR, f"{args.chunks}.jsonl"))
    total_chunks = sum(a["chunks"] for a in acts.values())
    say(f"нарезка: {args.chunks}; актов {len(acts)}, фрагментов {total_chunks}")

    live = live_act_ids(acts, [os.path.join(QDIR, "qrels.tsv"),
                               os.path.join(QDIR, "qrels_by_clause.tsv")])
    kind = {a["act_id"] for a in acts.values()
            if a["type"] == HELD_KIND and a["act_id"] not in live}

    rnd = random.Random(args.seed)
    rest = [a for a in acts.values() if a["act_id"] not in live and a["act_id"] not in kind]
    unseen = set(take_share(rest, args.unseen_share, rnd))
    rest = [a for a in rest if a["act_id"] not in unseen]
    dev = set(take_share(rest, args.dev_share, rnd))
    train = {a["act_id"] for a in rest if a["act_id"] not in dev}

    groups = {S.TRAIN: sorted(train), S.DEV: sorted(dev), S.TEST_LIVE: sorted(live),
              S.TEST_KIND: sorted(kind), S.TEST_UNSEEN: sorted(unseen)}

    stats = {}
    for name, ids in groups.items():
        n = sum(acts[i]["chunks"] for i in ids)
        stats[name] = {"актов": len(ids), "фрагментов": n,
                       "доля фрагментов": round(n / total_chunks, 3)}

    split = {
        "нарезка": args.chunks,
        "seed": args.seed,
        "правило": {
            "по чему режется": "по актам, а не по фрагментам",
            "test_zhivye": "акты, про которые спрашивают в «Разъяснениях» "
                           "и на которые указывает разметка живых вопросов",
            "test_zhanr": f"все акты вида «{HELD_KIND}» ({KIND_NAMES[HELD_KIND]}) "
                          "— отложенный класс документов",
            "test_nevidannye": f"доля {args.unseen_share} фрагментов, "
                               "стратифицированно по виду документа",
            "dev": f"доля {args.dev_share} фрагментов от оставшихся, "
                   "отложены целиком, а не фрагментами обучающих актов",
            "train": "всё остальное",
        },
        "статистика": stats,
        "группы": groups,
    }
    # ключи латиницей для кода, кириллицей — для чтения человеком
    split["gruppy"] = split.pop("группы")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=1)

    say()
    say(f"{'группа':<18} {'актов':>7} {'фрагментов':>12} {'доля':>7}")
    for name in S.GROUPS:
        s = stats[name]
        say(f"{name:<18} {s['актов']:>7} {s['фрагментов']:>12} "
            f"{100 * s['доля фрагментов']:>6.1f}%")
    say()
    say("состав отложенного по видам документа:")
    for name in S.HELD_OUT:
        kinds = collections.Counter(acts[i]["type"] for i in groups[name])
        say(f"   {name:<18} " + ", ".join(f"{KIND_NAMES.get(k, k)} {v}"
                                          for k, v in kinds.most_common()))
    say()
    biggest = sorted(groups[S.TEST_LIVE], key=lambda i: -acts[i]["chunks"])[:6]
    say("акты живых вопросов (крупнейшие): " +
        ", ".join(f"{acts[i]['number']} ({acts[i]['chunks']} фрагм.)" for i in biggest))
    say(f"обучению остаётся {stats[S.TRAIN]['фрагментов']} фрагментов "
        f"в {stats[S.TRAIN]['актов']} актах")
    say(f"\nфайл: {os.path.relpath(args.out, ROOT)}")
    say("пересобирать после каждой новой разметки и до генерации обучающей выборки")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
