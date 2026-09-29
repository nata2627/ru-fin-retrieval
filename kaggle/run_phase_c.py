#!/usr/bin/env python3
"""Этап C на видеокарте: обучающая выборка, dev, синтетический тест, судья.

Порядок шагов не переставляется, и каждый следующий опирается на числа
предыдущего.

**0. Замер скорости** на сотне фрагментов. Объём обучающей выборки
назначается по замеру, а не по плану: десять тысяч вопросов за три часа
лучше пятнадцати, которые не досчитаются к концу сессии.

**1. Обучающая выборка** — по фрагментам обучающих актов сплита, одной
моделью.

**2. dev и синтетический тест** — **другой** моделью, из другого семейства.
Иначе тест померил бы, насколько ученик выучил стиль своего же генератора.
Обе модели уходят в `DATASET_CARD.md`.

**3. Третий фильтр** — учитель-кросс-энкодер обязан видеть исходный фрагмент
в первых пятидесяти. Считается отдельным шагом, потому что это самая дорогая
часть: `вопросов × глубина` проходов кросс-энкодера.

**4. Судья** размечает пул кандидатов, собранный локально по выдачам всех
конфигураций.

Шаги независимы: каждый проверяет, нет ли готового файла, и пропускает себя.
После обрыва сессии достаточно запустить ячейку заново.

Что обязательно проверить **до** запуска: токен HuggingFace подключён
к этой тетради. Секреты Kaggle привязываются к конкретной тетради через
браузер и новым ядром не наследуются — на этом проект уже потерял два часа
квоты, когда скачивание весов порезали по скорости.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gpu_common as common  # noqa: E402
import gpu_gen as G  # noqa: E402

from rufin import split as S  # noqa: E402


def free(llm) -> None:
    """Выгрузить модель: следующий шаг поднимает свою, и двум места нет."""
    import torch
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def chunks_of_groups(chunks: list[dict], split: dict, groups: list[str]) -> list[dict]:
    by_act = S.group_by_act(split)
    wanted = set(groups)
    out = [c for c in chunks if by_act.get(c["act_id"]) in wanted]
    print(f"фрагментов в группах {', '.join(groups)}: {len(out)} из {len(chunks)}",
          flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default=None)
    ap.add_argument("--split", default=None, help="split.json; ищется сам")
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--chunks-cache", default="/kaggle/working/chunks")
    ap.add_argument("--train-model", default=G.TRAIN_MODEL)
    ap.add_argument("--test-model", default=G.TEST_MODEL,
                    help="другое семейство: им генерируются dev и тест")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--dtype", default="float16",
                    help="на T4 только float16: bf16 на Turing нет")
    ap.add_argument("--quantization", default="awq",
                    help="пусто — без квантования")
    ap.add_argument("--tensor-parallel", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--probe", type=int, default=100,
                    help="0 — не мерить скорость")
    ap.add_argument("--train", type=int, default=0, help="сколько обучающих вопросов")
    ap.add_argument("--dev", type=int, default=0)
    ap.add_argument("--test", type=int, default=0)
    ap.add_argument("--teacher-filter", action="store_true",
                    help="третий фильтр: учитель видит эталон в первых пятидесяти")
    ap.add_argument("--filter-depth", type=int, default=100)
    ap.add_argument("--judge", action="store_true", help="разметить пул судьёй")
    ap.add_argument("--limit-acts", type=int, default=0)
    args = ap.parse_args()

    device = common.pick_device()
    print(f"устройство: {device}", flush=True)
    if device != "cuda":
        print("ВНИМАНИЕ: видеокарта не подключена. Accelerator -> GPU T4 x2", flush=True)

    quant = args.quantization or None
    qdir = os.path.join(args.out, "queries")
    os.makedirs(qdir, exist_ok=True)
    report_path = os.path.join(args.out, "report_phase_c.json")
    report: dict = json.load(open(report_path, encoding="utf-8")) \
        if os.path.exists(report_path) else {}

    def save() -> None:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)

    ruler = common.make_ruler()
    acts = common.load_acts(args.acts, limit=args.limit_acts)
    chunks = common.build_chunks_cached(acts, "base", ruler, args.chunks_cache)
    report["всего фрагментов"] = len(chunks)

    split_path = args.split or common.find_file("split.json")
    if split_path is None:
        raise SystemExit("не найден split.json: подключите датасет с набором запросов. "
                         "Генерировать без сплита нельзя — вопросы попадут "
                         "на акты теста, и утечку потом не отследить.")
    split = S.load(split_path)
    print(f"сплит: {split_path}", flush=True)
    for name, stat in split["статистика"].items():
        print(f"   {name:<18} актов {stat['актов']:>5}, фрагментов {stat['фрагментов']:>7}",
              flush=True)

    all_texts = [c["text"] for c in chunks]

    # ---- 0. замер скорости ----
    if args.probe and "скорость" not in report:
        llm = G.load_llm(args.train_model, args.dtype, quant, args.tensor_parallel,
                         args.max_model_len)
        report["скорость"] = G.probe_speed(llm, chunks_of_groups(chunks, split, [S.TRAIN]),
                                          sample=args.probe)
        report["скорость"]["модель"] = args.train_model
        save()
        if not (args.train or args.dev or args.test):
            print("\nзамер сделан. Объём назначается по нему: перезапустите "
                  "с --train N", flush=True)
            return
    else:
        llm = None

    # ---- 1. обучающая выборка ----
    train_path = os.path.join(qdir, "synthetic_train.jsonl")
    if args.train and not os.path.exists(train_path):
        if llm is None:
            llm = G.load_llm(args.train_model, args.dtype, quant, args.tensor_parallel,
                             args.max_model_len)
        report["train"] = G.make_queries(
            llm, chunks_of_groups(chunks, split, [S.TRAIN]), train_path,
            target=args.train, prefix="tr", idf_texts=all_texts)
        report["train"]["модель"] = args.train_model
        save()
    if llm is not None:
        free(llm)
        llm = None

    # ---- 2. dev и синтетический тест: другой моделью ----
    dev_path = os.path.join(qdir, "synthetic_dev.jsonl")
    test_path = os.path.join(qdir, "synthetic_test.jsonl")
    need_other = ((args.dev and not os.path.exists(dev_path))
                  or (args.test and not os.path.exists(test_path)))
    if need_other:
        llm = G.load_llm(args.test_model, args.dtype, quant, args.tensor_parallel,
                         args.max_model_len)
        if args.dev and not os.path.exists(dev_path):
            report["dev"] = G.make_queries(
                llm, chunks_of_groups(chunks, split, [S.DEV]), dev_path,
                target=args.dev, prefix="dv", idf_texts=all_texts)
            report["dev"]["модель"] = args.test_model
            save()
        if args.test and not os.path.exists(test_path):
            report["test"] = G.make_queries(
                llm, chunks_of_groups(chunks, split, [S.TEST_UNSEEN, S.TEST_KIND]),
                test_path, target=args.test, prefix="ts", idf_texts=all_texts)
            report["test"]["модель"] = args.test_model
            save()
        free(llm)
        llm = None

    # ---- 3. третий фильтр ----
    if args.teacher_filter:
        for имя, path in (("train", train_path), ("test", test_path)):
            if not os.path.exists(path) or f"фильтр учителя {имя}" in report:
                continue
            with open(path, encoding="utf-8") as f:
                queries = [json.loads(l) for l in f if l.strip()]
            kept, stats = G.filter_by_teacher(queries, chunks, depth=args.filter_depth,
                                              device=device)
            with open(path, "w", encoding="utf-8") as f:
                for q in kept:
                    f.write(json.dumps(q, ensure_ascii=False) + "\n")
            report[f"фильтр учителя {имя}"] = stats
            save()

    # ---- 4. судья ----
    if args.judge:
        import gpu_judge as J
        pool_path = common.find_file("pool_candidates.tsv")
        if pool_path is None:
            raise SystemExit("не найден pool_candidates.tsv: соберите его локально "
                             "(`make pool`) после прогона всех конфигураций")
        pairs = J.read_pool(pool_path)
        print(f"пул: {pool_path}, пар {len(pairs)}", flush=True)
        texts = {c["chunk_id"]: c["text"] for c in chunks}
        llm = G.load_llm(args.judge_model or J.DEFAULT_JUDGE, args.dtype, quant,
                         args.tensor_parallel, args.max_model_len)
        report["судья"] = J.judge(llm, pairs, texts, os.path.join(qdir, "judge.tsv"))
        report["судья"]["модель"] = args.judge_model or J.DEFAULT_JUDGE
        free(llm)
        save()

    print("\n==== итог этапа C ====", flush=True)
    for key, value in report.items():
        if isinstance(value, dict):
            краткое = {k: v for k, v in value.items()
                       if k in ("accepted", "rejected", "модель", "принято",
                                "пар размечено", "вопросов в секунду")}
            print(f"   {key:<22} {краткое or value}", flush=True)
        else:
            print(f"   {key:<22} {value}", flush=True)
    print("\nскачать обязательно: queries/*.jsonl, queries/judge.tsv, "
          "report_phase_c.json — все маленькие", flush=True)


if __name__ == "__main__":
    main()
