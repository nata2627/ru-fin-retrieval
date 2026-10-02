#!/usr/bin/env python3
"""Этап D на видеокарте: дообучение ретривера по рецепту.

Прогоны разделены на шаги, и каждый шаг проверяет, нет ли готового
результата, и пропускает себя. После обрыва сессии достаточно запустить
ячейку заново.

    --step baseline   нулевая точка: dev необученного ученика
    --step train      обучение и оценка по dev, этап за этапом
    --step final      один прогон по всему набору запросов
    --step forget     катастрофическое забывание на посторонних данных

Подготовка (D0) живёт отдельным скриптом `gpu_prepare.py`: она считается
один раз, стоит дороже всего остального вместе и своим файлом пользуются
сразу два этапа рецепта.

**Тест не трогается до шага `final`.** Весь подбор идёт по dev, 300
вопросов. Если смотреть на тест между этапами, выбор рецепта начнёт
подгоняться под него, и итоговая цифра перестанет что-либо значить.
Поэтому `--step train` физически не умеет считать по тесту: он фильтрует
запросы по составу dev и отказывается работать, если dev не нашёлся.

Порядок наследования. Каждый этап обучается **с исходных весов**,
а рецепт наращивается составом: B это A плюс трудные негативы, C это B
плюс дистилляция. Почему так, а не дообучением по цепочке, написано
в `rufin.training.config`.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gpu_common as common  # noqa: E402
import gpu_traineval as E  # noqa: E402

from rufin import split as SP  # noqa: E402
from rufin.benchmark import read_qrels  # noqa: E402
from rufin.training import trainer as T  # noqa: E402
from rufin.training.config import BY_TAG, RECIPE, STUDENT, TrainConfig, stage_e  # noqa: E402
from rufin.training.journal import Journal, entry_from_per_query  # noqa: E402
from rufin.training.negatives import NegativeRules, detect_scale, pick_all  # noqa: E402
from rufin.training.pairs import (  # noqa: E402
    build_pairs,
    check_group,
    leaked_acts,
    read_jsonl,
    train_acts,
)

БАЗОВАЯ = "базовая"


def освободить(*что) -> None:
    import torch
    for x in что:
        del x
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def найти(имя: str, путь: str | None = None) -> str:
    нашлось = путь or common.find_file(имя)
    if нашлось is None:
        raise SystemExit(f"не найден {имя}: подключите датасет с набором запросов")
    return нашлось


def сверить_нарезку(chunks: list[dict], split: dict, cache: str) -> None:
    """Нарезка обязана совпасть с той, по которой считались выдачи и разметка.

    Идентификатор фрагмента позиционный, и при сдвиге границ он не исчезает,
    а начинает указывать на другой текст: эталоны съедут молча, а обучение
    будет учить модель находить не то.
    """
    по_сплиту = sum(stat["фрагментов"] for stat in split["статистика"].values())
    if по_сплиту != len(chunks):
        raise SystemExit(
            f"нарезка разошлась со сплитом: здесь {len(chunks)} фрагментов, "
            f"а сплит посчитан по {по_сплиту}. Положите каноническую нарезку "
            f"в {cache}/base.jsonl и снесите то, что там лежит сейчас: "
            f"готовая берётся раньше, чем считается новая.")
    print(f"нарезка сошлась со сплитом: {len(chunks)} фрагментов", flush=True)


class Окружение:
    """Всё, что нужно любому шагу: корпус, сплит, запросы, разметка.

    Собирается один раз на прогон: нарезка стоит минуты, а шагов в прогоне
    обычно несколько.
    """

    def __init__(self, args) -> None:
        self.device = common.pick_device()
        print(f"устройство: {self.device}", flush=True)
        if self.device != "cuda":
            print("ВНИМАНИЕ: видеокарта не подключена. "
                  "Settings -> Accelerator -> GPU T4", flush=True)
        ruler = common.make_ruler()
        acts = common.load_acts(args.acts)
        self.chunks = common.build_chunks_cached(acts, "base", ruler, args.chunks_cache)
        self.split = SP.load(найти("split.json", args.split))
        сверить_нарезку(self.chunks, self.split, args.chunks_cache)
        self.texts = {c["chunk_id"]: c["text"] for c in self.chunks}

        self.queries = read_jsonl(найти("queries.jsonl", args.queries))
        self.qrels = read_qrels(найти("qrels.tsv", args.qrels))
        subsets = None
        путь = args.subsets or common.find_file("podvyborki.json")
        if путь:
            with open(путь, encoding="utf-8") as f:
                subsets = json.load(f)
        self.dev = E.dev_ids(self.queries, self.qrels, subsets)
        if not self.dev:
            raise SystemExit(
                "dev не нашёлся: без него подбор рецепта пошёл бы по тесту, "
                "и итоговая цифра перестала бы что-либо значить. Нужен "
                "podvyborki.json или идентификаторы dev с приставкой «dv».")
        print(f"запросов всего {len(self.queries)}, из них dev {len(self.dev)} "
              f"(подбор рецепта идёт только по ним)", flush=True)
        self.dev_queries = [q for q in self.queries if q["query_id"] in self.dev]
        self.dev_qrels = {q: v for q, v in self.qrels.items() if q in self.dev}

    # ---- обучающая часть: нужна только шагу train ----

    def обучающее(self, args) -> tuple[list, dict, dict, str]:
        train = read_jsonl(найти("synthetic_train.jsonl", args.train))
        утечка = leaked_acts(train, self.dev_queries)
        if утечка:
            raise SystemExit(
                f"обучение и dev делят {len(утечка)} актов, например "
                f"{sorted(утечка)[:3]}. Фрагменты одного акта нарезаны "
                f"с перекрытием, то есть dev мерил бы узнавание знакомого "
                f"текста, а не поиск.")
        чужие = check_group(train, self.split, SP.TRAIN)
        if чужие:
            raise SystemExit(f"{len(чужие)} обучающих вопросов не в группе train: "
                             f"{list(чужие.items())[:3]}")
        pairs, stats = build_pairs(train, self.texts)
        print(f"пары: {stats}", flush=True)
        if args.limit_train:
            pairs = pairs[:args.limit_train]
            print(f"ПРОБА: обучающая выборка урезана до {len(pairs)} пар. "
                  f"Результат в таблицу рецепта не годится", flush=True)
        if stats["эталона нет в нарезке"]:
            raise SystemExit("часть эталонов обучающей выборки отсутствует "
                             "в нарезке: нарезка разошлась с генерацией")

        prepared_path = найти("train_prepared.jsonl", args.prepared)
        подготовка = read_jsonl(prepared_path)
        по_id = {r["query_id"]: r for r in подготовка}
        scale = detect_scale([r["gold_score"] for r in подготовка])
        правила = NegativeRules(per_query=args.per_query,
                                train_acts=train_acts(self.split))
        негативы, neg_stats = pick_all(подготовка, правила, scale)
        print(f"подготовка: {prepared_path}, шкала {scale}", flush=True)
        for k, v in neg_stats.items():
            print(f"   {k:<28} {v}", flush=True)
        return pairs, негативы, по_id, scale


def оценить_dev(путь: str, env: Окружение, dims: tuple[int, ...],
                batch_size: int) -> dict:
    return E.evaluate(путь, env.chunks, env.dev_queries, env.dev_qrels,
                      env.device, dims=dims, batch_size=batch_size)


def шаг_baseline(env: Окружение, args, journal: Journal) -> None:
    """Нулевая точка рецепта: необученный ученик, тем же протоколом.

    Брать цифру из отчёта этапа 2 нельзя, хотя она там есть: сравнение
    парное, и для него нужны поквериные значения, посчитанные тем же кодом
    на том же корпусе. Полчаса видеокарты дешевле, чем сомнение в главной
    цифре отчёта.
    """
    if journal.by_tag(БАЗОВАЯ):
        print("нулевая точка уже в журнале, пропуск", flush=True)
        return
    print(f"\n=== нулевая точка: {STUDENT} ===", flush=True)
    итог = оценить_dev(STUDENT, env, (), args.batch_size)
    dim = next(d for d in итог if isinstance(d, int))
    journal.add(entry_from_per_query(
        БАЗОВАЯ, "базовая", итог[dim]["per_query"], итог[dim]["qids"],
        note="необученный multilingual-e5-small, тот же протокол",
        weights=STUDENT))
    journal.save(args.journal)
    print(f"журнал: {args.journal}", flush=True)


def шаг_train(env: Окружение, args, journal: Journal) -> None:
    if not journal.by_tag(БАЗОВАЯ):
        raise SystemExit("сначала --step baseline: без нулевой точки первый "
                         "этап не с чем сравнивать")
    pairs, негативы, подготовка, scale = env.обучающее(args)

    for cfg in выбрать_этапы(args, journal):
        if journal.by_tag(cfg.tag) and not args.force:
            print(f"\n=== {cfg.tag}: уже в журнале, пропуск ===", flush=True)
            continue
        print(f"\n=== этап {cfg.stage}, метка {cfg.tag} ===", flush=True)
        print(f"    {cfg.note}", flush=True)
        out_dir = os.path.join(args.weights, cfg.tag)
        наборы, data_stats = T.build_datasets(cfg, pairs, негативы, подготовка,
                                             env.texts, scale)
        for имя, s in data_stats.items():
            print(f"    набор {имя}: {s}", flush=True)
        t0 = time.time()
        отчёт = T.train(cfg, наборы, out_dir)
        освободить()

        итог = оценить_dev(out_dir, env, cfg.matryoshka, args.batch_size)
        полная = max(d for d in итог if isinstance(d, int))
        запись = entry_from_per_query(
            cfg.tag, cfg.stage, итог[полная]["per_query"], итог[полная]["qids"],
            note=cfg.note, extends=cfg.extends, weights=out_dir,
            seconds=round(time.time() - t0, 1), params=отчёт["параметры"],
            config=cfg.as_dict())
        if cfg.matryoshka:
            запись.config["матрёшка на dev"] = {
                str(d): итог[d]["NDCG@10"] for d in sorted(итог) if isinstance(d, int)}
        journal.add(запись)
        journal.save(args.journal)
        если_есть = (f", разница {запись.diff['mean']:+.3f} "
                     f"с «{запись.compared_with}»") if запись.diff else ""
        print(f"\n{(запись.verdict or 'без вердикта').upper()}: "
              f"dev NDCG@10 {запись.ndcg:.3f}{если_есть}", flush=True)
        освободить()

    занято = sum(os.path.getsize(os.path.join(база, имя))
                 for база, _, файлы in os.walk(args.weights) for имя in файлы)
    print(f"\nвеса занимают {занято / 1073741824:.1f} ГБ в {args.weights}: "
          f"около 0,47 ГБ на этап. Если места мало, папки отклонённых этапов "
          f"можно снести — в журнале остались и цифры, и конфигурация",
          flush=True)
    print("\n==== таблица рецепта ====", flush=True)
    print(journal.table(), flush=True)
    лучшее = journal.best()
    print(f"\nлучшее принятое: {лучшее.tag}, dev NDCG@10 {лучшее.ndcg:.3f}", flush=True)


def выбрать_этапы(args, journal: Journal) -> list[TrainConfig]:
    """Какие этапы считать в этом прогоне.

    Сетка этапа E строится от состава лучшего принятого этапа, а не от
    записанного в рецепте по умолчанию: «поверх лучшего» и означает это.
    """
    if args.tags:
        известные = []
        for метка in args.tags:
            if метка in BY_TAG:
                известные.append(BY_TAG[метка])
            else:
                raise SystemExit(f"нет такого этапа: {метка}. Есть: "
                                 f"{', '.join(BY_TAG)}")
        return известные
    этапы = [c for c in RECIPE if c.stage != "E"]
    лучшее = journal.best()
    основа = BY_TAG.get(лучшее.tag) if лучшее else None
    return этапы + list(stage_e(основа) if основа else ())


def шаг_final(env: Окружение, args, journal: Journal) -> None:
    """D6: один прогон по всему набору запросов. Единственное касание теста.

    Метрики здесь не печатаются по подвыборкам — это делает локальный
    `make metrics` по тем же файлам выдач, тем же кодом, которым посчитаны
    все остальные конфигурации. Разбивка на заголовочную и живые вопросы
    важнее итоговой цифры, и считать её вторым кодом нельзя.
    """
    метки = args.tags or ([journal.best().tag] if journal.best() else [])
    метки = [m for m in метки if m != БАЗОВАЯ]
    if not метки:
        raise SystemExit("нечего прогонять: ни один этап не принят. "
                         "Укажите --tags вручную, если это сознательно")
    os.makedirs(args.runs, exist_ok=True)
    итоги: dict = {}
    for метка in метки:
        cfg = BY_TAG.get(метка)
        out_dir = os.path.join(args.weights, метка)
        if not os.path.exists(out_dir):
            raise SystemExit(f"нет весов этапа {метка}: {out_dir}")
        dims = cfg.matryoshka if cfg else ()
        print(f"\n=== финал: {метка}, размерности {dims or 'полная'} ===", flush=True)
        итог = E.evaluate(out_dir, env.chunks, env.queries, env.qrels,
                          env.device, dims=dims, batch_size=args.batch_size)
        полная = max(d for d in итог if isinstance(d, int))
        for dim in sorted((d for d in итог if isinstance(d, int)), reverse=True):
            имя = "dense-rufin" if dim == полная else f"dense-rufin-{dim}"
            путь = E.save_runs(args.runs, итог[dim]["runs"], имя)
            print(f"   {имя}: {путь}", flush=True)
            итоги[имя] = {"dev+тест вместе, NDCG@10": итог[dim]["NDCG@10"],
                          "размер матрицы МБ": итог[dim]["размер матрицы МБ"],
                          "метка": метка, "размерность": dim}
        # матрица полной размерности уезжает на мак: по ней снимается задержка
        # и размер индекса, а обрезанные размерности получаются из неё срезом
        сохранить_матрицу(out_dir, env, args)
    with open(os.path.join(args.runs, "report_phase_d.json"), "w", encoding="utf-8") as f:
        json.dump({"финал": итоги, "правило": "разбивку по подвыборкам считает "
                                              "локальный make metrics"},
                  f, ensure_ascii=False, indent=1)
    print("\nскачать: runs/*.jsonl, report_phase_d.json, journal.json, "
          "embeddings/base/rufin/", flush=True)


def сохранить_матрицу(out_dir: str, env: Окружение, args) -> None:
    """Эмбеддинги корпуса обученной моделью, в раскладке проекта.

    Нужны на маке: `make latency` строит по ним индекс и мерит задержку
    там, где система и работает. Размерности матрёшки получаются из этой
    матрицы срезом, второго прогона они не требуют.
    """
    import numpy as np
    папка = os.path.join(args.embeddings, "base", "rufin")
    if os.path.exists(os.path.join(папка, "vectors.npy")):
        print("   матрица уже сохранена, пропуск", flush=True)
        return
    os.makedirs(папка, exist_ok=True)
    model = E.load_model(out_dir, env.device)
    from rufin.retrieval.model_specs import MODELS
    spec = MODELS["e5-small"]
    vec = E.encode(model, [c["text"] for c in env.chunks], spec.passage_prefix,
                   args.batch_size, progress=True)
    np.save(os.path.join(папка, "vectors.npy"), vec.astype("float16"))
    with open(os.path.join(папка, "ids.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(c["chunk_id"] for c in env.chunks) + "\n")
    with open(os.path.join(папка, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"model": "rufin", "weights": out_dir, "chunks": len(env.chunks),
                   "dim": int(vec.shape[1]), "dtype": "float16"}, f,
                  ensure_ascii=False, indent=1)
    освободить(model)
    print(f"   матрица сохранена: {папка}, "
          f"{os.path.getsize(os.path.join(папка, 'vectors.npy')) / 1048576:.0f} МБ",
          flush=True)


def шаг_forget(env: Окружение, args, journal: Journal) -> None:
    import gpu_forget as F
    метки = args.tags or ([journal.best().tag] if journal.best() else [])
    пути = {БАЗОВАЯ: STUDENT}
    for метка in метки:
        пути[метка] = os.path.join(args.weights, метка)
    итог = F.measure(пути, env.device, task=args.forget_task,
                     queries=args.forget_queries, batch_size=args.batch_size)
    путь = os.path.join(args.runs, "zabyvanie.json")
    os.makedirs(args.runs, exist_ok=True)
    with open(путь, "w", encoding="utf-8") as f:
        json.dump(итог, f, ensure_ascii=False, indent=1)
    print(f"\nзабывание: {путь}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True,
                    choices=("baseline", "train", "final", "forget"))
    ap.add_argument("--tags", nargs="*", default=None,
                    help="какие этапы считать; по умолчанию весь рецепт")
    ap.add_argument("--acts", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--queries", default=None)
    ap.add_argument("--qrels", default=None)
    ap.add_argument("--subsets", default=None, help="podvyborki.json")
    ap.add_argument("--train", default=None, help="synthetic_train.jsonl")
    ap.add_argument("--prepared", default=None, help="train_prepared.jsonl из D0")
    ap.add_argument("--chunks-cache", default="/kaggle/working/chunks")
    ap.add_argument("--weights", default="/kaggle/working/weights")
    ap.add_argument("--runs", default="/kaggle/working/runs")
    ap.add_argument("--embeddings", default="/kaggle/working/embeddings")
    ap.add_argument("--journal", default="/kaggle/working/train/journal.json")
    ap.add_argument("--batch-size", type=int, default=128,
                    help="размер батча при подсчёте эмбеддингов, не при обучении")
    ap.add_argument("--per-query", type=int, default=2)
    ap.add_argument("--limit-train", type=int, default=0,
                    help="урезать обучающую выборку. Только для пробы: журнал "
                         "и папку весов при этом задавайте отдельные, иначе "
                         "проба попадёт в таблицу рецепта как результат")
    ap.add_argument("--forget-task", default="RuBQRetrieval")
    ap.add_argument("--forget-queries", type=int, default=300)
    ap.add_argument("--force", action="store_true",
                    help="переделать этап, уже записанный в журнал")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.journal), exist_ok=True)
    journal = Journal.load(args.journal)
    env = Окружение(args)
    {"baseline": шаг_baseline, "train": шаг_train,
     "final": шаг_final, "forget": шаг_forget}[args.step](env, args, journal)


if __name__ == "__main__":
    main()
