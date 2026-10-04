"""Сам прогон обучения. Единственный модуль пакета, которому нужен torch.

Запускается только на видеокарте Kaggle. На маке не разворачивается ни одна
модель: восемь гигабайт общей памяти, и обучение держит сверх весов ещё
градиенты, состояния оптимизатора и батч из 128 текстов по 512 токенов.

Два набора данных и две функции потерь на этапах C и далее. У контрастива
столбцы «вопрос, позитив, негативы», у дистилляции — «вопрос и список
документов плюс оценки учителя»; одной таблицей их не собрать.
`SentenceTransformerTrainer` принимает словарь наборов и словарь функций
потерь с теми же ключами, и сам чередует их по пропорции размеров.

**Негативы просеиваются для контрастива и не просеиваются для дистилляции,
и это не недосмотр.** Контрастив считает негатив абсолютным нулём: ложный
негатив прямо учит отталкивать правильный ответ. Дистилляция же получает
на каждый документ оценку учителя, и неразмеченный правильный ответ
приходит с высокой оценкой — то есть обрабатывается верно сам собой.
Отсюда и порядок: в список дистилляции идут кандидаты как есть, в пары
контрастива — только прошедшие отсечку.
"""
from __future__ import annotations

import inspect
import json
import os
import time
from collections.abc import Mapping, Sequence

from ..retrieval.model_specs import MODELS
from .config import STUDENT, TrainConfig
from .freeze import apply_freeze
from .negatives import logit
from .pairs import TrainPair, as_dataset


def build_distill_columns(pairs: Sequence[TrainPair], prepared: Mapping[str, Mapping],
                          texts: Mapping[str, str], docs: int, scale: str,
                          spec_name: str = "e5-small") -> dict[str, list]:
    """Столбцы набора для дистилляции: вопрос, `docs` документов, оценки.

    Список всегда одной длины: `datasets` требует одинакового числа
    столбцов во всех строках, а softmax по списку переменной длины
    пришлось бы маскировать. Вопросы, у которых кандидатов меньше,
    в набор не идут, и сколько их — считает вызывающая сторона.

    Эталон ставится первым нарочно: порядок внутри списка softmax-у
    безразличен, а при разборе глазами удобно, когда первая оценка —
    оценка правильного ответа.
    """
    spec = MODELS[spec_name]
    cols: dict[str, list] = {"anchor": []}
    for i in range(1, docs + 1):
        cols[f"doc_{i}"] = []
    cols["label"] = []
    for p in pairs:
        rec = prepared.get(p.query_id)
        if rec is None:
            continue
        отобрано = [(p.gold_chunk_id, rec["gold_score"])]
        for chunk_id, score in rec["candidates"]:
            if chunk_id == p.gold_chunk_id or chunk_id not in texts:
                continue
            отобрано.append((chunk_id, score))
            if len(отобрано) == docs:
                break
        if len(отобрано) < docs:
            continue
        cols["anchor"].append(p.anchor)
        for i, (chunk_id, _) in enumerate(отобрано, start=1):
            cols[f"doc_{i}"].append(spec.passage_prefix + texts[chunk_id])
        cols["label"].append([logit(s, scale) for _, s in отобрано])
    return cols


def build_datasets(cfg: TrainConfig, pairs: Sequence[TrainPair],
                   negatives: Mapping[str, list[str]] | None,
                   prepared: Mapping[str, Mapping] | None,
                   texts: Mapping[str, str], scale: str = "logit") -> tuple[dict, dict]:
    """Наборы данных по конфигурации этапа. Второе значение — статистика."""
    from datasets import Dataset

    наборы: dict = {}
    stats: dict = {}
    if cfg.with_negatives and not negatives:
        raise SystemExit(
            f"этапу {cfg.tag} нужны трудные негативы, а подготовки D0 нет. "
            f"Посчитайте gpu_prepare.py и подключите его вывод входом.")
    if "pairs" in cfg.datasets:
        cols = as_dataset(pairs, negatives if cfg.with_negatives else None,
                          texts if cfg.with_negatives else None,
                          per_query=cfg.negatives_per_query)
        наборы["pairs"] = Dataset.from_dict(cols)
        stats["pairs"] = {"строк": len(cols["anchor"]),
                          "столбцов": list(cols),
                          "выброшено за недостатком негативов":
                              len(pairs) - len(cols["anchor"])}
    if "distill" in cfg.datasets:
        if not prepared:
            raise SystemExit(
                f"этапу {cfg.tag} нужна дистилляция, а файла подготовки D0 "
                f"с оценками учителя нет. Посчитайте gpu_prepare.py "
                f"и подключите его вывод входом.")
        cols = build_distill_columns(pairs, prepared, texts, cfg.distill_docs, scale)
        наборы["distill"] = Dataset.from_dict(cols)
        stats["distill"] = {"строк": len(cols["anchor"]),
                            "документов в списке": cfg.distill_docs,
                            "выброшено за недостатком кандидатов":
                                len(pairs) - len(cols["anchor"])}
    if not наборы:
        raise ValueError(f"этап {cfg.tag}: наборов данных нет, обучать нечего")
    return наборы, stats


def load_student(cfg: TrainConfig, weights: str = "", base: str = ""):
    """Поднять ученика.

    `base` — путь к исходным весам. Передавать его обязательно там, где
    хаб недоступен: имя `intfloat/multilingual-e5-small` потянуло бы
    загрузку из сети, а на Kaggle это не падение, а зависание.
    """
    from sentence_transformers import SentenceTransformer
    path = weights or base or STUDENT
    if "/" in path and not os.path.exists(path):
        raise SystemExit(
            f"веса {path} пришлось бы качать из сети. Передайте путь "
            f"к подключённой модели: обучение не должно зависеть от хаба.")
    model = SentenceTransformer(path, local_files_only=True)
    model.max_seq_length = cfg.max_seq_length
    return model


def build_losses(cfg: TrainConfig, model) -> dict:
    """Функции потерь по ключам наборов данных."""
    from sentence_transformers.losses import (
        CachedMultipleNegativesRankingLoss,
        MatryoshkaLoss,
    )

    from .losses import make_listwise_kl

    def обернуть(loss):
        if not cfg.matryoshka:
            return loss
        return MatryoshkaLoss(model, loss, matryoshka_dims=list(cfg.matryoshka))

    out: dict = {}
    if "pairs" in cfg.datasets:
        out["pairs"] = обернуть(CachedMultipleNegativesRankingLoss(
            model, scale=cfg.scale, mini_batch_size=cfg.mini_batch))
    if "distill" in cfg.datasets:
        out["distill"] = обернуть(make_listwise_kl(
            model, scale=cfg.scale, temperature=cfg.kl_temperature))
    return out


def _освободить() -> None:
    import gc

    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def steps_per_epoch(cfg: TrainConfig, наборы: dict) -> int:
    """Сколько шагов оптимизатора в одной эпохе.

    Нужно замеру: он меряет секунды на шаг, а цена этапа это шаги,
    помноженные на эпохи. Строки считаются по всем наборам сразу, потому
    что сэмплер чередует их пропорционально размеру.
    """
    строк = sum(len(н) for н in наборы.values())
    за_шаг = max(cfg.batch * cfg.accumulate, 1)
    return max(1, -(-строк // за_шаг))


def ужать(cfg: TrainConfig) -> TrainConfig | None:
    """Та же конфигурация, но помещающаяся в меньшую память.

    Способ ужатия зависит от функции потерь, и подменять один другим нельзя.

    У контрастива память держит `mini_batch`: GradCache делит батч на куски
    и хранит между проходами только эмбеддинги. Сам батч при этом трогать
    **нельзя** — он задаёт число негативов, и уменьшить его значит упростить
    задачу, а не удешевить её.

    У дистилляции связи между примерами нет вовсе, потери считаются
    по каждому вопросу отдельно. Значит батч можно делить, если во столько
    же раз увеличить накопление: эффективный размер сохраняется, а это
    математически тот же шаг.

    Возвращает `None`, когда ужимать больше нечего.
    """
    from dataclasses import replace
    if "distill" in cfg.datasets:
        if cfg.batch <= 1:
            return None
        return replace(cfg, batch=cfg.batch // 2, accumulate=cfg.accumulate * 2,
                       mini_batch=max(1, cfg.mini_batch // 2))
    if cfg.mini_batch <= 1:
        return None
    return replace(cfg, mini_batch=cfg.mini_batch // 2)


def train(cfg: TrainConfig, наборы: dict, out_dir: str, weights: str = "",
          model=None, base: str = "", замер_шагов: int = 0,
          ужимать: int = 4) -> dict:
    """Обучить один этап и сохранить веса. Возвращает отчёт для журнала.

    Про типы данных: `fp16` и `bf16` выключены явно, и это решение этапа,
    а не умолчание библиотеки. T4 — это Turing, bf16 на нём нет вовсе,
    а fp16 с масштабированием лосса добавляет риск молчаливого расхождения
    ради выигрыша по времени, которого на модели в 118M не нужно.
    """
    import torch
    from sentence_transformers import (
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.training_args import BatchSamplers, MultiDatasetBatchSamplers

    # Нехватка памяти это не повод падать: конфигурацию можно ужать,
    # сохранив смысл этапа. Сколько раз пробовать, задаётся ключом,
    # и каждая попытка называет себя — подобранный размер уходит в отчёт
    # и в рецепт, а не остаётся догадкой.
    if ужимать:
        меньше = cfg
        for попытка in range(ужимать + 1):
            try:
                return train(меньше, наборы, out_dir, weights, model, base,
                             замер_шагов, ужимать=0)
            except Exception as e:  # noqa: BLE001
                if "out of memory" not in str(e).lower() and \
                        type(e).__name__ != "OutOfMemoryError":
                    raise
                дальше = ужать(меньше)
                print(f"[{меньше.tag}] не хватило памяти при батче "
                      f"{меньше.batch} и мини-батче {меньше.mini_batch}; "
                      f"{'ужимаю' if дальше else 'ужимать больше нечего'}",
                      flush=True)
                if дальше is None:
                    raise
                меньше = дальше
                model = None
                _освободить()
        raise RuntimeError(f"[{cfg.tag}] память не нашлась за {ужимать} попыток")

    model = model or load_student(cfg, weights, base)
    plan = apply_freeze(model, freeze_vocabulary=cfg.freeze_vocabulary)
    print(f"[{cfg.tag}] {plan}, батч {cfg.batch} на {cfg.accumulate}, "
          f"мини-батч {cfg.mini_batch}", flush=True)

    losses = build_losses(cfg, model)
    # NO_DUPLICATES нужен контрастиву: один и тот же позитив, попавший в батч
    # дважды, служит сам себе негативом. Набору дистилляции он вреден —
    # там в строке восемь текстов, и дедупликация по тексту выбросила бы
    # половину батча, поэтому при двух наборах остаётся обычный сэмплер.
    sampler = BatchSamplers.NO_DUPLICATES if tuple(cfg.datasets) == ("pairs",) \
        else BatchSamplers.BATCH_SAMPLER
    параметры = dict(
        output_dir=os.path.join(out_dir, "trainer"),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch,
        gradient_accumulation_steps=cfg.accumulate,
        learning_rate=cfg.lr,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        fp16=False, bf16=False,
        seed=cfg.seed,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        batch_sampler=sampler,
        multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
    )
    if замер_шагов:
        # Замер: несколько шагов вместо эпохи. Веса такого прогона никому
        # не нужны, нужна только секунда на шаг.
        параметры["max_steps"] = замер_шагов
    # Версия библиотеки на видеокарте заранее не известна и успела уйти
    # далеко вперёд от той, на которой код писался. Имена параметров
    # обучения между версиями переименовывались, и падение на неизвестном
    # ключе стоило бы прогона целиком. Поэтому неизвестные ключи
    # отбрасываются вслух: пропажа сэмплера меняет обучение, и знать
    # об этом надо из вывода, а не гадать по цифрам.
    известные = set(inspect.signature(SentenceTransformerTrainingArguments.__init__)
                    .parameters)
    # Разогрев переименовали. В пятой версии transformers `warmup_ratio`
    # помечен устаревшим, а доля разогрева задаётся через `warmup_steps`
    # дробным числом. Просто выбросить неизвестный ключ нельзя: обучение
    # пошло бы вовсе без разогрева, не сказав об этом ни слова, а первые
    # шаги с полным шагом обучения портят веса тем сильнее, чем меньше
    # обучаемая часть.
    if "warmup_ratio" not in известные and "warmup_steps" in известные:
        параметры["warmup_steps"] = параметры.pop("warmup_ratio")
        print(f"[{cfg.tag}] разогрев задан через warmup_steps: эта версия "
              f"библиотеки не знает warmup_ratio", flush=True)
    лишние = [k for k in параметры if k not in известные]
    for k in лишние:
        print(f"[{cfg.tag}] параметр {k} эта версия библиотеки не знает, "
              f"обучение пойдёт без него", flush=True)
        параметры.pop(k)
    args = SentenceTransformerTrainingArguments(**параметры)
    # При двух наборах передаётся DatasetDict, а не обычный словарь: ключи
    # набора и ключи функций потерь обязаны совпасть, и DatasetDict это
    # требование выражает, а не подразумевает.
    if len(наборы) > 1:
        from datasets import DatasetDict
        данные, потери = DatasetDict(наборы), losses
    else:
        данные, потери = next(iter(наборы.values())), next(iter(losses.values()))
    trainer = SentenceTransformerTrainer(model=model, args=args,
                                         train_dataset=данные, loss=потери)
    t0 = time.time()
    result = trainer.train()
    seconds = time.time() - t0

    if замер_шагов:
        return {"метка": cfg.tag, "секунд": round(seconds, 1),
                "шагов": int(result.global_step),
                "секунд на шаг": round(seconds / max(result.global_step, 1), 2),
                "батч": cfg.batch, "накопление": cfg.accumulate,
                "мини-батч": cfg.mini_batch,
                "параметры": plan.as_dict(), "замер": True}

    os.makedirs(out_dir, exist_ok=True)
    # save_pretrained — нынешнее имя, save — прежнее. Веса, не сохранённые
    # из-за переименования метода, означают потерянный прогон.
    сохранить = getattr(model, "save_pretrained", None) or model.save
    сохранить(out_dir)
    отчёт = {
        "метка": cfg.tag,
        "секунд": round(seconds, 1),
        "шагов": int(result.global_step),
        "лосс": round(float(result.training_loss), 4),
        "параметры": plan.as_dict(),
        "веса": out_dir,
        "устройство": "cuda" if torch.cuda.is_available() else "cpu",
        "точность": "fp32",
    }
    with open(os.path.join(out_dir, "obuchenie.json"), "w", encoding="utf-8") as f:
        json.dump({"конфигурация": cfg.as_dict(), "отчёт": отчёт}, f,
                  ensure_ascii=False, indent=1)
    print(f"[{cfg.tag}] обучено за {seconds / 60:.1f} мин, шагов "
          f"{отчёт['шагов']}, лосс {отчёт['лосс']}", flush=True)
    return отчёт
