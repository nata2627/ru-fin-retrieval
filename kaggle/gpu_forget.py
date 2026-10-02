#!/usr/bin/env python3
"""Катастрофическое забывание: что дообучение на домене испортило снаружи.

Дообучение на узком домене улучшает домен и портит всё остальное. Это
не предположение, а свойство метода, и единственный способ про него
что-то сказать — померить числом до и после на наборе, которого в обучении
не было вовсе.

Набор нужен **посторонний**, а не отложенный: отложенные акты это тот же
Банк России, тот же канцелярит и та же нарезка. Поэтому берётся задача
русскоязычного поиска из MTEB — другая предметная область, другой язык
вопросов, другая длина документов.

Протокол тот же, что у всего остального: префиксы e5, косинус, NDCG@10
из `rufin.metrics`. Корпус задачи при необходимости урезается, и урезание
печатается числом: на 56 тысячах документов замер стоил бы столько же,
сколько оценка на своём корпусе, а нужен он здесь не как главная цифра,
а как ответ «сильно испортилось или нет». Эталонные документы в урезанный
корпус входят всегда — иначе метрика померила бы отсутствие ответа.

Если забывание сильное, дистилляция (этап C) его обычно уменьшает,
и это ещё один довод за неё. Поэтому мерить надо не только лучший этап,
но и этап A: разница между ними и есть ответ на вопрос, помогает ли
дистилляция удержать общее качество.
"""
from __future__ import annotations

import random
import sys
import time

from rufin import metrics as M
from rufin.retrieval.model_specs import MODELS
from rufin.training.config import SPEC

DEFAULT_TASK = "RuBQRetrieval"
CORPUS_CAP = 20000


def load_task(name: str, split: str = "test") -> tuple[dict, dict, dict]:
    """Загрузить задачу MTEB: корпус, запросы, разметка.

    Пакет `mteb` менял устройство объекта задачи между версиями, поэтому
    поля берутся не по одному известному имени, а по тому, что нашлось,
    и при неудаче печатается, что именно не нашлось. Падать здесь нельзя:
    забывание — отдельный замер, и он не должен уносить с собой прогон
    рецепта.
    """
    import mteb
    задачи = mteb.get_tasks(tasks=[name])
    if not задачи:
        raise RuntimeError(f"MTEB не знает задачи {name}")
    задача = задачи[0]
    задача.load_data()
    def достать(имя: str):
        значение = getattr(задача, имя, None)
        if значение is None:
            raise RuntimeError(
                f"у задачи {name} нет поля {имя}. Версия пакета mteb "
                f"изменила устройство объекта задачи: "
                f"{sorted(v for v in vars(задача) if not v.startswith('_'))[:12]}")
        return значение[split] if split in значение else next(iter(значение.values()))
    return достать("corpus"), достать("queries"), достать("relevant_docs")


def doc_text(doc) -> str:
    if isinstance(doc, str):
        return doc
    части = [doc.get("title", ""), doc.get("text", "")]
    return "\n".join(ч for ч in части if ч).strip()


def prepare(corpus: dict, queries: dict, qrels: dict, сколько: int,
            cap: int = CORPUS_CAP, seed: int = 13) -> tuple[list, list, dict]:
    """Урезать задачу до размера, который имеет смысл считать.

    Эталонные документы входят в урезанный корпус всегда: иначе метрика
    померила бы, что ответа в корпусе нет, и упала бы у всех моделей
    одинаково, то есть не сказала бы о забывании ничего.
    """
    rng = random.Random(seed)
    отобранные = sorted(q for q in queries if qrels.get(q))
    rng.shuffle(отобранные)
    отобранные = отобранные[:сколько] if сколько else отобранные
    нужные = {d for q in отобранные for d, g in qrels[q].items() if g > 0}
    прочие = [d for d in corpus if d not in нужные]
    rng.shuffle(прочие)
    берём = list(нужные) + прочие[:max(0, cap - len(нужные))]
    док = [{"id": d, "text": doc_text(corpus[d])} for d in берём]
    зап = [{"query_id": q, "text": queries[q] if isinstance(queries[q], str)
            else doc_text(queries[q])} for q in отобранные]
    разметка = {q: {d: int(g) for d, g in qrels[q].items()} for q in отобранные}
    return док, зап, разметка


def measure(модели: dict[str, str], device: str, task: str = DEFAULT_TASK,
            queries: int = 300, batch_size: int = 128) -> dict:
    """NDCG@10 на посторонней задаче для каждой модели и падение к базовой."""
    import gpu_traineval as E

    try:
        corpus, texts, qrels = load_task(task)
    except Exception as e:  # noqa: BLE001  любая беда здесь — не повод ронять прогон
        print(f"ЗАМЕР ЗАБЫВАНИЯ НЕ СОСТОЯЛСЯ: {type(e).__name__}: {e}",
              file=sys.stderr, flush=True)
        print("Поставьте пакет: pip install mteb. Остальные шаги от этого "
              "не зависят", file=sys.stderr, flush=True)
        return {"задача": task, "ошибка": f"{type(e).__name__}: {e}"}

    док, зап, разметка = prepare(corpus, texts, qrels, queries)
    print(f"задача {task}: документов {len(док)} из {len(corpus)}, "
          f"запросов {len(зап)}", flush=True)
    spec = MODELS[SPEC]

    итог: dict = {"задача": task, "документов": len(док), "запросов": len(зап),
                  "модели": {}}
    поквериные: dict[str, dict] = {}
    for имя, путь in модели.items():
        t0 = time.time()
        model = E.load_model(путь, device, spec.max_seq_length)
        ids = [d["id"] for d in док]
        corpus_vec = E.encode(model, [d["text"] for d in док], spec.passage_prefix,
                              batch_size, progress=True)
        qvec = E.encode(model, [q["text"] for q in зап], spec.query_prefix, batch_size)
        выдача = E.search(corpus_vec, ids, qvec, device)
        runs = {зап[i]["query_id"]: [c for c, _ in выдача[i]][:10] for i in выдача}
        pq = M.per_query(runs, разметка)
        поквериные[имя] = pq
        ci = M.bootstrap_ci(pq["NDCG@10"])
        итог["модели"][имя] = {"NDCG@10": ci.__dict__, "секунд": round(time.time() - t0, 1),
                               "веса": путь}
        print(f"   {имя:<16} NDCG@10 {ci}", flush=True)
        del model, corpus_vec, qvec
        import torch
        if device == "cuda":
            torch.cuda.empty_cache()

    база = next(iter(модели))
    for имя in модели:
        if имя == база:
            continue
        разница, p = M.paired_diff_ci(поквериные[имя]["NDCG@10"],
                                      поквериные[база]["NDCG@10"])
        итог["модели"][имя]["падение против базовой"] = {
            "mean": разница.mean, "lo": разница.lo, "hi": разница.hi, "p": p}
        знак = "просело" if разница.mean < 0 else "выросло"
        print(f"   {имя:<16} против «{база}»: {разница.mean:+.3f} "
              f"[{разница.lo:+.3f}; {разница.hi:+.3f}] p={p:.3f} — {знак}", flush=True)
    return итог
