#!/usr/bin/env python3
"""D0: трудные негативы и оценки учителя для обучающей выборки.

Один прогон на все этапы, которым нужен учитель. Этап B берёт отсюда
негативы, этап C — оценки для дистилляции, и пересчитывать одно и то же
дважды незачем: это самая дорогая часть обучения, `вопросов × глубина`
проходов кросс-энкодера.

Что считается и в каком порядке.

**1. Кандидаты.** Гибридная выдача (BM25 плюс плотный поиск необученным
учеником, слияние через RRF) по фрагментам **обучающих актов сплита**
и только по ним. Негатив из акта dev или теста не портит разметку,
но даёт модели увидеть текст отложенного акта на обучении, а весь смысл
сплита по актам в том, что она его не видела.

**2. Эталон вставляется силой, если выдача его не нашла.** Иначе у части
вопросов не будет оценки позитива, а без неё не работает отсечка негативов:
порог считается от неё.

**3. Оценки учителя — сырые логиты.** Берутся прямо у модели через
`AutoModelForSequenceClassification`, а не через обёртку `CrossEncoder`.
Причина не в скорости: обёртка в разных версиях библиотеки применяет
сигмоиду то сама, то нет, и приехавшее число означает то вероятность,
то логит. Порядок кандидатов от этого не меняется (сигмоида монотонна),
а вот порог отсечки 0,5 и softmax дистилляции — меняются полностью.
Поэтому шкала не угадывается, а задаётся: логиты. Для страховки
записанное всё равно проверяется по разбросу, и шкала пишется в файл.

**4. Результат пишется по частям и переживает обрыв.** Сессия на Kaggle
ограничена по времени, логи на ходу не отдаются, и прогон на несколько
часов — это несколько часов вслепую. Поэтому вопросы считаются пачками,
и после каждой готовое дописывается в файл. Убитый на середине прогон
оставляет посчитанное, а следующий начинает с того места, где
остановился предыдущий: уже посчитанные вопросы он просто пропускает.

**5. Учитель сверяется с тем, которым измерено 0,690.** Веса берутся
из модели, подключённой входом Kaggle, а не качаются с HuggingFace:
секреты через API не прицепить, а без токена скачивание режется
по скорости. Но зеркало — это чужая копия, и подменённые или просто
другие веса ничем себя не выдадут: дистилляция пойдёт, лосс упадёт,
а учить будут не тому. Поэтому перед дорогой частью на двадцати запросах
воспроизводится уже посчитанная выдача `base__hybrid-rerank.jsonl`.
Если первые места совпали, это тот самый учитель. Тридцать секунд против
полутора часов неизвестности.

Цена прогона печатается заранее числом: при глубине 30 и 6112 вопросах
это 183 тысячи проходов кросс-энкодера.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gpu_common as common  # noqa: E402
import numpy as np  # noqa: E402

from rufin import split as SP  # noqa: E402
from rufin.retrieval.bm25 import BM25Index  # noqa: E402
from rufin.retrieval.hybrid import rrf  # noqa: E402
from rufin.retrieval.model_specs import MODELS  # noqa: E402
from rufin.training.config import SPEC, STUDENT, TEACHER  # noqa: E402
from rufin.training.negatives import NegativeRules, detect_scale, pick_all  # noqa: E402
from rufin.training.pairs import assert_only_train_acts, read_jsonl, train_acts  # noqa: E402


def probe_speed(model_path: str, chunks: list[dict], device: str,
                sample: int = 512, batch_size: int = 64) -> dict:
    """Замер скорости учителя до того, как потрачены часы.

    Правило проекта, выведенное на этапе C и здесь поначалу не применённое:
    **объём назначается по замеру, а не по арифметике.** Арифметика
    ошиблась вчетверо — она не знает ни про точность, которую библиотека
    молча сменила, ни про накладные раздачи по картам, ни про то, как эта
    версия выполняет внимание на этой карте.

    Замер идёт на настоящих парах, настоящей длины, тем же кодом, которым
    пойдёт весь прогон. Печатается цена всего прогона в часах — до того,
    как он начат, и по ней видно, стоит ли его начинать.
    """
    тексты = [c["text"] for c in chunks[:sample]]
    пары = [("Какие требования установлены к порядку расчёта резерва?", t)
            for t in тексты]
    t0 = time.time()
    score_pairs(пары, model_path, device, batch_size=batch_size)
    секунд = time.time() - t0
    скорость = len(пары) / секунд
    return {"пар": len(пары), "секунд": round(секунд, 1),
            "пар в секунду": round(скорость, 1)}


# Замер идёт на объёме, сравнимом с рабочей пачкой, а не на коротком куске.
# На пятистах парах скорость выходит выше настоящей: загрузка модели
# и прогрев размазываются по слишком малому числу пар, и предел по часам
# пропускает прогон, который в него не укладывается.
ЗАМЕР_ПО_УМОЛЧАНИЮ = 2048


def check_arch(model_path: str, ожидания: dict, кто: str) -> dict:
    """Сверить устройство модели с ожидаемым и напечатать его в отчёт.

    Проверяется не всё подряд, а ровно то, от чего код зависит молча:
    у учителя одна голова классификатора (иначе `logits.view(-1)` вернул бы
    не оценки), у ученика 384 измерения (на них построен весь проект).
    Остальное печатается для журнала.
    """
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_path)
    свойства = {k: getattr(cfg, k, None) for k in
                ("model_type", "vocab_size", "hidden_size", "num_hidden_layers",
                 "num_labels")}
    print(f"{кто}: {свойства}", flush=True)
    for ключ, ждём in ожидания.items():
        есть = свойства.get(ключ)
        if есть != ждём:
            raise SystemExit(
                f"{кто}: {ключ} = {есть}, а должно быть {ждём}. Подключена "
                f"не та модель. Веса берутся из чужого зеркала, и ошибка "
                f"здесь была бы тихой: обучение пошло бы, а учило бы не тому.")
    return свойства


def verify_teacher(model_path: str, chunks: list[dict], device: str,
                   сколько: int = 20, batch_size: int = 64) -> dict:
    """Тот ли это учитель, которым измерено 0,690.

    Берётся уже посчитанная гибридная выдача и её переранжированный
    вариант — оба лежат в проекте с этапа 2, оба получены весами
    с HuggingFace. Подключённая модель переранжирует те же кандидаты,
    и порядок сравнивается. Совпали первые места — тот самый учитель.

    Почему именно первые места, а не весь порядок. Половинная точность
    и другой размер батча двигают близкие оценки, и пара перестановок
    в хвосте ничего не значит. А первое место — это то, что мерит метрика,
    и его совпадение на двадцати запросах уже не случайность.
    """
    гибрид = common.find_file("base__hybrid.jsonl")
    эталон = common.find_file("base__hybrid-rerank.jsonl")
    запросы = common.find_file("queries.jsonl")
    if not (гибрид and эталон and запросы):
        print("СВЕРКА УЧИТЕЛЯ НЕ СДЕЛАНА: среди входов нет готовых выдач "
              "base__hybrid.jsonl и base__hybrid-rerank.jsonl. Подключите "
              "датасет запросов — без сверки источник весов ничем "
              "не подтверждён", flush=True)
        return {"сверка": "нечем"}

    кандидаты = {r["query_id"]: r["ranked"] for r in read_jsonl(гибрид)}
    было = {r["query_id"]: r["ranked"] for r in read_jsonl(эталон)}
    тексты_запросов = {q["query_id"]: q["text"] for q in read_jsonl(запросы)}
    тексты = {c["chunk_id"]: c["text"] for c in chunks}

    общие = [q for q in sorted(кандидаты) if q in было and q in тексты_запросов
             and all(c in тексты for c in кандидаты[q])]
    общие = общие[:сколько]
    if not общие:
        print("СВЕРКА УЧИТЕЛЯ НЕ СДЕЛАНА: не нашлось запросов, у которых есть "
              "и выдача, и переранжирование, и все тексты", flush=True)
        return {"сверка": "нечем"}

    пары, адрес = [], []
    for qid in общие:
        for chunk_id in кандидаты[qid]:
            пары.append((тексты_запросов[qid], тексты[chunk_id]))
            адрес.append((qid, chunk_id))
    print(f"сверка учителя: {len(общие)} запросов, {len(пары)} пар", flush=True)
    оценки = score_pairs(пары, model_path, device, batch_size=batch_size)

    по_вопросу: dict[str, dict[str, float]] = {}
    for (qid, chunk_id), оценка in zip(адрес, оценки):
        по_вопросу.setdefault(qid, {})[chunk_id] = float(оценка)

    совпало_первых = 0
    совпало_пятёрок = 0
    for qid in общие:
        наш = [c for c, _ in sorted(по_вопросу[qid].items(), key=lambda kv: -kv[1])]
        совпало_первых += int(наш[0] == было[qid][0])
        совпало_пятёрок += len(set(наш[:5]) & set(было[qid][:5])) / 5
    доля = совпало_первых / len(общие)
    итог = {"сверка": "сделана", "запросов": len(общие),
            "совпало первых мест": round(доля, 3),
            "пересечение первых пяти": round(совпало_пятёрок / len(общие), 3)}
    print(f"сверка учителя: первые места совпали у {совпало_первых} из "
          f"{len(общие)}, пересечение первых пяти "
          f"{итог['пересечение первых пяти']:.2f}", flush=True)
    if доля < 0.9:
        raise SystemExit(
            f"учитель не воспроизводит выдачу этапа 2: первые места совпали "
            f"только у {доля:.0%} запросов. Это другие веса, и дистилляция "
            f"из них училась бы не тому, чем измерено 0,690. Проверьте, какая "
            f"модель подключена входом, или уберите её и качайте с HuggingFace.")
    return итог


def score_pairs(pairs: list[tuple[str, str]], model_path: str, device: str,
                batch_size: int = 64, max_length: int = 512) -> np.ndarray:
    """Логиты кросс-энкодера по парам «вопрос, фрагмент».

    Пары сортируются по длине внутри прогона и возвращаются в исходном
    порядке: при выравнивании по самой длинной паре батча это заметная
    разница во времени, а на результат не влияет вовсе.

    **Считается на одной карте, и это решение, а не недосмотр.** Попытка
    раздать работу по двум через `DataParallel` кончилась тем, что прогон
    встал на первой же партии: тысяча пар не досчиталась за двенадцать
    часов, при том что первая партия даже не напечатала строку о себе.
    Раздача по картам на этом узле либо упирается в обмен между ними,
    либо не уживается с режимом вывода, и разбираться в этом дороже, чем
    двукратное ускорение. Одна карта предсказуема, а сколько она стоит,
    говорит замер перед прогоном.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    # Быстрый токенизатор — не мелочь. Медленный разбирает текст на питоне,
    # и на сотнях тысяч длинных пар он, а не видеокарта, становится узким
    # местом: карта при этом простаивает, а по длительности прогона
    # не отличить одно от другого. Зеркало модели может не содержать
    # tokenizer.json, и тогда разбор молча откатывается на медленный.
    if not getattr(tok, "is_fast", False):
        print("   ВНИМАНИЕ: токенизатор медленный (питоновский). На длинных "
              "парах он будет узким местом, а не видеокарта", flush=True)
    kwargs = common.half_kwargs(device)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, **kwargs)
    model = common.ensure_half(model.to(device), device).eval()

    порядок = sorted(range(len(pairs)), key=lambda i: len(pairs[i][1]))
    out = np.zeros(len(pairs), dtype="float32")
    t0 = time.time()
    # Время делится на разбор текста и счёт. Без этого деления «медленно»
    # не отличить от «не на той точности» и от «карта простаивает»:
    # все три выглядят одинаково — как долгий прогон.
    на_разбор = 0.0
    # no_grad, а не inference_mode: режим вывода создаёт тензоры, которые
    # не везде принимаются, а выигрыш на этой задаче незаметен.
    with torch.no_grad():
        for start in range(0, len(порядок), batch_size):
            кусок = порядок[start:start + batch_size]
            t_tok = time.time()
            batch = tok([pairs[i][0] for i in кусок], [pairs[i][1] for i in кусок],
                        padding=True, truncation=True, max_length=max_length,
                        return_tensors="pt").to(device)
            на_разбор += time.time() - t_tok
            logits = model(**batch).logits.view(-1).float().cpu().numpy()
            out[кусок] = logits
            # Печатаем часто и с первой партии: прогон, который молчит,
            # невозможно отличить от прогона, который встал.
            if (start // batch_size) % 20 == 0:
                сделано = start + len(кусок)
                скорость = сделано / max(time.time() - t0, 1e-6)
                осталось = (len(порядок) - сделано) / max(скорость, 1e-6) / 60
                print(f"   учитель: {сделано}/{len(порядок)} пар, "
                      f"{скорость:.0f} пар/с, осталось ~{осталось:.0f} мин", flush=True)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    всего = time.time() - t0
    print(f"   учитель: {len(порядок)} пар за {всего / 60:.1f} мин "
          f"({len(порядок) / всего:.1f} пар/с), из них на разбор текста "
          f"{на_разбор / 60:.1f} мин ({100 * на_разбор / всего:.0f}%)", flush=True)
    return out


def candidates_for(queries: list[dict], chunks: list[dict], depth: int,
                   device: str, batch_size: int = 128,
                   student: str = STUDENT) -> dict[str, list[str]]:
    """Гибридная выдача по обучающим фрагментам: BM25 плюс плотный поиск."""
    ids = [c["chunk_id"] for c in chunks]
    texts = [c["text"] for c in chunks]

    t0 = time.time()
    bm25 = BM25Index.build(ids, texts)
    print(f"   BM25 по {len(ids)} фрагментам построен за {time.time() - t0:.0f} с",
          flush=True)

    spec = MODELS[SPEC]
    # Загрузка под будильником. Зависание — не ошибка, его не ловит ни один
    # `try`: сессия просто стоит, пока не кончится время. Один прогон уже
    # простоял так двенадцать часов и не оставил ничего.
    import gpu_search as S
    t0 = time.time()
    with S.time_limit(S.MODEL_TIMEOUT):
        model = common.load_encoder(student, device, spec.max_seq_length)
    print(f"   ученик загружен за {time.time() - t0:.0f} с", flush=True)

    def закодировать(энкодер, что: list[str], имя: str, блок: int = 4096) -> np.ndarray:
        """Кодирование блоками, с отчётом после каждого.

        Один вызов на сорок тысяч текстов — это полчаса молчания,
        неотличимого от зависания. Блоками видно скорость с первой минуты,
        и прогон, который встал, виден сразу.

        Полоса прогресса выключена нарочно: в журнале Kaggle она
        разворачивается в сотни тысяч строк и сама становится обузой.
        """
        куски = []
        t = time.time()
        for начало in range(0, len(что), блок):
            кусок = что[начало:начало + блок]
            куски.append(энкодер.encode(кусок, batch_size=batch_size,
                                      convert_to_numpy=True,
                                      normalize_embeddings=True,
                                      show_progress_bar=False))
            сделано = начало + len(кусок)
            прошло = time.time() - t
            print(f"   {имя}: {сделано}/{len(что)}, {сделано / прошло:.0f} текст/с, "
                  f"осталось ~{(len(что) - сделано) / max(сделано / прошло, 1e-6) / 60:.0f} мин",
                  flush=True)
        return np.vstack(куски) if len(куски) > 1 else куски[0]

    corpus = закодировать(model, [spec.passage_prefix + t for t in texts],
                          "фрагменты")
    qvec = закодировать(model, [spec.query_prefix + q["text"] for q in queries],
                        "вопросы")
    del model

    import torch
    if device == "cuda":
        torch.cuda.empty_cache()
    dtype = torch.float16 if device == "cuda" else torch.float32
    mat = torch.from_numpy(corpus).to(device, dtype)
    q = torch.from_numpy(qvec).to(device, dtype)
    плотный: dict[str, list[tuple[str, float]]] = {}
    for start in range(0, q.shape[0], 64):
        scores = q[start:start + 64] @ mat.T
        top = torch.topk(scores.float(), k=min(depth, scores.shape[1]), dim=1)
        for row, (idx, val) in enumerate(zip(top.indices.tolist(), top.values.tolist())):
            плотный[queries[start + row]["query_id"]] = [
                (ids[j], float(s)) for j, s in zip(idx, val)]
    del mat, q
    if device == "cuda":
        torch.cuda.empty_cache()

    out: dict[str, list[str]] = {}
    for query in queries:
        qid = query["query_id"]
        лексика = [(c, 0.0) for c in bm25.search(query["text"], depth)]
        слито = [c for c, _ in rrf([лексика, плотный[qid]], top=depth)]
        # эталон вставляется силой: без его оценки не работает отсечка
        if query["gold_chunk_id"] not in слито:
            слито = слито[:depth - 1] + [query["gold_chunk_id"]]
        out[qid] = слито
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--train", default=None, help="synthetic_train.jsonl; ищется сам")
    ap.add_argument("--out", default="/kaggle/working/train")
    ap.add_argument("--chunks-cache", default="/kaggle/working/chunks")
    ap.add_argument("--teacher", default=TEACHER,
                    help="учитель: имя на HuggingFace или путь. Подключённая "
                         "входом модель находится сама")
    ap.add_argument("--student", default=STUDENT,
                    help="ученик: им считается плотная часть гибридной выдачи")
    ap.add_argument("--probe", type=int, default=ЗАМЕР_ПО_УМОЛЧАНИЮ,
                    help="на скольких парах мерить скорость учителя перед "
                         "прогоном; 0 — не мерить. Объём назначается "
                         "по замеру, а не по арифметике: она здесь "
                         "ошибалась вчетверо")
    ap.add_argument("--max-hours", type=float, default=0.0,
                    help="отказаться считать, если по замеру прогон не "
                         "уложится в столько часов")
    ap.add_argument("--verify-teacher", type=int, default=20,
                    help="на скольких запросах воспроизводить выдачу этапа 2; "
                         "0 — не сверять, но тогда источник весов ничем "
                         "не подтверждён")
    ap.add_argument("--depth", type=int, default=30,
                    help="сколько кандидатов на вопрос. Цена прогона линейна "
                         "по глубине: 30 даёт 183 тысячи проходов учителя")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--shard", type=int, default=250,
                    help="сколько вопросов считать между записями на диск. "
                         "Пачка поменьше — чаще отчёт и меньше потерь "
                         "при обрыве, побольше — меньше накладных")
    ap.add_argument("--limit", type=int, default=0, help="для пробы")
    ap.add_argument("--per-query", type=int, default=2,
                    help="сколько негативов на вопрос нужно этапу B")
    args = ap.parse_args()

    device = common.pick_device()
    print(f"устройство: {device}", flush=True)
    if device != "cuda":
        print("ВНИМАНИЕ: видеокарта не подключена. Settings -> Accelerator -> GPU",
              flush=True)

    # Веса: подключённая входом модель, иначе HuggingFace. Какой источник
    # сработал, печатается — молча подменять источник весов нельзя.
    teacher = common.resolve_model(args.teacher)
    student = common.resolve_model(args.student)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "train_prepared.jsonl")
    report_path = os.path.join(args.out, "report_prepare.json")
    # Уже посчитанное не пересчитывается, и это не оптимизация, а условие
    # работы: сессия ограничена по времени, а прогон идёт часами. Прежний
    # файл дочитывается, его вопросы выбрасываются из очереди, новые
    # дописываются в хвост.
    список_готовых: list[dict] = []
    if os.path.exists(out_path):
        список_готовых = read_jsonl(out_path)
        print(f"найдено посчитанное раньше: {len(список_готовых)} вопросов "
              f"в {out_path}", flush=True)

    # Устройство моделей сверяется до всякого счёта: зеркало это чужая
    # копия, и подменённые веса ничем себя не выдадут.
    арх_учителя = check_arch(teacher, {"num_labels": 1}, "учитель")
    арх_ученика = check_arch(student, {"hidden_size": 384}, "ученик")

    ruler = common.make_ruler()
    acts = common.load_acts(args.acts)
    chunks = common.build_chunks_cached(acts, "base", ruler, args.chunks_cache)

    split_path = args.split or common.find_file("split.json")
    if split_path is None:
        raise SystemExit("не найден split.json: кандидаты обязаны браться только "
                         "из обучающих актов, и без сплита это не проверить")
    split = SP.load(split_path)
    по_сплиту = sum(stat["фрагментов"] for stat in split["статистика"].values())
    if по_сплиту != len(chunks):
        raise SystemExit(
            f"нарезка разошлась со сплитом: здесь {len(chunks)} фрагментов, "
            f"а сплит посчитан по {по_сплиту}. Идентификатор фрагмента "
            f"позиционный, поэтому эталоны съедут молча. Положите каноническую "
            f"нарезку в {args.chunks_cache}/base.jsonl и снесите то, что там лежит.")
    print(f"нарезка сошлась со сплитом: {len(chunks)} фрагментов", flush=True)

    разрешено = train_acts(split)
    обучающие = [c for c in chunks if c["chunk_id"].split("#", 1)[0] in разрешено]
    print(f"обучающих фрагментов {len(обучающие)} из {len(chunks)} "
          f"({len(разрешено)} актов)", flush=True)

    train_path = args.train or common.find_file("synthetic_train.jsonl")
    if train_path is None:
        raise SystemExit("не найден synthetic_train.jsonl: подключите датасет "
                         "с набором запросов")
    все_вопросы = read_jsonl(train_path)
    if args.limit:
        все_вопросы = все_вопросы[:args.limit]
    готово = {r["query_id"] for r in список_готовых}
    queries = [q for q in все_вопросы if q["query_id"] not in готово]
    print(f"обучающих вопросов {len(все_вопросы)}: {train_path}", flush=True)
    if готово:
        print(f"   из них посчитано раньше {len(готово)}, осталось {len(queries)}",
              flush=True)
    if not queries:
        print("считать нечего: все вопросы уже посчитаны", flush=True)
    assert_only_train_acts([q["gold_chunk_id"] for q in queries], split, "эталоны обучения")

    # Замер скорости — первым делом. Минута, которая говорит, во что
    # обойдётся весь прогон, и даёт отказаться до того, как часы потрачены.
    замер = {"замер": "выключен"}
    if args.probe:
        замер = probe_speed(teacher, обучающие, device, sample=args.probe,
                            batch_size=args.batch_size)
        всего_пар = len(queries) * args.depth
        часов = всего_пар / замер["пар в секунду"] / 3600
        замер["прогон, часов"] = round(часов, 2)
        print(f"\nзамер: {замер['пар в секунду']} пар/с, весь прогон "
              f"({всего_пар} пар) — {часов:.1f} ч", flush=True)
        if args.max_hours and часов > args.max_hours:
            raise SystemExit(
                f"по замеру прогон займёт {часов:.1f} ч при пределе "
                f"{args.max_hours} ч. Уменьшите глубину (--depth) или "
                f"поднимите предел осознанно: начатый и убитый на середине "
                f"прогон не оставляет ничего.")

    # Сверка учителя — до дорогой части. Тридцать секунд против полутора
    # часов, потраченных на оценки чужой модели.
    сверка = ({"сверка": "выключена"} if not args.verify_teacher
              else verify_teacher(teacher, chunks, device,
                                  сколько=args.verify_teacher,
                                  batch_size=args.batch_size))

    print(f"\nпроходов кросс-энкодера: {len(queries)} × {args.depth} = "
          f"{len(queries) * args.depth}", flush=True)

    t0 = time.time()
    кандидаты = (candidates_for(queries, обучающие, args.depth, device,
                                student=student) if queries else {})
    секунд_выдача = time.time() - t0

    тексты = {c["chunk_id"]: c["text"] for c in обучающие}

    # Пачками, с дописыванием после каждой. Прогон на несколько часов идёт
    # вслепую — логи Kaggle отдаёт только по завершении, — и единственное,
    # что отличает убитый прогон от бесполезного, это записанное на диск.
    def записать(пачка: list[dict], адрес: list, оценки) -> list[dict]:
        по_вопросу: dict[str, dict[str, float]] = {}
        for (qid, chunk_id), score in zip(адрес, оценки):
            по_вопросу.setdefault(qid, {})[chunk_id] = float(score)
        вышло = []
        with open(out_path, "a", encoding="utf-8") as f:
            for q in пачка:
                свои = по_вопросу[q["query_id"]]
                gold = q["gold_chunk_id"]
                упорядочено = sorted(свои.items(), key=lambda kv: -kv[1])
                rec = {"query_id": q["query_id"], "gold_chunk_id": gold,
                       "gold_score": свои[gold],
                       "teacher_place": [c for c, _ in упорядочено].index(gold) + 1,
                       "candidates": [[c, s] for c, s in упорядочено]}
                вышло.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return вышло

    записи: list[dict] = список_готовых
    все_оценки: list[float] = []
    всего_пачек = (len(queries) + args.shard - 1) // args.shard
    for н, начало in enumerate(range(0, len(queries), args.shard), start=1):
        пачка = queries[начало:начало + args.shard]
        пары, адрес = [], []
        for q in пачка:
            for chunk_id in кандидаты[q["query_id"]]:
                пары.append((q["text"], тексты[chunk_id]))
                адрес.append((q["query_id"], chunk_id))
        print(f"\nпачка {н} из {всего_пачек}: вопросов {len(пачка)}, "
              f"пар {len(пары)}", flush=True)
        оценки = score_pairs(пары, teacher, device, batch_size=args.batch_size)
        все_оценки += оценки.tolist()
        записи += записать(пачка, адрес, оценки)
        прошло = time.time() - t0
        осталось = прошло / н * (всего_пачек - н)
        print(f"   записано всего {len(записи)} вопросов, прошло "
              f"{прошло / 60:.0f} мин, осталось ~{осталось / 60:.0f} мин",
              flush=True)

    scale = detect_scale(все_оценки or [r["gold_score"] for r in записи])
    правила = NegativeRules(per_query=args.per_query, train_acts=разрешено)
    _, stats = pick_all(записи, правила, scale)
    места = sorted(r["teacher_place"] for r in записи)
    report = {
        "учитель": args.teacher,
        "веса учителя": teacher,
        "устройство учителя": арх_учителя,
        "ученик": args.student,
        "веса ученика": student,
        "устройство ученика": арх_ученика,
        "сверка учителя с этапом 2": сверка,
        "замер скорости": замер,
        "шкала оценок": scale,
        "глубина": args.depth,
        "вопросов": len(записи),
        "посчитано в этом прогоне": len(queries),
        "обучающих фрагментов": len(обучающие),
        "обучающих актов": len(разрешено),
        "проходов учителя": len(пары),
        "секунд на выдачу": round(секунд_выдача, 1),
        "секунд всего": round(time.time() - t0, 1),
        "медиана места эталона у учителя": места[len(места) // 2],
        "эталон первым у учителя": sum(1 for m in места if m == 1),
        "негативы": stats,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)

    print("\n==== D0 готово ====", flush=True)
    for k, v in report.items():
        print(f"   {k:<34} {v if not isinstance(v, dict) else ''}", flush=True)
        if isinstance(v, dict):
            for k2, v2 in v.items():
                print(f"      {k2:<32} {v2}", flush=True)
    print(f"\nфайл: {out_path}, {os.path.getsize(out_path) / 1048576:.1f} МБ", flush=True)
    print("его обязательно сохранить: этапы B и C берут негативы и оценки "
          "отсюда и не пересчитывают", flush=True)


if __name__ == "__main__":
    main()
