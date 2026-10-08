#!/usr/bin/env python3
"""Живые вопросы без номеров актов: выдачи на видеокарте.

Опыт парный. Те же 142 живых вопроса в трёх видах — как заданы, без
номеров актов и пунктов, без ссылок целиком. Правила правки и причины,
по которым сравнение именно парное, в `src/rufin/references.py`; разбор
первых результатов в `docs/08-nomera-aktov.md`.

Здесь считается то, для чего нужна карта: эмбеддинги корпуса исходной
`multilingual-e5-small` и дообученной моделью, выдачи по 426 запросам.
BM25 считается заодно — индекс строится один раз на все варианты.
Метрики, интервалы и парные тесты остаются локальными
(`scripts/act_number_effect.py`): по спискам идентификаторов они
не требуют ни памяти, ни карты.

Матрицы корпуса сохраняются в вывод. На этапе A они не сохранялись, и
из-за этого любой опыт, меняющий только запросы, требовал пересчёта
корпуса заново. Матрица e5-small весит 46 МБ — несопоставимо дешевле
трёх минут карты за каждый такой вопрос.

Нарезка берётся готовой из входов и сверяется с `ids.txt` той матрицы,
по которой посчитаны прежние выдачи. Пересчёт нарезки зависит от версии
токенизатора: при сдвиге границ идентификатор «акт#номер» не исчезает,
а начинает указывать на другой текст, и выдачи молча оказываются
несравнимыми с уже посчитанными. Поэтому расхождение здесь — отказ
считать, а не предупреждение.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gpu_common as common  # noqa: E402
import gpu_search as S  # noqa: E402
import gpu_traineval as E  # noqa: E402

from rufin.retrieval.hybrid import rrf  # noqa: E402
from rufin.retrieval.model_specs import MODELS  # noqa: E402
from rufin.training.config import SPEC, STUDENT  # noqa: E402

ТЕГ = "live-variants"
ЗАПРОСЫ = "live_variants.jsonl"
ЭТАЛОННЫЕ_ИДЕНТИФИКАТОРЫ = "ids_base.txt"


def сверить_идентификаторы(chunks: list[dict], путь: str | None) -> dict:
    """Те же фрагменты, что у прежних выдач, или отказ считать."""
    мои = [c["chunk_id"] for c in chunks]
    if not путь:
        print(f"ВНИМАНИЕ: {ЭТАЛОННЫЕ_ИДЕНТИФИКАТОРЫ} среди входов нет, "
              f"сверить нарезку не с чем: {len(мои)} фрагментов", flush=True)
        return {"фрагментов": len(мои), "сверено": False}
    with open(путь, encoding="utf-8") as f:
        эталон = [l.rstrip("\n") for l in f if l.strip()]
    if эталон != мои:
        расхождение = next((i for i, (a, b) in enumerate(zip(эталон, мои)) if a != b),
                           min(len(эталон), len(мои)))
        raise SystemExit(
            f"нарезка разошлась с той, по которой посчитаны прежние выдачи: "
            f"здесь {len(мои)} фрагментов, там {len(эталон)}, первое "
            f"расхождение на месте {расхождение}. Выдачи были бы "
            f"несравнимы, а эталон съехал бы молча.")
    print(f"нарезка сверена с {os.path.basename(путь)}: {len(мои)} фрагментов",
          flush=True)
    return {"фрагментов": len(мои), "сверено": True}


def найти_веса(метка: str, корень: str = "/kaggle/input") -> str | None:
    """Папка с дообученными весами среди входов."""
    для = sorted(glob.glob(os.path.join(корень, "**", метка, "config.json"),
                           recursive=True), key=len)
    return os.path.dirname(для[0]) if для else None


def сохранить_матрицу(корень: str, имя: str, vec, ids: list[str], откуда: str) -> str:
    import numpy as np
    папка = os.path.join(корень, "base", имя)
    os.makedirs(папка, exist_ok=True)
    np.save(os.path.join(папка, "vectors.npy"), vec.astype("float16"))
    with open(os.path.join(папка, "ids.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(ids) + "\n")
    with open(os.path.join(папка, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"model": имя, "weights": откуда, "chunks": len(ids),
                   "dim": int(vec.shape[1]), "dtype": "float16"}, f,
                  ensure_ascii=False, indent=1)
    мб = os.path.getsize(os.path.join(папка, "vectors.npy")) / 1048576
    print(f"   матрица сохранена: {папка}, {мб:.0f} МБ", flush=True)
    return папка


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default=None)
    ap.add_argument("--queries", default=None)
    ap.add_argument("--chunks-cache", default="/kaggle/working/chunks")
    ap.add_argument("--runs", default="/kaggle/working/runs")
    ap.add_argument("--embeddings", default="/kaggle/working/embeddings")
    ap.add_argument("--weights-tag", default="d-matryoshka",
                    help="какой этап рецепта считать дообученной моделью")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--top", type=int, default=S.TOP)
    ap.add_argument("--no-bm25", action="store_true")
    ap.add_argument("--limit-chunks", type=int, default=0,
                    help="обрезать корпус: только для проверки самого прогона")
    args = ap.parse_args()

    device = common.pick_device()
    print(f"устройство: {device}", flush=True)
    spec = MODELS[SPEC]

    acts = common.load_acts(args.acts)
    chunks = common.build_chunks_cached(acts, "base", common.make_ruler,
                                        args.chunks_cache)
    отчёт: dict = {"устройство": device, "запросов": 0, "выдачи": {}}
    if args.limit_chunks:
        chunks = chunks[:args.limit_chunks]
        print(f"ВНИМАНИЕ: корпус обрезан до {len(chunks)} фрагментов, "
              f"числа несравнимы с прежними", flush=True)
        отчёт["обрезан корпус"] = len(chunks)
    else:
        отчёт["нарезка"] = сверить_идентификаторы(
            chunks, common.find_file(ЭТАЛОННЫЕ_ИДЕНТИФИКАТОРЫ))

    путь_запросов = args.queries or common.find_file(ЗАПРОСЫ)
    if not путь_запросов:
        raise SystemExit(
            f"не найден {ЗАПРОСЫ}: его собирает scripts/live_variants.py "
            f"и увозит `python3 scripts/kaggle_run.py queries`")
    with open(путь_запросов, encoding="utf-8") as f:
        queries = [json.loads(l) for l in f if l.strip()]
    варианты = sorted({q.get("variant", "?") for q in queries})
    print(f"запросы: {путь_запросов}, {len(queries)} штук, "
          f"вариантов {len(варианты)} ({', '.join(варианты)})", flush=True)
    отчёт["запросов"] = len(queries)
    отчёт["варианты"] = варианты

    os.makedirs(args.runs, exist_ok=True)
    выдачи: dict[str, dict[str, list[str]]] = {}

    if not args.no_bm25:
        print("\n=== BM25 ===", flush=True)
        runs, секунд = S.bm25_runs(chunks, queries)
        выдачи["bm25"] = runs
        путь = S.save_run(args.runs, ТЕГ, "bm25", runs)
        отчёт["выдачи"]["bm25"] = {"файл": os.path.basename(путь),
                                   "секунд на индекс": round(секунд, 1)}
        print(f"   {путь}", flush=True)

    # Исходная модель и дообученная считаются одним и тем же кодом: пулинг
    # средний по маске, префиксы «query: » и «passage: ». Иначе сравнение
    # моделей стало бы сравнением протоколов.
    модели: list[tuple[str, str]] = []
    исходная = common.resolve_model(STUDENT, обязательно=False)
    if os.path.exists(os.path.join(исходная, "config.json")):
        модели.append(("e5-small", исходная))
    else:
        print(f"ВНИМАНИЕ: исходная {STUDENT} входом не подключена, "
              f"пропускается", flush=True)
    веса = найти_веса(args.weights_tag)
    if веса:
        модели.append(("rufin", веса))
    else:
        print(f"ВНИМАНИЕ: весов {args.weights_tag} среди входов нет, "
              f"дообученная модель пропускается", flush=True)
    if not модели:
        raise SystemExit("ни одной модели на входе: нечего считать")

    ids = [c["chunk_id"] for c in chunks]
    тексты = [c["text"] for c in chunks]
    for имя, путь_весов in модели:
        print(f"\n=== {имя}: {путь_весов} ===", flush=True)
        model = E.load_model(путь_весов, device, spec.max_seq_length)
        t0 = time.time()
        corpus = E.encode(model, тексты, spec.passage_prefix, args.batch_size,
                          progress=True)
        секунд = time.time() - t0
        print(f"   корпус: {len(ids)} фрагментов за {секунд:.0f} с "
              f"({len(ids) / секунд:.0f} фрагм./с)", flush=True)
        сохранить_матрицу(args.embeddings, имя, corpus, ids, путь_весов)
        qvec = E.encode(model, [q["text"] for q in queries], spec.query_prefix,
                        args.batch_size)
        del model
        import torch
        if device == "cuda":
            torch.cuda.empty_cache()

        найдено = E.search(corpus, ids, qvec, device, args.top)
        runs = {queries[i]["query_id"]: [c for c, _ in найдено[i]] for i in найдено}
        выдачи[f"dense-{имя}"] = runs
        путь = S.save_run(args.runs, ТЕГ, f"dense-{имя}", runs)
        отчёт["выдачи"][f"dense-{имя}"] = {
            "файл": os.path.basename(путь), "веса": путь_весов,
            "секунд на корпус": round(секунд, 1),
            "размерность": int(corpus.shape[1])}
        print(f"   {путь}", flush=True)
        del corpus, qvec

        if "bm25" in выдачи:
            гибрид = {}
            for qid, плотная in runs.items():
                слито = rrf([[(c, 0.0) for c in выдачи["bm25"][qid]],
                             [(c, 0.0) for c in плотная]], top=args.top)
                гибрид[qid] = [c for c, _ in слито]
            путь = S.save_run(args.runs, ТЕГ, f"hybrid-{имя}", гибрид)
            отчёт["выдачи"][f"hybrid-{имя}"] = {"файл": os.path.basename(путь)}
            print(f"   {путь}", flush=True)

    with open(os.path.join(args.runs, "report_variants.json"), "w",
              encoding="utf-8") as f:
        json.dump(отчёт, f, ensure_ascii=False, indent=1)
    print("\nскачать: runs/live-variants__*.jsonl, report_variants.json, "
          "embeddings/base/*/", flush=True)
    print("дальше локально: python3 scripts/act_number_effect.py", flush=True)


if __name__ == "__main__":
    main()
