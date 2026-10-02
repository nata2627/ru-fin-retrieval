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


def score_pairs(pairs: list[tuple[str, str]], model_path: str, device: str,
                batch_size: int = 64, max_length: int = 512) -> np.ndarray:
    """Логиты кросс-энкодера по парам «вопрос, фрагмент».

    Пары сортируются по длине внутри прогона и возвращаются в исходном
    порядке: при выравнивании по самой длинной паре батча это заметная
    разница во времени, а на результат не влияет вовсе.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    kwargs = common.half_kwargs(device)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, **kwargs)
    model.to(device).eval()

    порядок = sorted(range(len(pairs)), key=lambda i: len(pairs[i][1]))
    out = np.zeros(len(pairs), dtype="float32")
    t0 = time.time()
    with torch.inference_mode():
        for start in range(0, len(порядок), batch_size):
            кусок = порядок[start:start + batch_size]
            batch = tok([pairs[i][0] for i in кусок], [pairs[i][1] for i in кусок],
                        padding=True, truncation=True, max_length=max_length,
                        return_tensors="pt").to(device)
            logits = model(**batch).logits.view(-1).float().cpu().numpy()
            out[кусок] = logits
            if (start // batch_size) % 100 == 0:
                сделано = start + len(кусок)
                скорость = сделано / max(time.time() - t0, 1e-6)
                осталось = (len(порядок) - сделано) / max(скорость, 1e-6) / 60
                print(f"   учитель: {сделано}/{len(порядок)} пар, "
                      f"{скорость:.0f} пар/с, осталось ~{осталось:.0f} мин", flush=True)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"   учитель: {len(порядок)} пар за {(time.time() - t0) / 60:.1f} мин", flush=True)
    return out


def candidates_for(queries: list[dict], chunks: list[dict], depth: int,
                   device: str, batch_size: int = 128) -> dict[str, list[str]]:
    """Гибридная выдача по обучающим фрагментам: BM25 плюс плотный поиск."""
    ids = [c["chunk_id"] for c in chunks]
    texts = [c["text"] for c in chunks]

    t0 = time.time()
    bm25 = BM25Index.build(ids, texts)
    print(f"   BM25 по {len(ids)} фрагментам построен за {time.time() - t0:.0f} с",
          flush=True)

    spec = MODELS[SPEC]
    model = common.load_encoder(STUDENT, device, spec.max_seq_length)
    t0 = time.time()
    corpus = model.encode([spec.passage_prefix + t for t in texts],
                          batch_size=batch_size, convert_to_numpy=True,
                          normalize_embeddings=True, show_progress_bar=True)
    qvec = model.encode([spec.query_prefix + q["text"] for q in queries],
                        batch_size=batch_size, convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=False)
    print(f"   эмбеддинги необученного ученика за {time.time() - t0:.0f} с", flush=True)
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
    ap.add_argument("--teacher", default=TEACHER)
    ap.add_argument("--depth", type=int, default=30,
                    help="сколько кандидатов на вопрос. Цена прогона линейна "
                         "по глубине: 30 даёт 183 тысячи проходов учителя")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="для пробы")
    ap.add_argument("--per-query", type=int, default=2,
                    help="сколько негативов на вопрос нужно этапу B")
    args = ap.parse_args()

    device = common.pick_device()
    print(f"устройство: {device}", flush=True)
    if device != "cuda":
        print("ВНИМАНИЕ: видеокарта не подключена. Settings -> Accelerator -> GPU",
              flush=True)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "train_prepared.jsonl")
    report_path = os.path.join(args.out, "report_prepare.json")
    if os.path.exists(out_path) and not args.limit:
        print(f"готово раньше: {out_path}. Пересчитывать незачем — удалите файл, "
              f"если нужен новый прогон", flush=True)
        return

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
    queries = read_jsonl(train_path)
    if args.limit:
        queries = queries[:args.limit]
    print(f"обучающих вопросов {len(queries)}: {train_path}", flush=True)
    assert_only_train_acts([q["gold_chunk_id"] for q in queries], split, "эталоны обучения")

    print(f"\nпроходов кросс-энкодера: {len(queries)} × {args.depth} = "
          f"{len(queries) * args.depth}", flush=True)

    t0 = time.time()
    кандидаты = candidates_for(queries, обучающие, args.depth, device)
    секунд_выдача = time.time() - t0

    тексты = {c["chunk_id"]: c["text"] for c in обучающие}
    пары: list[tuple[str, str]] = []
    адрес: list[tuple[str, str]] = []
    for q in queries:
        for chunk_id in кандидаты[q["query_id"]]:
            пары.append((q["text"], тексты[chunk_id]))
            адрес.append((q["query_id"], chunk_id))
    оценки = score_pairs(пары, args.teacher, device, batch_size=args.batch_size)

    по_вопросу: dict[str, dict[str, float]] = {}
    for (qid, chunk_id), score in zip(адрес, оценки):
        по_вопросу.setdefault(qid, {})[chunk_id] = float(score)

    scale = detect_scale(оценки.tolist())
    записи = []
    with open(out_path, "w", encoding="utf-8") as f:
        for q in queries:
            свои = по_вопросу[q["query_id"]]
            gold = q["gold_chunk_id"]
            упорядочено = sorted(свои.items(), key=lambda kv: -kv[1])
            rec = {"query_id": q["query_id"], "gold_chunk_id": gold,
                   "gold_score": свои[gold],
                   "teacher_place": [c for c, _ in упорядочено].index(gold) + 1,
                   "candidates": [[c, s] for c, s in упорядочено]}
            записи.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    правила = NegativeRules(per_query=args.per_query, train_acts=разрешено)
    _, stats = pick_all(записи, правила, scale)
    места = sorted(r["teacher_place"] for r in записи)
    report = {
        "учитель": args.teacher,
        "шкала оценок": scale,
        "глубина": args.depth,
        "вопросов": len(queries),
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
