"""Выдачи поисковых конфигураций на видеокарте.

Считается на Kaggle, потому что на восьми гигабайтах памяти это не помещается:
индекс BM25 по шестидесяти тысячам фрагментов держит в памяти все
токенизированные документы, а переранжирование двухсот запросов по пятьдесят
кандидатов — это десять тысяч проходов кросс-энкодера.

Обратно едут только списки идентификаторов — сотни килобайт вместо гигабайтов.
Метрики, доверительные интервалы и разбор ошибок считаются локально по этим
спискам: там уже нет ничего тяжёлого.

Задержка здесь не меряется. Её снимают на маке, где система и работает:
замер на чужой видеокарте не имеет отношения к делу.
"""
from __future__ import annotations

import json
import os
import signal
import time
from contextlib import contextmanager

import numpy as np

from rufin.retrieval.bm25 import BM25Index
from rufin.retrieval.hybrid import rrf
from rufin.retrieval.model_specs import MODELS

TOP = 50          # глубина выдачи: уходит в реранкер и в разбор ошибок
RERANKER = "BAAI/bge-reranker-v2-m3"
# Сколько ждать одну модель. Скачивание с HuggingFace без токена режется
# по скорости и может встать намертво: перехват исключений тут не помогает,
# потому что зависание — не ошибка. Поэтому жёсткий будильник.
MODEL_TIMEOUT = 600


class ModelTimeout(RuntimeError):
    pass


@contextmanager
def time_limit(seconds: int):
    def ring(signum, frame):
        raise ModelTimeout(f"не уложилось в {seconds} с")
    old = signal.signal(signal.SIGALRM, ring)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def bm25_runs(chunks: list[dict], queries: list[dict]) -> tuple[dict, float]:
    ids = [c["chunk_id"] for c in chunks]
    t0 = time.time()
    index = BM25Index.build(ids, [c["text"] for c in chunks])
    build = time.time() - t0
    print(f"   BM25 построен за {build:.0f} с", flush=True)
    out = {q["query_id"]: [c for c, _ in index.search(q["text"], TOP)] for q in queries}
    return out, build


def dense_runs(emb_dir: str, model_name: str, queries: list[dict],
               batch_size: int = 64, device: str = "cuda") -> tuple[dict, dict]:
    import torch

    from gpu_common import load_encoder

    vec = np.load(os.path.join(emb_dir, "vectors.npy"))
    ids = [l.rstrip("\n") for l in open(os.path.join(emb_dir, "ids.txt"), encoding="utf-8")]
    meta = json.load(open(os.path.join(emb_dir, "meta.json"), encoding="utf-8"))

    spec = MODELS[model_name]
    with time_limit(MODEL_TIMEOUT):
        model = load_encoder(spec.path, device, spec.max_seq_length)
    texts = [spec.query_prefix + q["text"] for q in queries] if spec.query_prefix \
        else [q["text"] for q in queries]
    qvec = model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=False)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # векторы нормированы, поэтому внутреннее произведение и есть косинус.
    # На процессоре половинная точность не поддерживается — считаем в float32.
    dtype = torch.float16 if device == "cuda" else torch.float32
    mat = torch.from_numpy(vec).to(device, dtype)
    q = torch.from_numpy(qvec).to(device, dtype)
    out = {}
    for start in range(0, q.shape[0], 32):
        scores = q[start:start + 32] @ mat.T
        top = torch.topk(scores.float(), k=min(TOP, scores.shape[1]), dim=1)
        for row, (idx, val) in enumerate(zip(top.indices.tolist(), top.values.tolist())):
            out[queries[start + row]["query_id"]] = [(ids[j], float(s)) for j, s in zip(idx, val)]
    del mat, q
    if device == "cuda":
        torch.cuda.empty_cache()
    return out, meta


def rerank_runs(queries: list[dict], candidates: dict, texts: dict,
                model_path: str = RERANKER, batch_size: int = 32,
                device: str = "cuda") -> dict:
    import torch

    from gpu_common import load_cross_encoder
    with time_limit(MODEL_TIMEOUT):
        model = load_cross_encoder(model_path, device)
    out = {}
    t0 = time.time()
    for n, q in enumerate(queries, 1):
        cand = candidates[q["query_id"]]
        if not cand:
            out[q["query_id"]] = []
            continue
        pairs = [(q["text"], texts[c]) for c, _ in cand]
        scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        order = sorted(zip((c for c, _ in cand), scores), key=lambda x: -float(x[1]))
        out[q["query_id"]] = [c for c, _ in order]
        if n % 50 == 0:
            print(f"   переранжировано {n}/{len(queries)}, {time.time() - t0:.0f} с", flush=True)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def save_run(out_root: str, tag: str, config: str, run: dict) -> str:
    os.makedirs(out_root, exist_ok=True)
    safe = config.replace(":", "-").replace("+", "-")
    path = os.path.join(out_root, f"{tag}__{safe}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for qid, docs in run.items():
            ranked = [d[0] if isinstance(d, tuple) else d for d in docs]
            f.write(json.dumps({"query_id": qid, "ranked": ranked}, ensure_ascii=False) + "\n")
    return path
