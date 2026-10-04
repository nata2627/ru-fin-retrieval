"""Оценка обученной модели — ровно тем же путём, что у остальных моделей.

Это требование, а не удобство. Та же нарезка `base`, те же префиксы
«query: » и «passage: » из `model_specs`, та же глубина выдачи, те же
метрики из `rufin.metrics`. Любое отличие в протоколе превращает
сравнение моделей в сравнение протоколов, и заметить это по самим числам
нельзя.

Поэтому здесь нет ничего своего: эмбеддинги считаются как в `gpu_embed`,
поиск — как в `gpu_search`, метрики — тем же кодом, что считает их на маке.
Отличие одно: модель берётся из папки с весами, а не из таблицы моделей,
потому что в таблицу она попадёт только после публикации.

**Про матрёшку.** Обрезка вектора не требует второго прогона: вектор
считается один раз на полную размерность, а затем берутся первые 256
и первые 128 измерений и нормируются заново. Нормировка обязательна —
после обрезки длина уже не единица, а косинус считается внутренним
произведением нормированных векторов.
"""
from __future__ import annotations

import os
import time

import numpy as np

from rufin import metrics as M
from rufin.retrieval.dense import truncate
from rufin.retrieval.model_specs import MODELS
from rufin.training.config import SPEC

TOP = 50          # та же глубина, что у всех конфигураций проекта


def load_model(path: str, device: str, max_seq_length: int = 512):
    """Поднять энкодер через transformers напрямую.

    **Не через `sentence-transformers`, и это дорого выученное решение.**
    Тот путь загрузки дважды уводил прогон в зависание на двенадцать часов,
    молча: ни ошибки, ни строки в журнале, будильник не срабатывал.
    Разбираться в чужом пути загрузки дороже, чем обойтись без него.

    Пулинг у e5 — среднее по токенам с маской, и это ровно то, что делает
    обёртка для этой модели. Результат совпадает, зависимостей меньше.
    Веса, сохранённые обучением, лежат в том же виде, что и у исходной
    модели, поэтому читаются так же.
    """
    import torch
    from gpu_common import ensure_half, half_kwargs
    from transformers import AutoModel, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(path, use_fast=True,
                                        local_files_only=True)
    model = AutoModel.from_pretrained(path, local_files_only=True,
                                      **half_kwargs(device))
    model = ensure_half(model.to(device), device).eval()
    print(f"   энкодер загружен за {time.time() - t0:.0f} с "
          f"(быстрый токенизатор: {getattr(tok, 'is_fast', False)})", flush=True)
    assert torch is not None
    return tok, model, max_seq_length


def encode(model, texts: list[str], prefix: str, batch_size: int = 128,
           progress: bool = False, блок: int = 4096) -> np.ndarray:
    """Эмбеддинги блоками, с отчётом о скорости после каждого.

    Один вызов на шестьдесят тысяч текстов — это несколько минут молчания,
    неотличимого от зависания, а неотличие уже стоило двух сессий.
    """
    import torch
    tok, сеть, max_len = model
    if prefix:
        texts = [prefix + t for t in texts]
    куски = []
    t = time.time()
    with torch.no_grad():
        for начало in range(0, len(texts), блок):
            кусок = texts[начало:начало + блок]
            части = []
            for н in range(0, len(кусок), batch_size):
                b = tok(кусок[н:н + batch_size], padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt").to(сеть.device)
                h = сеть(**b).last_hidden_state
                маска = b["attention_mask"].unsqueeze(-1).to(h.dtype)
                вектор = (h * маска).sum(1) / маска.sum(1).clamp(min=1e-9)
                части.append(torch.nn.functional.normalize(вектор, p=2, dim=1)
                             .float().cpu().numpy())
            куски.append(np.vstack(части))
            сделано = начало + len(кусок)
            if progress:
                прошло = time.time() - t
                print(f"   эмбеддинги: {сделано}/{len(texts)}, "
                      f"{сделано / прошло:.0f} текст/с", flush=True)
    return np.vstack(куски).astype("float32")


# Обрезка вектора берётся из `rufin.retrieval.dense`, а не пишется здесь:
# ею же пользуется локальный поиск, и расходиться им нельзя. Разойдись —
# обученная модель на маке искала бы иначе, чем на видеокарте, где её мерили.


def search(corpus: np.ndarray, ids: list[str], qvec: np.ndarray, device: str,
           top: int = TOP, block: int = 64) -> dict[int, list[tuple[str, float]]]:
    """Поиск внутренним произведением. Возвращает выдачу по номеру запроса."""
    import torch
    dtype = torch.float16 if device == "cuda" else torch.float32
    mat = torch.from_numpy(corpus).to(device, dtype)
    q = torch.from_numpy(qvec).to(device, dtype)
    out: dict[int, list[tuple[str, float]]] = {}
    for start in range(0, q.shape[0], block):
        scores = q[start:start + block] @ mat.T
        лучшие = torch.topk(scores.float(), k=min(top, scores.shape[1]), dim=1)
        for row, (idx, val) in enumerate(zip(лучшие.indices.tolist(), лучшие.values.tolist())):
            out[start + row] = [(ids[j], float(s)) for j, s in zip(idx, val)]
    del mat, q
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def evaluate(path: str, chunks: list[dict], queries: list[dict], qrels: dict,
             device: str, dims: tuple[int, ...] = (), batch_size: int = 128,
             top: int = TOP, model=None) -> dict:
    """Оценить модель на наборе запросов. Ключ результата — размерность.

    `dims` пустой означает «только полная размерность». Эмбеддинги корпуса
    считаются один раз на все размерности: обрезка их не требует.
    """
    spec = MODELS[SPEC]
    свою = model is None
    model = model or load_model(path, device, spec.max_seq_length)
    ids = [c["chunk_id"] for c in chunks]

    t0 = time.time()
    corpus = encode(model, [c["text"] for c in chunks], spec.passage_prefix,
                    batch_size, progress=True)
    секунд = time.time() - t0
    print(f"   корпус: {len(ids)} фрагментов за {секунд:.0f} с "
          f"({len(ids) / секунд:.0f} фрагм./с)", flush=True)
    qvec = encode(model, [q["text"] for q in queries], spec.query_prefix, batch_size)
    if свою:
        del model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out: dict = {}
    for dim in (dims or (corpus.shape[1],)):
        выдача = search(truncate(corpus, dim), ids, truncate(qvec, dim), device, top)
        runs = {queries[i]["query_id"]: [c for c, _ in выдача[i]] for i in выдача}
        общие = {q: v for q, v in qrels.items() if q in runs}
        pq = M.per_query({q: runs[q][:10] for q in общие}, общие)
        out[int(dim)] = {
            "runs": runs,
            "per_query": {name: pq[name].tolist() for name in M.METRICS},
            "qids": [str(q) for q in pq["_qids"]],
            "NDCG@10": M.bootstrap_ci(pq["NDCG@10"]).__dict__,
            "размер матрицы МБ": round(
                len(ids) * int(dim) * 2 / 1048576, 1),  # как float16, в чём она и хранится
        }
        print(f"   dim {dim}: NDCG@10 {M.bootstrap_ci(pq['NDCG@10'])} "
              f"по {len(общие)} запросам", flush=True)
    out["секунд на корпус"] = round(секунд, 1)
    return out


def размерности(итог: dict) -> list[int]:
    """Размерности из результата оценки, по убыванию.

    Отдельной функцией, потому что в этом словаре ключи разного рода:
    размерности числами и «секунд на корпус» строкой. Прямой `sorted`
    по такому словарю падает, сравнивая число со строкой, и падает он
    в отчёте — то есть после того, как этап уже обучен и оценён.
    Один раз это стоило шестнадцати минут видеокарты.
    """
    return sorted((d for d in итог if isinstance(d, int)), reverse=True)


def dev_ids(queries: list[dict], qrels: dict, subsets: dict | None = None) -> set[str]:
    """Какие запросы считать dev.

    Состав берётся из `podvyborki.json`, если он приехал: это тот же файл,
    по которому считает разбивку локальный отчёт, и расходиться им нельзя.
    Если файла нет — по приставке идентификатора, которую назначает
    генерация dev.
    """
    размечено = {q["query_id"] for q in queries} & set(qrels)
    if subsets and "dev" in subsets:
        return размечено & set(subsets["dev"])
    return {q for q in размечено if q.startswith("dv")}


def save_runs(out_root: str, runs: dict[str, list[str]], name: str,
              config: str = "base") -> str:
    """Сохранить выдачу в том же виде, в каком её ждёт локальный отчёт."""
    import gpu_search as S
    os.makedirs(out_root, exist_ok=True)
    return S.save_run(out_root, config, name, runs)
