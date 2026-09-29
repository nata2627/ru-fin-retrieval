"""Метрики качества поиска и доверительные интервалы к ним.

Считаются Recall@k, MRR@10 и NDCG@10. Различие в один-два процентных пункта
на двух сотнях запросов — это чаще всего шум, поэтому голое среднее здесь
бесполезно: к каждой метрике считается бутстрэп-интервал по запросам,
а разница между конфигурациями — парным бутстрэпом на одних и тех же запросах.
Парный вариант важен: конфигурации сравниваются на общих данных, и их ошибки
скоррелированы, так что независимые интервалы завысили бы неопределённость.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

# оценки релевантности: 0 — не релевантен, 1 — частично, 2 — прямой ответ
Qrels = Mapping[str, Mapping[str, int]]        # query_id -> {chunk_id: оценка}
Runs = Mapping[str, Sequence[str]]             # query_id -> список chunk_id по убыванию


def recall_at_k(ranked: Sequence[str], rel: Mapping[str, int], k: int) -> float:
    """Доля найденных релевантных среди первых k.

    Релевантными считаются чанки с оценкой 1 и выше.
    """
    good = {c for c, g in rel.items() if g > 0}
    if not good:
        return math.nan
    return len(good & set(ranked[:k])) / len(good)


def mrr_at_k(ranked: Sequence[str], rel: Mapping[str, int], k: int = 10) -> float:
    for i, c in enumerate(ranked[:k], start=1):
        if rel.get(c, 0) > 0:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], rel: Mapping[str, int], k: int = 10) -> float:
    """NDCG с градациями релевантности: вклад = (2^оценка - 1) / log2(позиция + 1)."""
    dcg = sum((2 ** rel.get(c, 0) - 1) / math.log2(i + 1)
              for i, c in enumerate(ranked[:k], start=1))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg else math.nan


METRICS = {
    "Recall@1":  lambda r, rel: recall_at_k(r, rel, 1),
    "Recall@5":  lambda r, rel: recall_at_k(r, rel, 5),
    "Recall@10": lambda r, rel: recall_at_k(r, rel, 10),
    "MRR@10":    lambda r, rel: mrr_at_k(r, rel, 10),
    "NDCG@10":   lambda r, rel: ndcg_at_k(r, rel, 10),
}


def per_query(run: Runs, qrels: Qrels) -> dict[str, np.ndarray]:
    """Значение каждой метрики отдельно по каждому запросу.

    Хранить поквериные значения обязательно: без них не посчитать ни интервал,
    ни парное сравнение, а по среднему нельзя восстановить разброс.
    """
    qids = [q for q in qrels if q in run]
    out = {}
    for name, fn in METRICS.items():
        out[name] = np.array([fn(run[q], qrels[q]) for q in qids], dtype=float)
    out["_qids"] = np.array(qids)
    return out


@dataclass
class Interval:
    mean: float
    lo: float
    hi: float

    def __str__(self) -> str:
        return f"{self.mean:.3f} [{self.lo:.3f}; {self.hi:.3f}]"


def bootstrap_ci(values: np.ndarray, n: int = 10000, level: float = 0.95,
                 seed: int = 0) -> Interval:
    """Доверительный интервал среднего бутстрэпом по запросам."""
    v = values[~np.isnan(values)]
    if v.size == 0:
        return Interval(math.nan, math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n, v.size), replace=True).mean(axis=1)
    a = (1 - level) / 2
    return Interval(float(v.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a)))


def unpaired_diff_ci(a: np.ndarray, b: np.ndarray, n: int = 10000, level: float = 0.95,
                     seed: int = 0) -> Interval:
    """Интервал разницы средних для двух **разных** наборов запросов.

    Нужен там, где сравниваются группы, а не конфигурации: например, разрыв
    с базовой линией на длинных эталонах против разрыва на коротких. Запросы
    в группах разные, пары нет, и парный бутстрэп здесь неприменим — каждая
    выборка пересобирается независимо.

    Интервал получается заметно шире парного, и это не недостаток метода,
    а честная цена: сравнение по разным запросам действительно знает меньше.
    """
    x = a[~np.isnan(a)]
    y = b[~np.isnan(b)]
    if x.size == 0 or y.size == 0:
        return Interval(math.nan, math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = (rng.choice(x, size=(n, x.size), replace=True).mean(axis=1)
             - rng.choice(y, size=(n, y.size), replace=True).mean(axis=1))
    lo, hi = np.quantile(means, ((1 - level) / 2, 1 - (1 - level) / 2))
    return Interval(float(x.mean() - y.mean()), float(lo), float(hi))


def paired_diff_ci(a: np.ndarray, b: np.ndarray, n: int = 10000, level: float = 0.95,
                   seed: int = 0) -> tuple[Interval, float]:
    """Интервал разницы (a - b) парным бутстрэпом и доля перестановок против.

    Возвращает интервал разницы и p-значение перестановочного теста: доля
    случаев, когда случайная смена знака даёт разницу не меньше наблюдаемой.
    Если интервал накрывает ноль, разницу нельзя считать установленной.
    """
    mask = ~(np.isnan(a) | np.isnan(b))
    d = (a - b)[mask]
    if d.size == 0:
        return Interval(math.nan, math.nan, math.nan), math.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n, d.size))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, (1 - level) / 2), np.quantile(means, 1 - (1 - level) / 2)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n, d.size))
    perm = (d * signs).mean(axis=1)
    p = float((np.abs(perm) >= abs(d.mean())).mean())
    return Interval(float(d.mean()), float(lo), float(hi)), p
