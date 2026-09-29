"""Метрики считаются руками на бумаге и сверяются с кодом.

Проверять метрики на настоящих данных бессмысленно: там не с чем сравнить.
Поэтому здесь крошечные выдачи, для которых правильный ответ выписывается
в комментарии.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from rufin import metrics as M


def test_recall_считает_долю_найденных():
    # релевантны a и z, в первой тройке нашёлся один из двух
    assert M.recall_at_k(["a", "b", "c"], {"a": 2, "z": 1}, 3) == pytest.approx(0.5)


def test_recall_учитывает_оценку_1_как_релевантную():
    assert M.recall_at_k(["a"], {"a": 1}, 1) == pytest.approx(1.0)


def test_recall_без_релевантных_не_число():
    # запрос без эталона не должен давать ноль: ноль — это «искали и не нашли»,
    # а здесь искать нечего, и такой запрос обязан выпасть из среднего
    assert math.isnan(M.recall_at_k(["a"], {"b": 0}, 5))


def test_recall_за_пределами_среза_не_считается():
    assert M.recall_at_k(["x", "y", "a"], {"a": 2}, 2) == pytest.approx(0.0)


def test_mrr_это_обратная_позиция_первого_релевантного():
    assert M.mrr_at_k(["x", "a", "b"], {"a": 2, "b": 2}) == pytest.approx(0.5)
    assert M.mrr_at_k(["x", "y"], {"a": 2}) == pytest.approx(0.0)


def test_ndcg_идеального_порядка_равен_единице():
    assert M.ndcg_at_k(["a", "b"], {"a": 2, "b": 1}) == pytest.approx(1.0)


def test_ndcg_градации_различает_двойку_и_единицу():
    # dcg = 1/log2(2) + 3/log2(3) = 1 + 1.89279 = 2.89279
    # idcg = 3/log2(2) + 1/log2(3) = 3 + 0.63093 = 3.63093
    assert M.ndcg_at_k(["b", "a"], {"a": 2, "b": 1}) == pytest.approx(0.79671, abs=1e-5)


def test_ndcg_не_бывает_больше_единицы():
    rel = {"a": 2, "b": 1, "c": 1}
    for ranked in (["a", "b", "c"], ["c", "b", "a"], ["a", "x", "b"]):
        assert M.ndcg_at_k(ranked, rel) <= 1.0 + 1e-9


def test_ndcg_без_эталона_не_число():
    assert math.isnan(M.ndcg_at_k(["a"], {}))


def test_per_query_берет_только_общие_запросы():
    run = {"q1": ["a"], "q2": ["b"], "q9": ["z"]}
    qrels = {"q1": {"a": 2}, "q2": {"c": 2}}
    out = M.per_query(run, qrels)
    assert list(out["_qids"]) == ["q1", "q2"]
    assert out["Recall@1"].tolist() == [1.0, 0.0]


def test_per_query_отдает_значения_по_каждому_запросу():
    # без поквериных значений не посчитать ни интервал, ни парное сравнение
    run = {f"q{i}": ["a"] for i in range(5)}
    qrels = {f"q{i}": {"a": 2} for i in range(5)}
    out = M.per_query(run, qrels)
    assert all(v.shape == (5,) for k, v in out.items() if not k.startswith("_"))


def test_интервал_накрывает_среднее():
    v = np.array([0.1, 0.5, 0.9, 0.4, 0.6, 0.3])
    ci = M.bootstrap_ci(v, n=2000, seed=0)
    assert ci.mean == pytest.approx(v.mean())
    assert ci.lo <= ci.mean <= ci.hi


def test_интервал_воспроизводим_по_зерну():
    v = np.random.default_rng(1).random(50)
    a = M.bootstrap_ci(v, n=1000, seed=7)
    b = M.bootstrap_ci(v, n=1000, seed=7)
    assert (a.lo, a.hi) == (b.lo, b.hi)


def test_интервал_сужается_с_ростом_выборки():
    # это и есть довод про размер набора запросов: ширина падает как корень из n
    rng = np.random.default_rng(0)
    узкий = M.bootstrap_ci(rng.random(1000), n=1000, seed=0)
    широкий = M.bootstrap_ci(rng.random(30), n=1000, seed=0)
    assert (узкий.hi - узкий.lo) < (широкий.hi - широкий.lo)


def test_интервал_пропускает_запросы_без_эталона():
    v = np.array([1.0, np.nan, 0.0])
    assert M.bootstrap_ci(v, n=500, seed=0).mean == pytest.approx(0.5)


def test_интервал_пустой_выборки_не_падает():
    ci = M.bootstrap_ci(np.array([np.nan, np.nan]), n=100)
    assert math.isnan(ci.mean) and math.isnan(ci.lo)


def test_парное_сравнение_одинаковых_выдач_не_находит_разницы():
    v = np.array([0.2, 0.7, 0.5, 0.9])
    ci, p = M.paired_diff_ci(v, v, n=2000, seed=0)
    assert ci.mean == pytest.approx(0.0)
    assert p == pytest.approx(1.0)


def test_парное_сравнение_находит_устойчивую_разницу():
    a = np.full(40, 0.8)
    b = np.full(40, 0.3)
    ci, p = M.paired_diff_ci(a, b, n=2000, seed=0)
    assert ci.mean == pytest.approx(0.5)
    assert ci.lo > 0          # интервал не накрывает ноль
    assert p < 0.01


def test_парное_сравнение_считает_только_общие_запросы():
    a = np.array([1.0, np.nan, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    ci, _ = M.paired_diff_ci(a, b, n=500, seed=0)
    assert ci.mean == pytest.approx(0.5)      # (1-0) и (0-0), второй выброшен


def test_интервал_печатается_с_тремя_знаками():
    assert str(M.Interval(0.5, 0.4, 0.6)) == "0.500 [0.400; 0.600]"
