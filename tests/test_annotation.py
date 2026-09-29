"""Согласие разметчиков и то, на чём оно держится.

Каппа — единственное число, которым в проекте измеряется качество самой
разметки, и ошибиться в ней значит выдать разметку судьи за проверенную.
Отдельно проверяется, что инструкция человеку и подсказка судье — один
и тот же текст: разойдись они, каппа померила бы разницу в инструкциях.
"""
from __future__ import annotations

import pathlib

import pytest

from rufin.annotation import RUBRIC, cohen_kappa, kappa_verdict, stratified_sample

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_полное_совпадение_даёт_единицу():
    assert cohen_kappa([0, 1, 2, 0], [0, 1, 2, 0]) == pytest.approx(1.0)


def test_согласие_на_уровне_случайного_даёт_ноль():
    """Обе разметки ставят одно и то же значение всем парам.

    Совпадение стопроцентное, но никакой информации в нём нет: каппа
    обязана это увидеть и вернуть ноль, а не единицу.
    """
    assert cohen_kappa([0] * 10, [0] * 10) == pytest.approx(1.0)
    # разметчики независимы и ставят метки в тех же долях
    a = [0, 0, 1, 1]
    b = [1, 1, 0, 0]
    assert cohen_kappa(a, b) < 0


def test_взвешенная_каппа_мягче_к_соседним_оценкам():
    """«2 против 1» — промах меньший, чем «2 против 0»."""
    близко = cohen_kappa([2, 2, 1, 1, 0, 0], [2, 1, 1, 2, 0, 0], weighted=True)
    далеко = cohen_kappa([2, 2, 1, 1, 0, 0], [2, 0, 1, 2, 0, 0], weighted=True)
    assert близко > далеко


def test_невзвешенная_каппа_не_различает_насколько_промах_велик():
    близко = cohen_kappa([2, 2, 1, 1, 0, 0], [2, 1, 1, 2, 0, 0])
    далеко = cohen_kappa([2, 2, 1, 1, 0, 0], [2, 0, 1, 2, 0, 0])
    assert близко == pytest.approx(далеко)


def test_разная_длина_списков_это_ошибка():
    with pytest.raises(ValueError):
        cohen_kappa([0, 1], [0])


def test_словесная_шкала_не_врёт_про_ноль():
    assert kappa_verdict(-0.1) == "хуже случайного"
    assert kappa_verdict(0.85) == "почти полное согласие"


def test_в_выборку_попадают_все_оценки_а_не_только_нули():
    пул = ([{"query_id": f"q{i}", "chunk_id": "c", "sud": 0} for i in range(300)]
           + [{"query_id": f"r{i}", "chunk_id": "c", "sud": 1} for i in range(20)]
           + [{"query_id": f"s{i}", "chunk_id": "c", "sud": 2} for i in range(10)])
    выборка = stratified_sample(пул, "sud", 30)
    assert len(выборка) == 30
    assert {r["sud"] for r in выборка} == {0, 1, 2}


def test_выборка_воспроизводится():
    пул = [{"query_id": f"q{i}", "chunk_id": "c", "sud": i % 3} for i in range(90)]
    assert stratified_sample(пул, "sud", 12) == stratified_sample(пул, "sud", 12)


def test_инструкция_в_документации_совпадает_с_подсказкой_судье():
    guide = (ROOT / "docs" / "ANNOTATION_GUIDE.md").read_text(encoding="utf-8")
    assert RUBRIC in guide, "docs/ANNOTATION_GUIDE.md разошёлся с annotation.RUBRIC"
