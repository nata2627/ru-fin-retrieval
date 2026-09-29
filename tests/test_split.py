"""Проверки сплита по актам.

Утечка через сплит — ошибка, которая не падает и не видна по метрикам:
качество просто оказывается выше, чем есть, и объяснить это будет нечем.
Поэтому проверок четыре, и три из них — про данные, а не про код.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from rufin import split as S

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPLIT = ROOT / "data" / "queries" / "split.json"


@pytest.fixture
def сплит() -> dict:
    if not SPLIT.exists():
        pytest.skip("сплита нет: `make split`")
    return S.load(SPLIT)


def test_группы_не_пересекаются(сплит):
    assert S.overlaps(сплит) == []


def test_каждый_акт_ровно_в_одной_группе(сплит):
    из_групп = [a for acts in сплит["gruppy"].values() for a in acts]
    assert len(из_групп) == len(set(из_групп))
    assert set(сплит["gruppy"]) == set(S.GROUPS)


@pytest.mark.data
def test_акты_с_эталонами_теста_не_попали_в_обучение():
    """Эталон живого вопроса не может лежать в обучающем акте.

    Иначе прирост от обучения будет отчасти узнаванием того же текста,
    и отделить одно от другого будет нечем.
    """
    if not SPLIT.exists():
        pytest.skip("сплита нет")
    сплит = S.load(SPLIT)
    train = set(сплит["gruppy"][S.TRAIN])
    нарушения = []
    for имя in ("qrels.tsv", "qrels_by_clause.tsv"):
        path = ROOT / "data" / "queries" / имя
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            next(f)
            for line in f:
                qid, cid = line.split("\t")[:2]
                if qid.startswith(("exp", "man")) and S.act_of_chunk(cid) in train:
                    нарушения.append((qid, cid))
    assert not нарушения, f"эталоны теста в обучающих актах: {нарушения[:5]}"


@pytest.mark.data
def test_класс_документов_отложен_целиком():
    """Все Инструкции вне обучения — иначе перенос на новый жанр не измерить."""
    if not SPLIT.exists():
        pytest.skip("сплита нет")
    сплит = S.load(SPLIT)
    corpus = ROOT / "data" / "corpus" / "acts.jsonl"
    if not corpus.exists():
        pytest.skip("корпус не собран")
    train = set(сплит["gruppy"][S.TRAIN])
    with corpus.open(encoding="utf-8") as f:
        инструкции = {json.loads(line)["act_id"] for line in f
                      if line.strip() and json.loads(line)["type"] == "И"}
    assert not (инструкции & train)


@pytest.mark.data
def test_объединение_групп_совпадает_с_корпусом():
    if not SPLIT.exists():
        pytest.skip("сплита нет")
    сплит = S.load(SPLIT)
    corpus = ROOT / "data" / "corpus" / "acts.jsonl"
    if not corpus.exists():
        pytest.skip("корпус не собран")
    with corpus.open(encoding="utf-8") as f:
        все = {json.loads(line)["act_id"] for line in f if line.strip()}
    из_групп = {a for acts in сплит["gruppy"].values() for a in acts}
    assert из_групп == все


def test_акт_фрагмента_отрезается_по_решётке():
    assert S.act_of_chunk("590-П_28062017#0034") == "590-П_28062017"
