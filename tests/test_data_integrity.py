"""Проверки собранных данных, а не кода.

Эти ошибки не падают и не видны по метрикам: разметка, указывающая на
несуществующий фрагмент, просто делает запрос ненаходимым, а выдача
с повторами завышает полноту. Один раз проект уже терял соответствие
эталонов фрагментам из-за разных версий токенизатора — и заметить это
удалось только отдельной сверкой.

Тесты помечены `data` и пропускаются, если данные не собраны.
"""
from __future__ import annotations

import json
import pathlib

import pytest

pytestmark = pytest.mark.data


def _jsonl(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _qrels(path: pathlib.Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8") as f:
        next(f)
        for line in f:
            qid, cid, score = line.rstrip("\n").split("\t")
            out.setdefault(qid, {})[cid] = int(score)
    return out


@pytest.fixture
def запросы(data_dir) -> list[dict]:
    return _jsonl(data_dir / "queries" / "queries.jsonl")


@pytest.fixture
def разметка(data_dir) -> dict[str, dict[str, int]]:
    path = data_dir / "queries" / "qrels.tsv"
    if not path.exists():
        pytest.skip("разметки ещё нет")
    return _qrels(path)


@pytest.fixture
def чанки(data_dir) -> set[str]:
    path = data_dir / "chunks" / "base.jsonl"
    if not path.exists():
        pytest.skip("нарезка не установлена: `make use-chunks`")
    return {json.loads(line)["chunk_id"] for line in path.open(encoding="utf-8")}


def test_идентификаторы_запросов_уникальны(запросы):
    ids = [q["query_id"] for q in запросы]
    assert len(ids) == len(set(ids))


def test_у_каждого_запроса_есть_текст(запросы):
    assert all(q["text"].strip() for q in запросы)


def test_у_каждого_запроса_указано_происхождение(запросы):
    # без происхождения нельзя разложить метрики по подвыборкам, а общая
    # цифра по смеси синтетики и живых вопросов ничего не значит
    assert all(q.get("origin") for q in запросы)


def test_размеченные_запросы_есть_в_наборе(запросы, разметка):
    известные = {q["query_id"] for q in запросы}
    assert set(разметка) <= известные


def test_эталоны_указывают_на_существующие_фрагменты(разметка, чанки):
    # ровно это разошлось из-за разных версий токенизатора: имена совпали,
    # а текст под ними стал другим
    эталоны = {c for d in разметка.values() for c in d}
    assert эталоны <= чанки


def test_оценки_из_допустимого_набора(разметка):
    оценки = {v for d in разметка.values() for v in d.values()}
    assert оценки <= {1, 2}, f"в разметке есть посторонние оценки: {оценки - {1, 2}}"


def test_у_каждого_размеченного_запроса_есть_прямой_ответ(разметка):
    # оценка 1 без единой 2 означает, что прямого ответа в корпусе нет;
    # такой запрос меряет не поиск, а полноту корпуса
    без_ответа = [q for q, d in разметка.items() if 2 not in d.values()]
    assert not без_ответа, f"запросы без оценки 2: {без_ответа[:5]}"


@pytest.mark.parametrize("конфиг", ["bm25", "dense-bge-m3", "hybrid", "hybrid-rerank"])
def test_выдачи_целы(data_dir, запросы, чанки, конфиг):
    path = data_dir / "runs" / f"base__{конфиг}.jsonl"
    if not path.exists():
        pytest.skip(f"выдачи {конфиг} нет: прогон на Kaggle не сделан")
    строки = _jsonl(path)
    ids = [r["query_id"] for r in строки]
    assert len(ids) == len(set(ids)), "запрос встречается в выдаче дважды"
    for r in строки:
        assert r["ranked"], f"{r['query_id']}: пустая выдача"
        assert len(r["ranked"]) == len(set(r["ranked"])), f"{r['query_id']}: повтор в выдаче"
        assert set(r["ranked"]) <= чанки, f"{r['query_id']}: фрагмент вне корпуса"


@pytest.mark.parametrize("конфиг", ["bm25", "dense-bge-m3", "hybrid", "hybrid-rerank"])
def test_выдача_покрывает_нынешний_набор_запросов(data_dir, запросы, конфиг):
    """Выдача считалась по тому набору запросов, который есть сейчас.

    Проверка не строгая, а сообщающая: набор запросов растёт локально,
    а выдачи считаются на Kaggle, и между двумя этими событиями выдача
    неизбежно отстаёт. Падать на этом нельзя — отставание не ошибка,
    а состояние работы. Но и молчать нельзя: запрос без выдачи в метрику
    не входит вовсе, и по метрикам этого не видно, они просто считаются
    по меньшему числу запросов.
    """
    path = data_dir / "runs" / f"base__{конфиг}.jsonl"
    if not path.exists():
        pytest.skip(f"выдачи {конфиг} нет: прогон на Kaggle не сделан")
    ids = {r["query_id"] for r in _jsonl(path)}
    набор = {q["query_id"] for q in запросы}
    нет_выдачи = набор - ids
    if нет_выдачи:
        pytest.skip(f"выдача {конфиг} отстала: без неё {len(нет_выдачи)} запросов "
                    f"из {len(набор)}. Нужен новый прогон этапа B.")
    assert not нет_выдачи
