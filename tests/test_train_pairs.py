"""Сборка обучающих пар и утечка по актам.

Два вида проверок. Первые работают на выдуманных данных и ловят ошибки
кода. Вторые помечены `data` и работают на настоящей обучающей выборке:
они ловят то, что кодом не ловится вовсе — съехавший эталон и вопрос,
попавший на отложенный акт.

Почему утечка проверяется по актам, а не по вопросам. Фрагменты одного
акта нарезаны с перекрытием 15%, то есть делят друг с другом текст.
Обучающий вопрос по акту из dev означает, что dev меряет узнавание
знакомого текста, а не поиск, и прирост по нему ничего не значит.
Разные вопросы по одному акту при этом совершенно законны — их
в выборке по нескольку на акт.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from rufin import split as S
from rufin.training.pairs import (
    acts_of,
    as_dataset,
    assert_only_train_acts,
    build_pairs,
    check_group,
    leaked_acts,
    read_jsonl,
    train_acts,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
QDIR = ROOT / "data" / "queries"
CHUNKS = ROOT / "data" / "chunks" / "base.jsonl"


# ---- на выдуманных данных ----

ТЕКСТЫ = {"590-П_28062017#0001": "Кредитная организация оценивает кредитный риск.",
          "590-П_28062017#0002": "Профессиональное суждение выносится по анализу.",
          "447-П_22122014#0005": "Куратор страховой организации назначается на срок."}

СПЛИТ = {"gruppy": {S.TRAIN: ["590-П_28062017"], S.DEV: ["447-П_22122014"],
                    S.TEST_LIVE: [], S.TEST_KIND: [], S.TEST_UNSEEN: []}}


def вопрос(qid: str, gold: str, text: str = "Как оценивается риск?") -> dict:
    return {"query_id": qid, "text": text, "gold_chunk_id": gold}


def test_акт_берётся_из_эталона_а_не_из_поля():
    """Поле `act_id` могло не доехать или разойтись с эталоном, а учится
    модель на эталоне."""
    q = dict(вопрос("tr1", "590-П_28062017#0001"), act_id="совсем-другой-акт")
    assert acts_of([q]) == {"590-П_28062017"}


def test_префиксы_ставятся_из_одного_источника():
    """«query: » и «passage: » обязаны совпасть с теми, что ставятся при
    построении индекса и при поиске. Отличие испортит качество, а причина
    будет неочевидна."""
    pairs, _ = build_pairs([вопрос("tr1", "590-П_28062017#0001")], ТЕКСТЫ)
    assert pairs[0].anchor.startswith("query: ")
    assert pairs[0].positive.startswith("passage: ")


def test_вопрос_без_эталона_в_нарезке_выбрасывается():
    """Идентификатор фрагмента позиционный: при сдвиге границ нарезки он
    не исчезает, а начинает указывать на другой текст. Отсутствие — тот
    случай, когда сдвиг виден, и молча подставлять пустоту нельзя."""
    pairs, stats = build_pairs([вопрос("tr1", "999-П_01012000#0001")], ТЕКСТЫ)
    assert pairs == []
    assert stats["эталона нет в нарезке"] == 1


def test_повтор_вопроса_не_удваивает_пример():
    q = вопрос("tr1", "590-П_28062017#0001")
    pairs, stats = build_pairs([q, q], ТЕКСТЫ)
    assert len(pairs) == 1
    assert stats["повтор вопроса"] == 1


def test_утечка_по_актам_видна():
    train = [вопрос("tr1", "590-П_28062017#0001")]
    dev_чистый = [вопрос("dv1", "447-П_22122014#0005")]
    dev_грязный = [вопрос("dv1", "590-П_28062017#0002")]
    assert leaked_acts(train, dev_чистый) == set()
    assert leaked_acts(train, dev_грязный) == {"590-П_28062017"}


def test_разные_вопросы_по_одному_акту_утечкой_не_считаются():
    train = [вопрос("tr1", "590-П_28062017#0001"), вопрос("tr2", "590-П_28062017#0002")]
    assert leaked_acts(train, [вопрос("dv1", "447-П_22122014#0005")]) == set()


def test_группа_сплита_сверяется():
    bad = check_group([вопрос("tr1", "447-П_22122014#0005")], СПЛИТ, S.TRAIN)
    assert bad == {"tr1": S.DEV}
    assert check_group([вопрос("tr1", "590-П_28062017#0001")], СПЛИТ, S.TRAIN) == {}


def test_эталон_вне_сплита_тоже_нарушение():
    bad = check_group([вопрос("tr1", "999-П_01012000#0001")], СПЛИТ, S.TRAIN)
    assert bad == {"tr1": "нет в сплите"}


def test_кандидаты_из_отложенных_актов_это_отказ():
    assert train_acts(СПЛИТ) == frozenset({"590-П_28062017"})
    assert_only_train_acts(["590-П_28062017#0002"], СПЛИТ)
    with pytest.raises(ValueError, match="не из обучающей группы"):
        assert_only_train_acts(["447-П_22122014#0005"], СПЛИТ)


def test_строка_с_недобором_негативов_в_набор_не_идёт():
    """Столбцов обязано быть одинаковое число во всех строках, иначе
    `datasets` не соберёт таблицу. Молча добить пустой строкой значило бы
    учить модель отталкивать пустоту."""
    pairs, _ = build_pairs([вопрос("tr1", "590-П_28062017#0001"),
                            вопрос("tr2", "447-П_22122014#0005")], ТЕКСТЫ)
    cols = as_dataset(pairs, {"tr1": ["590-П_28062017#0002"], "tr2": []},
                      ТЕКСТЫ, per_query=2)
    assert cols["anchor"] == []
    cols = as_dataset(pairs, {"tr1": ["590-П_28062017#0002", "447-П_22122014#0005"]},
                      ТЕКСТЫ, per_query=2)
    assert len(cols["anchor"]) == 1
    assert len(cols["negative_1"]) == len(cols["negative_2"]) == 1
    assert cols["negative_1"][0].startswith("passage: ")


# ---- на настоящих данных ----

@pytest.fixture
def обучающая():
    if not (QDIR / "synthetic_train.jsonl").exists():
        pytest.skip("обучающей выборки нет: её считает этап C на Kaggle")
    return read_jsonl(str(QDIR / "synthetic_train.jsonl"))


@pytest.fixture
def отложенные():
    пути = {"dev": QDIR / "synthetic_dev.jsonl", "тест": QDIR / "synthetic_test.jsonl"}
    if not all(p.exists() for p in пути.values()):
        pytest.skip("отложенных наборов нет")
    return {имя: read_jsonl(str(p)) for имя, p in пути.items()}


@pytest.mark.data
def test_dev_и_тест_не_пересекаются_с_обучением_по_актам(обучающая, отложенные):
    """Главная проверка всего этапа. Пересечение означает, что dev меряет
    узнавание, и прирост рецепта по нему ничего не значит."""
    for имя, набор in отложенные.items():
        утечка = leaked_acts(обучающая, набор)
        assert утечка == set(), f"{имя}: {len(утечка)} общих актов, например {sorted(утечка)[:3]}"


@pytest.mark.data
def test_каждый_вопрос_лежит_в_своей_группе_сплита(обучающая, отложенные):
    if not (QDIR / "split.json").exists():
        pytest.skip("сплита нет")
    сплит = S.load(str(QDIR / "split.json"))
    assert check_group(обучающая, сплит, S.TRAIN) == {}
    assert check_group(отложенные["dev"], сплит, S.DEV) == {}
    # тест собран из двух групп сразу, поэтому проверяется принадлежность
    # хотя бы к одной из них, а не к одной конкретной
    чужие = {qid: группа
             for qid, группа in check_group(отложенные["тест"], сплит, S.TEST_UNSEEN).items()
             if группа != S.TEST_KIND}
    assert чужие == {}


@pytest.mark.data
def test_пары_собираются_без_потерь(обучающая):
    """Ни один эталон не должен потеряться: потеря означает, что нарезка
    разошлась с той, по которой генерировались вопросы."""
    if not CHUNKS.exists():
        pytest.skip("нарезки нет")
    тексты = {}
    with open(CHUNKS, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            тексты[c["chunk_id"]] = c["text"]
    pairs, stats = build_pairs(обучающая, тексты)
    assert stats["эталона нет в нарезке"] == 0, "нарезка разошлась с обучающей выборкой"
    assert stats["пустой текст эталона"] == 0
    assert len(pairs) == len(обучающая)
