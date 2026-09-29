"""Нарезка актов на фрагменты.

Главное, что здесь проверяется, — жёсткость границы. Эмбеддер обрезает вход
на 512 токенах молча: чанк, не влезший в окно, индексируется наполовину,
и заметить это по метрикам нельзя, потому что они просто окажутся ниже.
"""
from __future__ import annotations

import pytest

from rufin.chunk_configs import BASE, BY_NAME, GRID
from rufin.chunking import TokenRuler, chunk_act, parse_units


@pytest.fixture
def ruler() -> TokenRuler:
    """Счётчик без токенизатора: длина оценивается по символам.

    Для тестов этого достаточно и даже лучше — результат не зависит от того,
    какая версия tokenizers стоит на машине. Ровно на этом расхождении
    проект один раз уже потерял соответствие эталонов фрагментам.
    """
    return TokenRuler(tokenizer=None)


# ---------- разбор структуры ----------

def test_пункты_опознаются_по_номеру(act, ruler):
    units = parse_units(act["text"])
    assert [u.number for u in units] == ["", "1.1", "1.2", ""]


def test_пункт_забирает_свои_продолжения(act):
    units = {u.number: u for u in parse_units(act["text"])}
    assert "комплексного анализа" in units["1.1"].text


def test_раздел_запоминается_и_переносится_на_пункты(act):
    units = parse_units(act["text"])
    assert units[1].section == "Глава 1. Общие положения"
    assert units[2].section == "Глава 1. Общие положения"
    assert units[3].section == "Приложение 1"


def test_текст_до_первого_пункта_становится_преамбулой(act):
    first = parse_units(act["text"])[0]
    assert first.number == ""
    assert first.text.startswith("Настоящее Положение")


def test_строка_оглавления_не_открывает_раздел():
    # «Приложение 3 . . . . . . 47» в оглавлении выглядит как заголовок
    # раздела. Подхвати его — и все следующие пункты уедут в чужой раздел.
    text = "1.1. Первый пункт текста акта.\nПриложение 3 . . . . . . . 47\n1.2. Второй пункт."
    assert all(u.section == "" for u in parse_units(text))


def test_пустые_единицы_выбрасываются():
    assert parse_units("\n\n   \n\n") == []


# ---------- нарезка ----------

def test_чанк_никогда_не_длиннее_размера(act, ruler):
    for size in (128, 256, 512):
        for chunk in chunk_act(act, ruler, size=size):
            assert ruler.count(chunk.text) <= size, f"{chunk.chunk_id} при size={size}"


def test_нарезка_по_длине_тоже_держит_границу(act, ruler):
    for chunk in chunk_act(act, ruler, size=128, strategy="length"):
        assert ruler.count(chunk.text) <= 128


def test_идентификаторы_последовательны_и_уникальны(act, ruler):
    chunks = chunk_act(act, ruler, size=128)
    assert [c.chunk_id for c in chunks] == [f"590-П_28062017#{i:04d}" for i in range(len(chunks))]
    assert [c.position for c in chunks] == list(range(len(chunks)))


def test_смещения_в_акте_не_убывают(act, ruler):
    starts = [c.char_start for c in chunk_act(act, ruler, size=128)]
    assert starts == sorted(starts)


def test_шапка_называет_акт(act, ruler):
    head = chunk_act(act, ruler, size=512)[0].text
    assert head.startswith("Положение Банка России № 590-П от 28.06.2017")


def test_шапка_не_повторяет_вид_документа_дважды(act, ruler):
    # в перечне название уже начинается с «Положение Банка России…»,
    # и без вычистки шапка получалась бы «Положение … Положение …»
    head = chunk_act(act, ruler, size=512)[0].text.split("\n\n")[0]
    assert head.count("Положение Банка России") == 1


def test_без_шапки_чанк_начинается_с_текста(act, ruler):
    first = chunk_act(act, ruler, size=512, add_heading=False)[0].text
    assert first.startswith("Настоящее Положение")


def test_шапка_режется_но_номер_и_дата_остаются(act, ruler):
    # у актов с названием в полторы тысячи знаков шапка съедала весь чанк
    длинный = dict(act, title="Положение Банка России «" + "о порядке " * 200 + "»")
    for chunk in chunk_act(длинный, ruler, size=256):
        assert "№ 590-П от 28.06.2017" in chunk.text
        assert ruler.count(chunk.text) <= 256


def test_раздел_попадает_в_шапку(act, ruler):
    chunks = chunk_act(act, ruler, size=128)
    assert any("Глава 1" in c.text for c in chunks)


def test_граница_чанка_проходит_по_границе_раздела(act, ruler):
    # пункты из разных разделов не должны оказаться в одном чанке:
    # иначе шапка чанка врёт про то, откуда взят текст
    for c in chunk_act(act, ruler, size=512):
        assert c.section in ("", "Глава 1. Общие положения", "Приложение 1")


def test_длинный_пункт_режется_а_не_теряется(ruler):
    act = {
        "act_id": "1-У_01012020", "number": "1-У", "date": "01.01.2020",
        "kind": "указание", "type": "Указание", "title": "О чем-то",
        "issue": "1", "issue_url": "u",
        "text": "1.1. " + "слово " * 400,
    }
    chunks = chunk_act(act, ruler, size=128)
    assert len(chunks) > 1
    assert all(ruler.count(c.text) <= 128 for c in chunks)


def test_пустой_акт_дает_пустую_нарезку(ruler):
    act = {
        "act_id": "0", "number": "0-У", "date": "01.01.2020", "kind": "указание",
        "type": "Указание", "title": "", "issue": "1", "issue_url": "u", "text": "",
    }
    assert chunk_act(act, ruler, size=128) == []


# ---------- сетка конфигураций ----------

def test_конфигурации_названы_уникально():
    assert len({c.name for c in GRID}) == len(GRID)
    assert BY_NAME.keys() == {c.name for c in GRID}


def test_каждая_конфигурация_отличается_от_базовой_ровно_одним_параметром():
    # иначе по результату абляции нельзя сказать, что именно повлияло
    поля = ("strategy", "size", "overlap", "heading")
    for c in GRID:
        if c.name == BASE.name:
            continue
        разошлись = [f for f in поля if getattr(c, f) != getattr(BASE, f)]
        assert len(разошлись) == 1, f"{c.name}: разошлись {разошлись}"
