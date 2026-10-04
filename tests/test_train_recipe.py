"""Рецепт как набор чисел: проверяется то, что уедет в отчёт и в прогон.

Эти проверки дешёвые, а ловят они дорогое. Этап обучения стоит минуты
видеокарты, но оценка после него — эмбеддинги всего корпуса, 62 594
фрагмента, и на каждый этап это полчаса. Опечатка в конфигурации
обнаруживается не на обучении, а на оценке, когда время уже потрачено.
"""
from __future__ import annotations

import pytest

from rufin.training.config import (
    BY_TAG,
    C_ОБА,
    FIXED,
    ORDER,
    RECIPE,
    STUDENT,
    A,
    C,
    TrainConfig,
    stage_d,
    stage_e,
)


def test_эффективный_батч_везде_один():
    """Этапы сравниваются между собой по dev, и разный размер батча сделал
    бы сравнение сравнением размеров батча."""
    assert {c.effective_batch for c in RECIPE} == {128}


def test_половинной_точности_в_конфигурации_нет_вовсе():
    """На T4 нет bf16 — это Turing, — а fp16 добавляет риск молчаливого
    расхождения ради выигрыша, который на модели в 118M не нужен.
    Поля нет, чтобы его нельзя было включить случайно."""
    поля = set(TrainConfig.__dataclass_fields__)
    assert not поля & {"fp16", "bf16", "half", "dtype", "precision"}


def test_все_этапы_рецепта_перечислены_в_порядке_отчёта():
    assert {c.stage for c in RECIPE} <= set(ORDER)
    assert set(ORDER) == {c.stage for c in RECIPE}


def test_метки_уникальны_и_годятся_в_имя_папки():
    assert len(BY_TAG) == len(RECIPE)
    for метка in BY_TAG:
        assert метка == метка.lower()
        assert "/" not in метка and " " not in метка and "." not in метка


def test_словарь_заморожен_на_каждом_этапе():
    """Иначе обучались бы 118M параметров вместо 22M, и на шести тысячах
    примеров это переписывание представлений слов, которых в выборке нет."""
    assert all(c.freeze_vocabulary for c in RECIPE)


def test_длина_входа_одна_и_та_же():
    """Протокол оценки у всех моделей проекта один: 512 токенов. Обучение
    на другой длине сравнивало бы протоколы, а не модели."""
    assert {c.max_seq_length for c in RECIPE} == {512}


def test_дистилляция_только_там_где_объявлена():
    with_distill = {c.tag for c in RECIPE if "distill" in c.datasets}
    assert with_distill == {"c-kl", "c-oba", "d-matryoshka"} | {c.tag for c in stage_e(C)}


def test_контрастив_и_дистилляция_вместе_только_отдельной_строкой():
    """Библиотека задаёт размер батча один на все наборы обучения, поэтому
    учить их вместе можно только батчем 16. Для контрастива это не дешевле,
    а легче: негативами ему служат эталоны остальных вопросов батча, и 16
    при накоплении 8 — это 16 негативов, а не 128. Значит прирост такого
    этапа нельзя отнести к дистилляции, и этап выделен в свою строку
    с пометкой."""
    оба = [c for c in RECIPE if set(c.datasets) == {"pairs", "distill"}]
    assert [c.tag for c in оба] == ["c-oba"]
    assert "батчем 16" in C_ОБА.note
    assert C.datasets == ("distill",)


def test_этапы_D_и_E_строятся_от_состава_а_не_зашиты():
    """Состав этапа D зависит от того, какой этап оказался принятым,
    и заранее это неизвестно."""
    от_оба = stage_d(C_ОБА)[0]
    assert от_оба.datasets == C_ОБА.datasets
    assert от_оба.extends == "c-oba"
    assert от_оба.tag == stage_d(C)[0].tag  # метка одна: строка таблицы одна
    assert all(c.datasets == C_ОБА.datasets for c in stage_e(C_ОБА))


def test_фиксированные_этапы_не_зависят_от_журнала():
    assert {c.stage for c in FIXED} == {"A", "A+", "B", "C"}
    assert all(not c.matryoshka for c in FIXED)


def test_матрёшка_начинается_с_полной_размерности():
    """Обрезка идёт от полного вектора: 384 — родная размерность e5-small,
    и без неё обученная модель перестала бы быть сравнимой с остальными."""
    for c in RECIPE:
        if c.matryoshka:
            assert c.matryoshka[0] == 384
            assert list(c.matryoshka) == sorted(c.matryoshka, reverse=True)


def test_список_дистилляции_длиннее_двух():
    """На одном документе softmax по списку тождественно равен единице,
    на двух задача сводится к паре. Списочная дистилляция начинается
    с трёх, и в рецепте стоит восемь."""
    for c in RECIPE:
        if "distill" in c.datasets:
            assert c.distill_docs >= 3


def test_перебор_и_конфигурация_переживают_запись_в_журнал():
    """Журнал пишется в JSON, а JSON не знает кортежей: после чтения
    `datasets` и `matryoshka` становятся списками. По ним сравниваются
    составы этапов, а список и кортеж не равны друг другу."""
    import json
    для_D = stage_d(C_ОБА)[0]
    снова = TrainConfig.from_dict(json.loads(json.dumps(для_D.as_dict())))
    assert снова == для_D


def test_сетка_этапа_E_не_разорит_квоту():
    """Четыре точки — это четыре обучения плюс четыре оценки по полчаса.
    Больше dev на 300 вопросах всё равно не различит: его интервал ±0,045."""
    assert len(stage_e(C)) == 4


def test_этап_E_наследует_состав_а_не_веса():
    """«Поверх лучшего» означает поверх состава: обучение всё равно
    начинается с исходных весов, иначе число эпох перестаёт быть числом
    эпох."""
    for c in stage_e(C):
        assert c.datasets == C.datasets
        assert c.with_negatives == C.with_negatives
        assert c.extends == C.tag
        assert c.epochs != C.epochs or c.kl_temperature != C.kl_temperature


@pytest.mark.parametrize("c", RECIPE, ids=[c.tag for c in RECIPE])
def test_мини_батч_не_больше_батча(c):
    """GradCache делит батч на мини-батчи; мини-батч крупнее батча значит,
    что деления нет, а память нужна на весь батч сразу."""
    assert c.mini_batch <= c.batch


def test_ученик_тот_самый():
    assert STUDENT == "intfloat/multilingual-e5-small"


# ---- ужатие под память ----

def test_контрастив_ужимается_мини_батчем_а_не_батчем():
    """Батч у контрастива это число негативов. Уменьшить его значит
    упростить задачу, а не удешевить её. Память там держит мини-батч:
    GradCache делит батч на куски и хранит между проходами только
    эмбеддинги."""
    from rufin.training.trainer import ужать
    меньше = ужать(A)
    assert меньше.batch == A.batch
    assert меньше.mini_batch == A.mini_batch // 2


def test_дистилляция_ужимается_батчем_с_сохранением_эффективного():
    """Связи между примерами нет, потери считаются по каждому вопросу
    отдельно. Значит деление батча с увеличением накопления это
    математически тот же шаг."""
    from rufin.training.trainer import ужать
    меньше = ужать(C)
    assert меньше.effective_batch == C.effective_batch
    assert меньше.batch == C.batch // 2
    assert меньше.accumulate == C.accumulate * 2


def test_ужатие_доходит_до_предела_и_честно_останавливается():
    from rufin.training.trainer import ужать
    ш = C
    while ш is not None:
        прежний, ш = ш, ужать(ш)
        if ш is not None:
            assert ш.effective_batch == C.effective_batch
    assert прежний.batch == 1
