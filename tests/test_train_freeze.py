"""Заморожена должна быть ровно матрица словаря, и ничто больше.

Цифра «118M параметров, 22M обучаемых» идёт в отчёт и в карточку модели,
поэтому она проверяется, а не берётся из статьи. Инвентарь параметров
собирается здесь по описанию архитектуры, а не читается у настоящей
модели: на маке модель не разворачивается.

**Инвентарь сверен с настоящей моделью на видеокарте 04.10.2026** и сходится
до единицы: 117 653 760 всего, 96 014 208 заморожено, 21 639 552 обучаемых.
Поэтому числа ниже проверяются точным равенством, а не округлением.
Прежний инвентарь не содержал пулера и расходился на 147 840 параметров;
это и видно было бы только на модели.

Главная ловушка не в том, забыли ли заморозить, а в том, **что именно
заморожено**. В слое эмбеддингов три матрицы: словарь, позиции и тип
отрезка. Заморозка по подстроке «embeddings» выключила бы заодно
позиционные представления и нормировку. Обучение при этом пойдёт,
лосс упадёт, а прирост окажется меньше без видимой причины.
"""
from __future__ import annotations

import pytest

from rufin.training.freeze import apply_freeze, is_vocabulary, plan_freeze

VOCAB = 250037
HIDDEN = 384
LAYERS = 12
INTERMEDIATE = 1536
POSITIONS = 512


def инвентарь() -> list[tuple[str, tuple[int, ...]]]:
    """Имена и формы параметров multilingual-e5-small, как их видит
    SentenceTransformer: с приставкой модуля."""
    p = "0.auto_model."
    out: list[tuple[str, tuple[int, ...]]] = [
        (p + "embeddings.word_embeddings.weight", (VOCAB, HIDDEN)),
        (p + "embeddings.position_embeddings.weight", (POSITIONS, HIDDEN)),
        (p + "embeddings.token_type_embeddings.weight", (2, HIDDEN)),
        (p + "embeddings.LayerNorm.weight", (HIDDEN,)),
        (p + "embeddings.LayerNorm.bias", (HIDDEN,)),
    ]
    for i in range(LAYERS):
        слой = f"{p}encoder.layer.{i}."
        for имя in ("attention.self.query", "attention.self.key", "attention.self.value",
                    "attention.output.dense"):
            out += [(слой + имя + ".weight", (HIDDEN, HIDDEN)),
                    (слой + имя + ".bias", (HIDDEN,))]
        out += [(слой + "attention.output.LayerNorm.weight", (HIDDEN,)),
                (слой + "attention.output.LayerNorm.bias", (HIDDEN,)),
                (слой + "intermediate.dense.weight", (INTERMEDIATE, HIDDEN)),
                (слой + "intermediate.dense.bias", (INTERMEDIATE,)),
                (слой + "output.dense.weight", (HIDDEN, INTERMEDIATE)),
                (слой + "output.dense.bias", (HIDDEN,)),
                (слой + "output.LayerNorm.weight", (HIDDEN,)),
                (слой + "output.LayerNorm.bias", (HIDDEN,))]
    # Пулер. Для поиска он не используется, но в модели есть и в счёт
    # параметров входит.
    out += [(p + "pooler.dense.weight", (HIDDEN, HIDDEN)),
            (p + "pooler.dense.bias", (HIDDEN,))]
    return out


def test_заморожена_ровно_матрица_словаря():
    plan = plan_freeze(инвентарь())
    assert plan.frozen == ("0.auto_model.embeddings.word_embeddings.weight",)
    assert plan.frozen_params == VOCAB * HIDDEN


def test_приставка_имени_может_смениться_и_это_не_ломает_заморозку():
    """На видеокарте параметр назвался `0.model.embeddings.word_embeddings
    .weight`, а не `0.auto_model...`: библиотека переименовала приставку.
    Сверка идёт по окончанию имени, поэтому заморозка устояла. Проверка
    на точное равенство имени сломалась бы молча, и обучались бы все 118M."""
    plan = plan_freeze([("0.model.embeddings.word_embeddings.weight", (VOCAB, HIDDEN)),
                        ("0.model.encoder.layer.0.output.dense.weight", (HIDDEN, 1536))])
    assert plan.frozen == ("0.model.embeddings.word_embeddings.weight",)


@pytest.mark.parametrize("имя", [
    "0.auto_model.embeddings.position_embeddings.weight",
    "0.auto_model.embeddings.token_type_embeddings.weight",
    "0.auto_model.embeddings.LayerNorm.weight",
])
def test_прочие_эмбеддинги_обучаются(имя):
    """Их заморозка — самая правдоподобная тихая ошибка: имя похоже."""
    assert not is_vocabulary(имя)
    plan = plan_freeze(инвентарь())
    assert имя not in plan.frozen


def test_цифры_отчёта_сходятся_с_настоящей_моделью():
    """Числа сверены с моделью на видеокарте и совпадают до единицы."""
    plan = plan_freeze(инвентарь())
    assert plan.total_params == 117_653_760
    assert plan.frozen_params == 96_014_208
    assert plan.trainable_params == 21_639_552
    # доля обучаемых меньше одной пятой: именно это делает обучение дешёвым
    assert round(plan.trainable_params / plan.total_params, 3) == 0.184


def test_без_заморозки_обучается_всё():
    plan = plan_freeze(инвентарь(), freeze_vocabulary=False)
    assert plan.frozen == ()
    assert plan.frozen_params == 0
    assert plan.trainable_params == plan_freeze(инвентарь()).total_params


class ПоддельнаяМодель:
    """Минимум, который нужен `apply_freeze`: имена, формы и requires_grad."""

    class Параметр:
        def __init__(self, shape):
            self.shape = shape
            self.requires_grad = True

    def __init__(self, имена):
        self._p = [(имя, self.Параметр(форма)) for имя, форма in имена]

    def named_parameters(self):
        return list(self._p)


def test_выполнение_плана_снимает_градиент_только_со_словаря():
    model = ПоддельнаяМодель(инвентарь())
    plan = apply_freeze(model)
    без_градиента = [имя for имя, p in model.named_parameters() if not p.requires_grad]
    assert без_градиента == list(plan.frozen)
    assert len(без_градиента) == 1


def test_отказ_если_матрицы_словаря_нет():
    """Имя параметра в другой версии библиотеки может оказаться другим.
    Тогда обучение пошло бы по всем 118M, и молча: ошибки нет, просто
    дороже и хуже. Поэтому отказ."""
    model = ПоддельнаяМодель([("encoder.layer.0.output.dense.weight", (384, 1536))])
    with pytest.raises(RuntimeError, match="матрица словаря не найдена"):
        apply_freeze(model)
