"""Что заморожено на обучении и сколько параметров остаётся обучаемых.

У `multilingual-e5-small` 118M параметров, и из них 96M — матрица словаря
XLM-R: 250 002 токена на 384 измерения. Обучать её на шести тысячах
примеров значит переписывать представления слов, большинство из которых
в обучающей выборке не встретится ни разу. Поэтому матрица словаря
замораживается, и обучается около 22M.

Это же делает обучение дешёвым: градиент и состояния оптимизатора
для 96M параметров не хранятся вовсе.

**Проверять надо не факт заморозки, а её адрес.** В слое эмбеддингов
XLM-R три матрицы: словарь (`word_embeddings`), позиции
(`position_embeddings`) и тип отрезка (`token_type_embeddings`).
Заморозить по подстроке «embeddings» значит заморозить все три плюс
нормировку, то есть заодно выключить обучение позиционных представлений.
Ошибка тихая: обучение пойдёт, лосс упадёт, а прирост окажется меньше
без всякой видимой причины. Отсюда точное окончание имени и тест на то,
что позиции и нормировка остались обучаемыми.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import prod

# Точное окончание имени параметра. В `SentenceTransformer` имя приезжает
# с приставкой модуля («0.auto_model.embeddings.word_embeddings.weight»),
# поэтому сверяется окончание, а не равенство.
VOCABULARY = "embeddings.word_embeddings.weight"


def is_vocabulary(name: str) -> bool:
    """Это матрица словаря, а не позиции и не тип отрезка."""
    return name.endswith(VOCABULARY)


@dataclass(frozen=True)
class FreezePlan:
    frozen: tuple[str, ...]
    trainable_params: int
    frozen_params: int

    @property
    def total_params(self) -> int:
        return self.trainable_params + self.frozen_params

    def as_dict(self) -> dict:
        return {"всего параметров": self.total_params,
                "обучаемых": self.trainable_params,
                "заморожено": self.frozen_params,
                "заморожено имён": list(self.frozen),
                "доля обучаемых": round(self.trainable_params / self.total_params, 3)
                if self.total_params else 0.0}

    def __str__(self) -> str:
        return (f"{self.total_params / 1e6:.0f}M параметров, "
                f"{self.trainable_params / 1e6:.0f}M обучаемых "
                f"({100 * self.trainable_params / self.total_params:.0f}%)")


def plan_freeze(named_shapes: Iterable[tuple[str, tuple[int, ...]]],
                freeze_vocabulary: bool = True) -> FreezePlan:
    """План заморозки по именам и формам параметров.

    Отдельно от самой модели нарочно: так план проверяется тестом на маке,
    где ни одна модель не разворачивается, а на видеокарте исполняется
    ровно он.
    """
    frozen: list[str] = []
    frozen_params = 0
    trainable_params = 0
    for name, shape in named_shapes:
        n = prod(shape) if shape else 1
        if freeze_vocabulary and is_vocabulary(name):
            frozen.append(name)
            frozen_params += n
        else:
            trainable_params += n
    return FreezePlan(frozen=tuple(frozen), trainable_params=trainable_params,
                      frozen_params=frozen_params)


def apply_freeze(model, freeze_vocabulary: bool = True) -> FreezePlan:
    """Выполнить план на настоящей модели и вернуть его же для отчёта.

    Если заморозить было нечего, это ошибка, а не мелочь: значит имя
    параметра в этой версии библиотеки другое, и обучение пошло бы
    по 118M параметрам вместо 22M. Молча оно не упадёт — просто станет
    дороже и хуже.
    """
    shapes = [(name, tuple(p.shape)) for name, p in model.named_parameters()]
    plan = plan_freeze(shapes, freeze_vocabulary=freeze_vocabulary)
    if freeze_vocabulary and not plan.frozen:
        raise RuntimeError(
            "матрица словаря не найдена среди параметров: ожидалось имя, "
            f"кончающееся на «{VOCABULARY}». Обучение пошло бы по всем "
            f"параметрам. Имена, которые есть: "
            f"{[n for n, _ in shapes[:5]]} ...")
    замороженные = set(plan.frozen)
    for name, p in model.named_parameters():
        p.requires_grad = name not in замороженные
    return plan
