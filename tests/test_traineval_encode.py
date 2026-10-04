"""Кодирование блоками даёт то же, что один вызов, и нормирует вектор.

Этот код исполняется на видеокарте после каждого этапа обучения,
по шестьдесят тысяч фрагментов за раз. Ошибка в сшивке блоков или
в пулинге не падает, а молча портит метрику, и отличить её от плохого
обучения будет нечем.

Настоящая модель тут не нужна: проверяется арифметика, а не веса.
Поэтому сеть подменяется случайной, но детерминированной.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kaggle"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import gpu_traineval as E  # noqa: E402


class Партия(dict):
    """То, что отдаёт настоящий токенизатор: словарь, умеющий .to(устройство)
    и распаковку через **."""

    def to(self, устройство):
        return Партия({k: v.to(устройство) for k, v in self.items()})


class Токенизатор:
    """Отдаёт длину текста токенами и маску с хвостом нулей."""

    def __call__(self, тексты, padding=True, truncation=True, max_length=16,
                 return_tensors="pt"):
        длины = [min(len(t.split()), max_length) for t in тексты]
        ширина = max(длины)
        ids = torch.zeros(len(тексты), ширина, dtype=torch.long)
        маска = torch.zeros(len(тексты), ширина, dtype=torch.long)
        for i, n in enumerate(длины):
            ids[i, :n] = torch.arange(1, n + 1)
            маска[i, :n] = 1
        return Партия({"input_ids": ids, "attention_mask": маска})


class Сеть(torch.nn.Module):
    """Эмбеддинг по номеру токена. Детерминирована и зависит от маски."""

    def __init__(self, dim: int = 8):
        super().__init__()
        torch.manual_seed(13)
        self.emb = torch.nn.Embedding(64, dim)
        self.device = torch.device("cpu")

    def forward(self, input_ids=None, attention_mask=None):
        class Выход:
            pass
        out = Выход()
        out.last_hidden_state = self.emb(input_ids)
        return out


def тексты(n: int) -> list[str]:
    return [" ".join(["сл"] * (3 + i % 7)) for i in range(n)]


@pytest.fixture
def модель():
    return (Токенизатор(), Сеть(), 16)


def test_блоки_сшиваются_в_том_же_порядке(модель):
    данные = тексты(37)
    одним = E.encode(модель, данные, "", batch_size=64, блок=1000)
    блоками = E.encode(модель, данные, "", batch_size=4, блок=8)
    assert одним.shape == блоками.shape == (37, 8)
    assert np.allclose(одним, блоками, atol=1e-5)


def test_векторы_нормированы(модель):
    вектор = E.encode(модель, тексты(20), "", batch_size=8, блок=8)
    assert np.allclose(np.linalg.norm(вектор, axis=1), 1.0, atol=1e-5)


def test_префикс_меняет_вектор(модель):
    """Префиксы e5 обязаны доезжать до модели. Если их потерять, качество
    просядет на ровном месте, а причина будет неочевидна."""
    без = E.encode(модель, тексты(5), "", batch_size=8)
    с_ним = E.encode(модель, тексты(5), "query: ", batch_size=8)
    assert not np.allclose(без, с_ним)


def test_паддинг_не_попадает_в_среднее(модель):
    """Тексты разной длины в одной партии выравниваются нулями. Если взять
    среднее без маски, короткий текст утянет вектор к нулевому токену,
    и зависеть результат начнёт от соседей по партии."""
    пара = ["сл сл сл", " ".join(["сл"] * 9)]
    вместе = E.encode(модель, пара, "", batch_size=2)
    порознь = np.vstack([E.encode(модель, [t], "", batch_size=1) for t in пара])
    assert np.allclose(вместе, порознь, atol=1e-5)
