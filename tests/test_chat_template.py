"""Сборка запроса переживает шаблон без системной роли.

Шаблоны семейств несовместимы, и это не оформление: Qwen принимает
системную роль, а Mistral-7B-Instruct требует строгого чередования
user/assistant и отвечает `TemplateError`. Модель другого семейства
для теста обязательна — иначе тест померит, насколько ученик выучил
стиль своего же генератора, — поэтому подстраивается код, а не выбор
модели. Прогон на Kaggle уже потерян на этом один раз.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kaggle"))

import gpu_gen as G  # noqa: E402


class ТерпимыйШаблон:
    """Принимает любые роли — так ведёт себя Qwen."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "|".join(f"{m['role']}:{m['content']}" for m in messages)


class СтрогийШаблон:
    """Отвергает системную роль — так ведёт себя Mistral-7B-Instruct."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        if any(m["role"] == "system" for m in messages):
            raise ValueError("Conversation roles must alternate user/assistant/...")
        return "|".join(f"{m['role']}:{m['content']}" for m in messages)


def test_системная_роль_используется_если_шаблон_её_принимает():
    готово = G._chat(ТерпимыйШаблон(), "задание")
    assert готово.startswith("system:")
    assert "user:задание" in готово


def test_шаблон_без_системной_роли_не_роняет_генерацию():
    готово = G._chat(СтрогийШаблон(), "задание")
    assert "system:" not in готово
    assert готово.startswith("user:")


def test_наставление_не_теряется_в_запасном_пути():
    """Выбросить SYSTEM — значит менять задачу: вопросы двух моделей
    стали бы несопоставимы, а тест на том и держится, что они сравнимы."""
    готово = G._chat(СтрогийШаблон(), "задание")
    assert G.SYSTEM in готово
    assert "задание" in готово


@pytest.mark.parametrize("шаблон", [ТерпимыйШаблон(), СтрогийШаблон()])
def test_оба_шаблона_доносят_само_задание(шаблон):
    assert "задание" in G._chat(шаблон, "задание")
