"""Дообучение ретривера: рецепт, данные обучения, журнал решений.

Почему пакет живёт в `src/rufin`, хотя обучение идёт только на Kaggle.
Тяжёлые зависимости (torch, sentence-transformers) импортируются внутри
функций, как в `rufin.retrieval.dense`, поэтому сборка пар, отсечка
негативов, план заморозки и правило приёмки этапа проверяются тестами
на маке, где ни одна модель не разворачивается. На видеокарту уезжает
тот же код — расходиться ему неоткуда.

Что где лежит:

* `config`   — рецепт: один этап, один объект, все числа в одном месте;
* `pairs`    — пары «вопрос — эталонный фрагмент» и проверки на утечку;
* `negatives`— трудные негативы и отсечка по оценке учителя;
* `freeze`   — что заморожено и сколько параметров обучается;
* `losses`   — списочная KL-дистилляция из кросс-энкодера;
* `trainer`  — сам прогон обучения, единственный модуль с torch;
* `journal`  — таблица «этап рецепта → dev» с вердиктом по каждому этапу.
"""
from __future__ import annotations

from .config import BY_TAG, RECIPE, TrainConfig
from .freeze import FreezePlan, plan_freeze
from .journal import Entry, Journal, verdict
from .negatives import NegativeRules, pick_negatives, probability
from .pairs import TrainPair, acts_of, build_pairs, check_group, leaked_acts

__all__ = [
    "BY_TAG", "RECIPE", "TrainConfig",
    "FreezePlan", "plan_freeze",
    "Entry", "Journal", "verdict",
    "NegativeRules", "pick_negatives", "probability",
    "TrainPair", "acts_of", "build_pairs", "check_group", "leaked_acts",
]
