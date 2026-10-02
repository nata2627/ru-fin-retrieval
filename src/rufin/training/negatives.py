"""Трудные негативы: отбор и отсечка по оценке учителя.

Трудный негатив — это фрагмент, который поиск поставил высоко, а правильным
ответом он не является. Такие примеры учат модель различать похожее, и это
дороже, чем отличать правильный ответ от случайного фрагмента корпуса.

**Главная опасность в том, что негатив может оказаться позитивом.**
Кандидат берётся из выдачи, а выдача не размечена. Фрагмент на втором
месте вполне может отвечать на вопрос — просто этого никто не отметил.
Обучение на нём как на негативе прямо противоположно цели: модель
получает указание отталкивать правильный ответ.

Поэтому три правила отсечки, и каждое отвечает на свою опасность.

**1. Абсолютный порог `max_score`.** Учитель — кросс-энкодер, его оценка
после сигмоиды это вероятность релевантности, и 0,5 — его собственная
граница решения. Всё, что выше, учитель считает релевантным, и спорить
с учителем, из которого мы же и дистиллируем, бессмысленно.

**2. Отступ от позитива `absolute_margin`.** Кандидат с оценкой, близкой
к оценке эталона, неотличим от эталона для самого учителя. Порог в долях
вероятности, а не в логитах: доля читается глазами и попадает в отчёт
без пересчёта.

**3. Запрет на фрагменты того же акта `drop_same_act`.** Это правило
не из общей практики, а из устройства этого корпуса: фрагменты нарезаны
с перекрытием 15%, то есть соседний фрагмент акта буквально содержит часть
текста эталона. Как негатив он почти всегда ложный, а выглядит идеальным
трудным негативом — высокая оценка, другой идентификатор.

Отдельно о том, **откуда вообще берутся кандидаты**: только из фрагментов
обучающих актов. Негатив из акта dev или теста не портит разметку, но даёт
модели увидеть текст отложенного акта на обучении, а весь смысл сплита
по актам в том, что она его не видела. Ограничение накладывается на этапе
подготовки (D0), и здесь проверяется ещё раз.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from ..split import act_of_chunk

# Причины отбраковки. Имена уезжают в отчёт как есть, поэтому они русские
# и означают ровно то, что написано.
ЭТАЛОН = "сам эталон"
ТОТ_ЖЕ_АКТ = "тот же акт"
ВЫШЕ_ПОРОГА = "учитель считает релевантным"
БЛИЗКО_К_ЭТАЛОНУ = "оценка близка к эталону"
НЕ_ИЗ_ОБУЧАЮЩИХ = "фрагмент не обучающего акта"
ЛИШНИЕ = "сверх нормы на вопрос"


@dataclass(frozen=True)
class NegativeRules:
    """Пороги отсечки. Уезжают в журнал решений и в карточку модели."""

    max_score: float = 0.5
    absolute_margin: float = 0.10
    drop_same_act: bool = True
    per_query: int = 2
    train_acts: frozenset[str] = field(default_factory=frozenset)

    def as_dict(self) -> dict:
        return {"max_score": self.max_score,
                "absolute_margin": self.absolute_margin,
                "drop_same_act": self.drop_same_act,
                "per_query": self.per_query,
                "обучающих актов": len(self.train_acts) or "не ограничено"}


def probability(score: float, scale: str = "logit") -> float:
    """Оценка учителя в виде вероятности.

    Кросс-энкодер отдаёт логит, но некоторые обёртки применяют сигмоиду
    сами и возвращают уже вероятность. Что именно приехало, определяется
    на этапе подготовки по наблюдаемому разбросу и записывается в файл
    полем `scale`. Угадывать нельзя: порог 0,5 в логитах и в вероятностях
    означает совершенно разные вещи.
    """
    if scale == "sigmoid":
        return float(score)
    if scale != "logit":
        raise ValueError(f"неизвестная шкала оценок учителя: {scale}")
    if score >= 0:
        return 1.0 / (1.0 + math.exp(-score))
    # то же выражение, переписанное так, чтобы exp не переполнялся
    e = math.exp(score)
    return e / (1.0 + e)


def logit(score: float, scale: str = "logit") -> float:
    """Оценка учителя в виде логита.

    Нужна дистилляции: softmax по списку осмыслен в логитах, а не
    в вероятностях. Если обёртка уже применила сигмоиду, вероятность
    возвращается в логиты, и крайние значения подрезаются: p = 1,0
    означало бы бесконечность, а это не оценка, а потеря точности
    при записи.
    """
    if scale == "logit":
        return float(score)
    p = min(max(float(score), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


@dataclass
class Pick:
    """Что отобрано для одного вопроса и почему отброшено остальное."""

    query_id: str
    negatives: list[str]
    gold_probability: float
    dropped: dict[str, int]


def pick_negatives(record: Mapping, rules: NegativeRules,
                   scale: str = "logit") -> Pick:
    """Отобрать негативы для одного вопроса.

    `record` — строка файла подготовки D0: эталон, его оценка учителем
    и список кандидатов с оценками, уже по убыванию.
    """
    gold = record["gold_chunk_id"]
    gold_act = act_of_chunk(gold)
    gold_p = probability(record["gold_score"], scale)
    порог_близости = gold_p - rules.absolute_margin

    negatives: list[str] = []
    dropped: dict[str, int] = {}

    def отбросить(причина: str) -> None:
        dropped[причина] = dropped.get(причина, 0) + 1

    for chunk_id, score in record["candidates"]:
        if chunk_id == gold:
            отбросить(ЭТАЛОН)
            continue
        if rules.train_acts and act_of_chunk(chunk_id) not in rules.train_acts:
            отбросить(НЕ_ИЗ_ОБУЧАЮЩИХ)
            continue
        if rules.drop_same_act and act_of_chunk(chunk_id) == gold_act:
            отбросить(ТОТ_ЖЕ_АКТ)
            continue
        p = probability(score, scale)
        if p >= rules.max_score:
            отбросить(ВЫШЕ_ПОРОГА)
            continue
        if p >= порог_близости:
            отбросить(БЛИЗКО_К_ЭТАЛОНУ)
            continue
        if len(negatives) >= rules.per_query:
            отбросить(ЛИШНИЕ)
            continue
        negatives.append(chunk_id)

    return Pick(query_id=record["query_id"], negatives=negatives,
                gold_probability=gold_p, dropped=dropped)


def pick_all(records: Iterable[Mapping], rules: NegativeRules,
             scale: str = "logit") -> tuple[dict[str, list[str]], dict]:
    """Отобрать негативы по всему файлу подготовки.

    Второе значение — статистика для отчёта: сколько вопросов осталось
    без негативов вовсе и по каким причинам отброшены кандидаты. Вопрос
    без негативов не ошибка: он просто не попадёт в набор этапа B,
    а в контрастиве этапа A по-прежнему участвует.
    """
    out: dict[str, list[str]] = {}
    причины: dict[str, int] = {}
    хватило = 0
    пусто = 0
    всего = 0
    gold_p: list[float] = []
    for rec in records:
        всего += 1
        pick = pick_negatives(rec, rules, scale)
        out[pick.query_id] = pick.negatives
        gold_p.append(pick.gold_probability)
        for причина, n in pick.dropped.items():
            причины[причина] = причины.get(причина, 0) + n
        if len(pick.negatives) >= rules.per_query:
            хватило += 1
        elif not pick.negatives:
            пусто += 1
    gold_p.sort()
    stats = {
        "вопросов": всего,
        "хватило негативов": хватило,
        "негативов меньше нормы": всего - хватило - пусто,
        "без негативов вовсе": пусто,
        "медиана оценки эталона": round(gold_p[len(gold_p) // 2], 3) if gold_p else None,
        "отброшено кандидатов": dict(sorted(причины.items(), key=lambda kv: -kv[1])),
        "правила": rules.as_dict(),
        "шкала оценок": scale,
    }
    return out, stats


def detect_scale(scores: Iterable[float]) -> str:
    """Какая шкала у оценок учителя: логиты или уже вероятности.

    Определяется по разбросу, а не задаётся ключом: ошибка здесь тихая,
    а последствие — отсечка, которая либо пропускает всё, либо режет всё.
    Вероятности лежат в [0, 1]; логиты кросс-энкодера на размеченных парах
    уходят далеко за эти границы в обе стороны. Если все оценки попали
    в [0, 1], но при этом ни одна не отрицательна, считаем вероятностями.
    """
    значения = list(scores)
    if not значения:
        raise ValueError("нечего определять: оценок нет")
    if min(значения) >= 0.0 and max(значения) <= 1.0:
        return "sigmoid"
    return "logit"
