"""Журнал рецепта: таблица «этап → dev» с вердиктом по каждому этапу.

Зачем журнал, а не просто лучшие веса. Отклонённый этап — такой же
результат, как принятый, и в отчёт он идёт целиком. Таблица, в которой
остались только удачные шаги, читается как «всё получилось с первого
раза» и ничего не говорит о том, что в этой задаче не работает.

**Правило приёмки объявлено заранее и записано здесь, а не подбирается
по полученным числам.** dev — это 300 вопросов, и доверительный интервал
отдельной цифры на нём ±0,045. Сравнивать поэтому надо не цифры, а разницу
на одних и тех же вопросах: парный бутстрэп даёт интервал втрое уже,
потому что ошибки двух моделей на одном вопросе скоррелированы.

Вердикты:

* **принят** — разница с предыдущим лучшим положительна и перестановочный
  тест даёт p < 0,05. Веса становятся новым лучшим;
* **не установлен** — разница положительна, но p ≥ 0,05. Этап не принимается:
  усложнение рецепта без свидетельства выигрыша — это подгонка под dev,
  и на тесте оно не воспроизведётся;
* **отклонён** — разница не положительна.

Отдельное правило для этапа D (матрёшка). Его цель не качество,
а применимость: вектор на 128 измерений вместо 384 уменьшает индекс втрое.
Поэтому он принимается, если **не ухудшил** dev заметно, то есть интервал
разницы накрывает ноль или разница положительна. Правило объявлено здесь
вместе с остальными, а не придумано после прогона.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np

from .. import metrics as M

ACCEPT_P = 0.05
# Этапы, которые принимаются по применимости, а не по приросту качества.
BY_APPLICABILITY = ("D",)

ПРИНЯТ = "принят"
НЕ_УСТАНОВЛЕН = "не установлен"
ОТКЛОНЁН = "отклонён"


def verdict(diff_mean: float, lo: float, hi: float, p: float,
            by_applicability: bool = False) -> str:
    """Вердикт по разнице с предыдущим лучшим. Правило объявлено заранее."""
    if by_applicability:
        return ПРИНЯТ if (diff_mean >= 0 or lo <= 0 <= hi) else ОТКЛОНЁН
    if diff_mean <= 0:
        return ОТКЛОНЁН
    return ПРИНЯТ if p < ACCEPT_P else НЕ_УСТАНОВЛЕН


@dataclass
class Entry:
    """Один этап: что обучали, что вышло на dev, принят или нет.

    Поквериные значения метрики хранятся целиком (300 чисел — это килобайты)
    и без них журнал бесполезен: по среднему нельзя посчитать ни интервал,
    ни парное сравнение с другим этапом задним числом.
    """

    tag: str
    stage: str
    note: str = ""
    extends: str = ""
    weights: str = ""
    dev: dict = field(default_factory=dict)          # метрика -> {mean, lo, hi}
    per_query: dict = field(default_factory=dict)    # метрика -> список значений
    qids: list[str] = field(default_factory=list)
    diff: dict | None = None                         # против предыдущего лучшего
    verdict: str = ""
    compared_with: str = ""
    seconds: float = 0.0
    params: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    @property
    def ndcg(self) -> float:
        return float(self.dev.get("NDCG@10", {}).get("mean", float("nan")))


class Journal:
    """Таблица этапов. Пишется на видеокарте, читается и печатается на маке."""

    METRIC = "NDCG@10"

    def __init__(self, entries: list[Entry] | None = None) -> None:
        self.entries: list[Entry] = entries or []

    # ---- чтение и запись ----

    @classmethod
    def load(cls, path: str) -> "Journal":
        import os
        if not os.path.exists(path):
            return cls()
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        return cls([Entry(**e) for e in raw.get("этапы", [])])

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"правило приёмки": {"p": ACCEPT_P,
                                           "по применимости": list(BY_APPLICABILITY)},
                       "этапы": [asdict(e) for e in self.entries]},
                      f, ensure_ascii=False, indent=1)

    # ---- наполнение ----

    def by_tag(self, tag: str) -> Entry | None:
        for e in self.entries:
            if e.tag == tag:
                return e
        return None

    def baseline(self) -> Entry | None:
        """Нулевая точка: необученный ученик. Всегда первая запись."""
        return self.by_tag("базовая")

    def best(self, кроме: str = "") -> Entry | None:
        """Последний принятый этап: с его составом сравнивается следующий.

        Именно последний, а не лучший по цифре. Принятым этап становится
        только если он обогнал предыдущее принятое, поэтому последний
        принятый и есть лучший — но это следствие правила, а не отдельное
        условие.

        `кроме` нужен при переделке этапа: этап нельзя сравнивать с собой,
        иначе переделка лучшего этапа осталась бы вовсе без вердикта
        и потеряла бы его в таблице.
        """
        принятые = [e for e in self.entries if e.verdict == ПРИНЯТ and e.tag != кроме]
        if принятые:
            return принятые[-1]
        базовая = self.baseline()
        return базовая if базовая and базовая.tag != кроме else None

    def add(self, entry: Entry) -> Entry:
        """Добавить этап, сравнив его с предыдущим лучшим.

        Сравнение парное и только по тем вопросам, что есть у обоих:
        разные знаменатели сделали бы разницу бессмысленной.
        """
        прежний = self.best(кроме=entry.tag)
        if прежний is not None:
            entry.compared_with = прежний.tag
            общие = [q for q in entry.qids if q in set(прежний.qids)]
            a = _values(entry, self.METRIC, общие)
            b = _values(прежний, self.METRIC, общие)
            interval, p = M.paired_diff_ci(a, b)
            entry.diff = {"metric": self.METRIC, "mean": interval.mean,
                          "lo": interval.lo, "hi": interval.hi, "p": p,
                          "запросов": len(общие)}
            entry.verdict = verdict(interval.mean, interval.lo, interval.hi, p,
                                    by_applicability=entry.stage in BY_APPLICABILITY)
        self.entries = [e for e in self.entries if e.tag != entry.tag] + [entry]
        return entry

    # ---- печать ----

    def table(self) -> str:
        """Таблица для отчёта, в markdown. Печатается целиком, с отклонённым."""
        строки = ["| этап | метка | dev NDCG@10 | разница с предыдущим лучшим | p | вердикт |",
                  "|---|---|---:|---|---:|---|"]
        for e in self.entries:
            dev = e.dev.get(self.METRIC, {})
            цифра = (f"{dev.get('mean', float('nan')):.3f} "
                     f"[{dev.get('lo', float('nan')):.3f}; {dev.get('hi', float('nan')):.3f}]")
            if e.diff:
                разница = (f"{e.diff['mean']:+.3f} "
                           f"[{e.diff['lo']:+.3f}; {e.diff['hi']:+.3f}]")
                p = f"{e.diff['p']:.3f}"
            else:
                разница, p = "базовая линия", ""
            строки.append(f"| {e.stage} | `{e.tag}` | {цифра} | {разница} | {p} "
                          f"| {e.verdict or '—'} |")
        return "\n".join(строки)


def _values(entry: Entry, metric: str, qids: list[str]) -> np.ndarray:
    по_id = dict(zip(entry.qids, entry.per_query.get(metric, [])))
    return np.array([по_id.get(q, float("nan")) for q in qids], dtype=float)


def entry_from_per_query(tag: str, stage: str, per_query: dict, qids: list[str],
                         **kwargs) -> Entry:
    """Запись журнала по поквериным значениям метрик.

    `per_query` — то, что вернул `rufin.metrics.per_query`, то есть массивы
    на запрос. Средние и интервалы считаются здесь, чтобы и на видеокарте,
    и на маке они считались одним кодом.
    """
    dev = {}
    чистый = {}
    for name in M.METRICS:
        values = np.asarray(per_query[name], dtype=float)
        ci = M.bootstrap_ci(values)
        dev[name] = {"mean": ci.mean, "lo": ci.lo, "hi": ci.hi}
        чистый[name] = [None if np.isnan(v) else float(v) for v in values]
    return Entry(tag=tag, stage=stage, dev=dev, per_query=чистый, qids=list(qids),
                 **kwargs)
