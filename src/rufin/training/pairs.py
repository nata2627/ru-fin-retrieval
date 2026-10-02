"""Пары «вопрос — эталонный фрагмент» и проверки, которые ломаются молча.

Три вещи здесь важнее остального кода обучения, потому что при поломке
они не падают, а портят результат.

**Первое. Эталон задан идентификатором фрагмента, а идентификатор
позиционный** («акт#номер по порядку»). Нарезка зависит от версии
токенизатора: сдвинулись границы — фрагмент под тем же именем стал другим
текстом. Поэтому эталон не берётся на веру: его текст обязан найтись
в нарезке, а число найденных печатается и сверяется.

**Второе. Сплит режется по актам.** Фрагменты одного акта нарезаны
с перекрытием 15%, то есть делят друг с другом текст. Обучающий вопрос,
попавший на акт из dev, означает, что dev меряет узнавание, а не поиск.
Проверяется не пересечение вопросов, а пересечение **актов**.

**Третье. Префиксы.** e5 обучена с пометками «query: » и «passage: »,
и они обязаны совпадать при обучении, при построении индекса и при поиске.
Берутся из `model_specs`, одного источника правды на весь проект.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from ..retrieval.model_specs import MODELS
from ..split import act_of_chunk


@dataclass(frozen=True)
class TrainPair:
    """Одна обучающая пара. Тексты уже с префиксами — дальше их не трогают."""

    query_id: str
    anchor: str
    positive: str
    gold_chunk_id: str
    act_id: str


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def acts_of(queries: Iterable[Mapping]) -> set[str]:
    """Акты, которых касается набор вопросов.

    Берётся не поле `act_id`, а акт из идентификатора эталонного фрагмента:
    поле могло не доехать или разойтись с эталоном, а эталон — это то,
    на чём модель учится.
    """
    return {act_of_chunk(q["gold_chunk_id"]) for q in queries}


def leaked_acts(train: Iterable[Mapping], held_out: Iterable[Mapping]) -> set[str]:
    """Акты, попавшие и в обучение, и в отложенную часть.

    Пустое множество — единственный допустимый ответ. Непустое означает,
    что dev (или тест) меряет узнавание текста, который модель видела,
    и прирост по нему ничего не значит.
    """
    return acts_of(train) & acts_of(held_out)


def check_group(queries: Iterable[Mapping], split: Mapping, group: str) -> dict[str, str]:
    """Вопросы, эталон которых лежит не в своей группе сплита.

    Возвращает `query_id -> фактическая группа` (или «нет в сплите»).
    Пустой словарь — набор чист.
    """
    where: dict[str, str] = {}
    for name, acts in split["gruppy"].items():
        for act_id in acts:
            where[act_id] = name
    bad: dict[str, str] = {}
    for q in queries:
        act = act_of_chunk(q["gold_chunk_id"])
        actual = where.get(act, "нет в сплите")
        if actual != group:
            bad[q["query_id"]] = actual
    return bad


def build_pairs(queries: Iterable[Mapping], texts: Mapping[str, str],
                spec_name: str = "e5-small") -> tuple[list[TrainPair], dict[str, int]]:
    """Собрать пары. Второе значение — статистика отбраковки.

    Вопрос выбрасывается, если его эталона нет в нарезке: учить на нём
    нечему, а молча подставить пустой текст значило бы учить модель
    притягивать вопрос к пустоте.
    """
    spec = MODELS[spec_name]
    out: list[TrainPair] = []
    stats = {"на входе": 0, "эталона нет в нарезке": 0, "пустой текст эталона": 0,
             "повтор вопроса": 0, "собрано пар": 0}
    видели: set[str] = set()
    for q in queries:
        stats["на входе"] += 1
        gold = q["gold_chunk_id"]
        body = texts.get(gold)
        if body is None:
            stats["эталона нет в нарезке"] += 1
            continue
        if not body.strip():
            stats["пустой текст эталона"] += 1
            continue
        if q["query_id"] in видели:
            stats["повтор вопроса"] += 1
            continue
        видели.add(q["query_id"])
        out.append(TrainPair(
            query_id=q["query_id"],
            anchor=spec.query_prefix + q["text"],
            positive=spec.passage_prefix + body,
            gold_chunk_id=gold,
            act_id=act_of_chunk(gold)))
    stats["собрано пар"] = len(out)
    return out, stats


def as_dataset(pairs: Iterable[TrainPair],
               negatives: Mapping[str, list[str]] | None = None,
               texts: Mapping[str, str] | None = None,
               spec_name: str = "e5-small",
               per_query: int = 2) -> dict[str, list]:
    """Пары в виде столбцов для `datasets.Dataset`.

    Без негативов столбцы `anchor` и `positive`: негативами служат эталоны
    остальных вопросов батча. С негативами добавляются `negative_1..k`,
    и число столбцов обязано быть одинаковым у всех строк — иначе
    `datasets` не соберёт таблицу. Поэтому вопрос, у которого после
    отсечки негативов осталось меньше `per_query`, в этот набор не идёт
    вовсе; сколько таких, считает вызывающая сторона.
    """
    spec = MODELS[spec_name]
    cols: dict[str, list] = {"anchor": [], "positive": []}
    if negatives is None:
        for p in pairs:
            cols["anchor"].append(p.anchor)
            cols["positive"].append(p.positive)
        return cols
    if texts is None:
        raise ValueError("с негативами нужны тексты фрагментов")
    for i in range(1, per_query + 1):
        cols[f"negative_{i}"] = []
    for p in pairs:
        negs = negatives.get(p.query_id, [])
        if len(negs) < per_query:
            continue
        cols["anchor"].append(p.anchor)
        cols["positive"].append(p.positive)
        for i, chunk_id in enumerate(negs[:per_query], start=1):
            cols[f"negative_{i}"].append(spec.passage_prefix + texts[chunk_id])
    return cols


def train_acts(split: Mapping) -> frozenset[str]:
    """Акты обучающей группы сплита.

    Нужны там, где собираются кандидаты в негативы и в списки дистилляции:
    кандидат из акта dev или теста не портит разметку, но даёт модели
    увидеть текст отложенного акта на обучении — а весь смысл сплита
    по актам в том, что она его не видела.
    """
    from ..split import TRAIN
    return frozenset(split["gruppy"][TRAIN])


def assert_only_train_acts(chunk_ids: Iterable[str], split: Mapping,
                           что: str = "кандидаты") -> None:
    """Отказаться работать, если среди фрагментов есть не обучающие акты.

    Отказ, а не предупреждение: утечку текста отложенного акта в обучение
    по метрикам не видно, а задним числом её не отменить — веса уже обучены.
    """
    разрешено = train_acts(split)
    чужие = sorted({act_of_chunk(c) for c in chunk_ids} - разрешено)
    if чужие:
        raise ValueError(
            f"{что}: {len(чужие)} актов не из обучающей группы сплита, "
            f"например {чужие[:3]}. Текст отложенного акта попал бы "
            f"в обучение, и по метрикам этого не видно.")
