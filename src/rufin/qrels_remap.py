"""Перенос эталонов с одной нарезки на другую по символьным границам.

Эталон записан идентификатором фрагмента вида «590-П_28062017#0034», а
нумерация фрагментов своя у каждой нарезки: у нарезки на 256 токенов под
тем же номером лежит другой текст, а чаще такого номера нет вовсе. Поэтому
метрики по другой нарезке, посчитанные с исходной разметкой, дают нули —
и ноль этот означает не «нарезка плохая», а «эталон не найден».

Привязка, которая переживает смену нарезки, одна: отрезок символов внутри
акта. У каждого фрагмента сохранены `act_id`, `char_start` и `char_end`,
и этого достаточно.

Правило переноса:

* кандидат — фрагмент целевой нарезки из того же акта, чей отрезок
  пересекается с отрезком эталона;
* оценку исходного эталона получает кандидат с наибольшим перекрытием
  в символах: он и есть «тот же текст» в новой нарезке. Двоек становится
  ровно столько же, сколько было, — это важно, иначе нарезки сравнивались
  бы при разном числе правильных ответов на запрос;
* остальные пересекающиеся получают 1, и только если перекрытие
  существенное: доля эталона, попавшая в кандидата, не ниже порога.
  Порог задаётся снаружи и записывается в отчёт — от него зависят все
  числа абляции;
* оценку 1 переносит только эталон с оценкой 2. Сосед соседа — уже
  догадка, а не измерение.

Доля считается от эталона, а не от кандидата, и это тот же смысл, что
у `MIN_SPAN_OVERLAP` в сборке разметки: «сколько эталонного текста лежит
в кандидате». Доля от кандидата отвечала бы на другой вопрос и при дроблении
512 -> 256 давала бы единицу любому вложенному куску — то есть записывала бы
в релевантные все три части эталона, хотя ответ лежит в одной.

Отдельно оговорено совпадение отрезков: если в целевой нарезке есть фрагмент
ровно с теми же границами, он забирает оценку, и соседи не добавляются.
Догадываться не о чем — текст воспроизведён точно. Отсюда следует, что
перенос нарезки в саму себя возвращает исходный файл без изменений при любом
пороге; это и проверяется тестом.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Span:
    """Фрагмент как отрезок символов внутри акта."""
    chunk_id: str
    act_id: str
    char_start: int
    char_end: int

    @property
    def length(self) -> int:
        return max(0, self.char_end - self.char_start)

    @classmethod
    def from_chunk(cls, chunk: dict) -> "Span":
        return cls(chunk["chunk_id"], chunk["act_id"],
                   int(chunk["char_start"]), int(chunk["char_end"]))


def overlap_chars(a: Span, b: Span) -> int:
    """Длина пересечения двух отрезков в символах; у разных актов — ноль."""
    if a.act_id != b.act_id:
        return 0
    return max(0, min(a.char_end, b.char_end) - max(a.char_start, b.char_start))


def share_of_gold(gold: Span, candidate: Span) -> float:
    """Какая доля эталонного отрезка попала в кандидата."""
    if gold.length <= 0:
        return 0.0
    return overlap_chars(gold, candidate) / gold.length


@dataclass
class Transfer:
    """Что получилось из одного эталона."""
    target: dict[str, int]
    exact: bool
    best_share: float
    extra_shares: list[float]


def transfer_one(gold: Span, score: int, candidates: Iterable[Span],
                 threshold: float) -> Transfer:
    """Перенести один эталон на целевую нарезку."""
    hits = [(c, overlap_chars(gold, c)) for c in candidates]
    hits = [(c, n) for c, n in hits if n > 0]
    if not hits:
        return Transfer({}, False, 0.0, [])

    for c, _ in hits:
        if c.char_start == gold.char_start and c.char_end == gold.char_end:
            return Transfer({c.chunk_id: score}, True, 1.0, [])

    # при равном перекрытии берётся фрагмент с более ранним началом:
    # выбор должен быть однозначным, иначе перенос не воспроизводится
    best, _ = max(hits, key=lambda h: (h[1], -h[0].char_start, h[0].chunk_id))
    out = {best.chunk_id: score}
    extra: list[float] = []
    if score >= 2:
        for c, _ in hits:
            if c.chunk_id == best.chunk_id:
                continue
            share = share_of_gold(gold, c)
            extra.append(share)
            if share >= threshold:
                out.setdefault(c.chunk_id, 1)
    return Transfer(out, False, share_of_gold(gold, best), extra)


@dataclass
class RemapStats:
    """Статистика переноса: без неё нельзя понять, чему верить в абляции."""
    entries: int = 0
    exact: int = 0
    lost: int = 0                 # эталону не нашлось ни одного пересечения
    unknown: int = 0              # эталона нет в исходной нарезке
    queries_in: int = 0
    queries_out: int = 0
    pairs_out: int = 0
    grade_two: int = 0
    grade_one: int = 0
    lost_ids: list[str] = field(default_factory=list)
    unknown_ids: list[str] = field(default_factory=list)
    best_shares: list[float] = field(default_factory=list)
    extra_shares: list[float] = field(default_factory=list)


def remap(qrels: dict[str, dict[str, int]], source: dict[str, Span],
          target: Sequence[Span], threshold: float,
          ) -> tuple[dict[str, dict[str, int]], RemapStats]:
    """Перенести всю разметку: новая разметка и статистика переноса.

    `source` — фрагменты исходной нарезки по идентификатору, `target` — все
    фрагменты целевой. Кандидаты ищутся только внутри того же акта: на
    шестидесяти тысячах фрагментов это разница между секундой и минутами.
    """
    by_act: dict[str, list[Span]] = {}
    for span in target:
        by_act.setdefault(span.act_id, []).append(span)
    for spans in by_act.values():
        spans.sort(key=lambda s: (s.char_start, s.char_end, s.chunk_id))

    stats = RemapStats()
    out: dict[str, dict[str, int]] = {}
    for qid in sorted(qrels):
        stats.queries_in += 1
        moved: dict[str, int] = {}
        for cid in sorted(qrels[qid]):
            score = qrels[qid][cid]
            stats.entries += 1
            gold = source.get(cid)
            if gold is None:
                stats.unknown += 1
                stats.unknown_ids.append(f"{qid}\t{cid}")
                continue
            t = transfer_one(gold, score, by_act.get(gold.act_id, ()), threshold)
            if not t.target:
                stats.lost += 1
                stats.lost_ids.append(f"{qid}\t{cid}")
                continue
            stats.exact += int(t.exact)
            stats.best_shares.append(t.best_share)
            stats.extra_shares.extend(t.extra_shares)
            for new_id, new_score in t.target.items():
                moved[new_id] = max(moved.get(new_id, 0), new_score)
        if moved:
            out[qid] = moved
            stats.queries_out += 1
            stats.pairs_out += len(moved)
            stats.grade_two += sum(1 for v in moved.values() if v == 2)
            stats.grade_one += sum(1 for v in moved.values() if v == 1)
    return out, stats
