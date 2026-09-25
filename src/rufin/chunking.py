"""Нарезка актов на чанки.

Две стратегии, которые потом сравниваются в абляциях:

* «по структуре» — акт разбирается на пункты («3.1.1. Профессиональное
  суждение выносится…»), пункты собираются в чанк подряд, пока помещаются.
  Граница чанка проходит по границе пункта, а не посреди фразы;
* «по длине» — скользящее окно по тексту без оглядки на структуру.

Длина считается в токенах того же токенизатора, которым пользуется эмбеддер
(XLM-RoBERTa — общий у bge-m3 и multilingual-e5). Считать длину в символах
нельзя: в юридическом тексте много цифр и сокращений, и отношение символов
к токенам гуляет достаточно, чтобы «512 токенов» превратились то в 380,
то в 700.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Iterator, Sequence

# заголовки разделов внутри акта
_CHAPTER = re.compile(r"^Глава\s+(\d+)\.?\s*(.*)$")
_ANNEX = re.compile(r"^(Приложение(?:\s+\d+)?)\b\s*(.*)$")
# строка оглавления: «Приложение 3 . . . . . . . 47». Выглядит как заголовок
# раздела, но им не является, и подхватывать её название нельзя.
_TOC_LINE = re.compile(r"\.\s*\.\s*\.|\.{4,}")
# начало пункта: «1.1.», «3.1.4.», «12.» — номер и точка в начале строки
_UNIT = re.compile(r"^(\d+(?:\.\d+)*)\.\s+(?=\S)")


@dataclass
class Chunk:
    chunk_id: str
    act_id: str
    number: str
    date: str
    type: str
    title: str
    issue: str
    url: str
    section: str
    units: str
    position: int
    char_start: int
    char_end: int
    text: str

    def as_dict(self) -> dict:
        return asdict(self)


class TokenRuler:
    """Счётчик длины в токенах.

    Если токенизатор недоступен (нет сети или модели), длина оценивается
    по символам. Коэффициент не выдуман: он измеряется на самом корпусе
    методом calibrate() и пишется в отчёт, чтобы разница была видна.
    """

    def __init__(self, tokenizer=None, chars_per_token: float = 3.2):
        self.tok = tokenizer
        self.chars_per_token = chars_per_token

    def count(self, text: str) -> int:
        if self.tok is None:
            return max(1, int(len(text) / self.chars_per_token))
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def offsets(self, text: str) -> list[tuple[int, int]]:
        """Границы токенов в символах — нужны для нарезки по длине."""
        if self.tok is None:
            step = max(1, int(self.chars_per_token))
            return [(i, min(i + step, len(text))) for i in range(0, len(text), step)]
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return [tuple(o) for o in enc["offset_mapping"] if o[1] > o[0]]

    def calibrate(self, texts: Sequence[str]) -> float:
        """Сколько символов приходится на токен в этом корпусе."""
        if self.tok is None:
            return self.chars_per_token
        chars = sum(len(t) for t in texts)
        toks = sum(self.count(t) for t in texts)
        return chars / toks if toks else self.chars_per_token


@dataclass
class Unit:
    """Структурная единица акта: пункт, абзац преамбулы или кусок приложения."""
    number: str
    section: str
    text: str
    start: int
    end: int


def parse_units(text: str) -> list[Unit]:
    """Разобрать акт на пункты с запоминанием раздела, к которому они относятся."""
    units: list[Unit] = []
    section = ""
    cur: Unit | None = None
    pos = 0
    for line in text.split("\n"):
        start, end = pos, pos + len(line)
        pos = end + 1

        if _TOC_LINE.search(line):
            pos = end + 1 if False else pos      # строка оглавления: не заголовок и не текст
            if cur is not None:
                cur.text += "\n" + line
                cur.end = end
            continue

        ch, an = _CHAPTER.match(line), _ANNEX.match(line)
        if ch:
            section = f"Глава {ch.group(1)}. {ch.group(2)}".strip().rstrip(".")
            cur = None
            continue
        if an:
            section = an.group(1) + (f". {an.group(2)}" if an.group(2) else "")
            cur = None
            continue

        m = _UNIT.match(line)
        if m:
            cur = Unit(number=m.group(1), section=section, text=line, start=start, end=end)
            units.append(cur)
        elif cur is not None:
            cur.text += "\n" + line
            cur.end = end
        else:
            # текст до первого пункта: преамбула раздела
            cur = Unit(number="", section=section, text=line, start=start, end=end)
            units.append(cur)
    return [u for u in units if u.text.strip()]


# название в перечне часто уже начинается с вида документа: «Указание Банка
# России "О порядке…"». Повторять его в шапке незачем.
_KIND_PREFIX = re.compile(
    r"^(Указание|Положение|Инструкция|Методические\s+рекомендации|"
    r"Официальное\s+разъяснение)(\s+Банка\s+России)?\s*", re.I)


def _header(act: dict, section: str, ruler: "TokenRuler | None" = None,
            max_tokens: int | None = None) -> str:
    """Шапка, приписываемая к чанку: чей это акт и какой раздел.

    Номер и дата идут первыми и не обрезаются никогда: по ним акт и опознают.
    Режется только название. Без обрезки шапка съедала весь чанк — у актов
    с названием в полторы тысячи знаков при размере чанка 256 токенов
    встречались шапки на 323 токена, и содержания в чанк не попадало вовсе.
    """
    kind = act["kind"].capitalize()
    stem = f"{kind} Банка России № {act['number']} от {act['date']}"
    title = _KIND_PREFIX.sub("", act["title"].strip()).strip(' "«»')
    tail = f". {section}" if section else ""

    if ruler is None or not max_tokens:
        return f"{stem}. {title}{tail}"

    room = max_tokens - ruler.count(stem + tail) - 2
    if room <= 4:
        return f"{stem}{tail}"
    offs = ruler.offsets(title)
    if len(offs) > room:
        title = title[:offs[room - 1][1]].rstrip(" ,.;-") + "…"
    return f"{stem}. {title}{tail}"


def _split_long(text: str, ruler: TokenRuler, size: int, overlap: int,
                base: int) -> Iterator[tuple[str, int, int]]:
    """Нарезать длинный кусок скользящим окном по токенам."""
    offs = ruler.offsets(text)
    if not offs:
        return
    step = max(1, size - overlap)
    for i in range(0, len(offs), step):
        window = offs[i:i + size]
        if not window:
            break
        s, e = window[0][0], window[-1][1]
        yield text[s:e], base + s, base + e
        if i + size >= len(offs):
            break


def chunk_act(act: dict, ruler: TokenRuler, *, size: int = 512, overlap_share: float = 0.15,
              strategy: str = "structure", add_heading: bool = True) -> list[Chunk]:
    """Нарезать один акт.

    size — целевая длина чанка в токенах; overlap_share — доля перекрытия;
    strategy — «structure» или «length»; add_heading — приписывать ли шапку
    с названием акта и раздела (проверяется отдельной абляцией).
    """
    overlap = int(size * overlap_share)
    text = act["text"]
    out: list[Chunk] = []
    # Размер чанка — жёсткая граница, а не пожелание: multilingual-e5 обрезает
    # вход на 512 токенах, и всё, что не поместилось, просто не будет
    # проиндексировано. Поэтому шапка вычитается из бюджета, а собранный
    # чанк на всякий случай подрезается по границе токена.
    # шапке отводится не больше 40% чанка: остальное обязано достаться тексту
    max_head = max(16, int(size * 0.4))
    # один и тот же раздел встречается в акте десятки раз, а сборка шапки
    # требует токенизации; без кэша она становится заметной частью работы
    head_cache: dict[str, tuple[str, int]] = {}

    def header_for(section: str) -> tuple[str, int]:
        if section not in head_cache:
            h = _header(act, section, ruler, max_head)
            head_cache[section] = (h, ruler.count(f"{h}\n\n"))
        return head_cache[section]

    head_cost = header_for("")[1] if add_heading else 0
    body_budget = max(32, size - head_cost)

    def emit(body: str, section: str, units: str, s: int, e: int) -> None:
        head, prefix_tokens = header_for(section) if add_heading else ("", 0)
        prefix = f"{head}\n\n" if head else ""
        allow = max(1, size - prefix_tokens)
        # подрезка повторяется: обрезанный по границе токена хвост при
        # повторной токенизации иногда склеивается с соседом и даёт +1 токен
        for _ in range(4):
            offs = ruler.offsets(body)
            if not offs or len(offs) <= allow:
                break
            if allow < 1:
                body = ""
                break
            body = body[:offs[allow - 1][1]]
            e = s + len(body)
            allow -= 1
        full = (prefix + body).strip()
        if not full:
            return
        out.append(Chunk(
            chunk_id=f"{act['act_id']}#{len(out):04d}",
            act_id=act["act_id"], number=act["number"], date=act["date"],
            type=act["type"], title=act["title"], issue=act["issue"], url=act["issue_url"],
            section=section, units=units, position=len(out),
            char_start=s, char_end=e, text=full,
        ))

    if strategy == "length":
        for body, s, e in _split_long(text, ruler, body_budget, overlap, 0):
            emit(body, "", "", s, e)
        return out

    # по структуре: пункты копятся, пока помещаются в размер
    budget = body_budget
    buf: list[Unit] = []
    buf_tokens = 0

    def flush() -> None:
        nonlocal buf, buf_tokens
        if not buf:
            return
        nums = [u.number for u in buf if u.number]
        units = f"{nums[0]}–{nums[-1]}" if len(nums) > 1 else (nums[0] if nums else "")
        emit("\n".join(u.text for u in buf), buf[0].section, units, buf[0].start, buf[-1].end)
        # перекрытие: последние пункты переходят в следующий чанк
        tail: list[Unit] = []
        acc = 0
        for u in reversed(buf):
            t = ruler.count(u.text)
            if acc + t > overlap:
                break
            tail.insert(0, u)
            acc += t
        buf, buf_tokens = (tail, acc) if len(tail) < len(buf) else ([], 0)

    for u in parse_units(text):
        t = ruler.count(u.text)
        if t > budget:                      # пункт сам по себе длиннее чанка
            flush()
            for body, s, e in _split_long(u.text, ruler, budget, overlap, u.start):
                emit(body, u.section, u.number, s, e)
            continue
        if buf and (buf_tokens + t > budget or buf[0].section != u.section):
            flush()
        buf.append(u)
        buf_tokens += t
    flush()
    return out
