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


def _header(act: dict, section: str) -> str:
    """Шапка, приписываемая к чанку: чей это акт и какой раздел."""
    head = f"{act['kind'].capitalize()} Банка России № {act['number']} от {act['date']}. {act['title']}"
    return f"{head}\n{section}" if section else head


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
    head_cost = ruler.count(_header(act, "") + "\n\n") if add_heading else 0
    body_budget = max(32, size - head_cost)

    def emit(body: str, section: str, units: str, s: int, e: int) -> None:
        head = _header(act, section) if add_heading else ""
        prefix = f"{head}\n\n" if head else ""
        allow = max(1, size - (ruler.count(prefix) if prefix else 0))
        # подрезка повторяется: обрезанный по границе токена хвост при
        # повторной токенизации иногда склеивается с соседом и даёт +1 токен
        for _ in range(4):
            offs = ruler.offsets(body)
            if len(offs) <= allow:
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
