"""Нарезка выпуска «Вестника» на отдельные нормативные акты.

Выпуск — сборник: информационные сообщения, приказы об отзыве лицензий, курсы
валют, ставки, объявления и раздел «Официальные документы» с текстами актов.
Нужен только последний раздел, разложенный на отдельные акты.

Границы актов. Вёрстка «Вестника» менялась: в выпусках примерно с 2016 года
каждая страница раздела несёт колонтитул с названием акта, в более ранних —
нет. Поэтому опорой служит не колонтитул, а начало самого акта, одинаковое
во всех выпусках: строка с видом документа и рядом строка с его номером.

    Зарегистрировано                  ПОЛОЖЕНИЕ
    Министерством юстиции             О порядке формирования кредитными
    Российской Федерации              организациями резервов ...
    28 декабря 2015 года              № 590-П
    Регистрационный № 40318
    3 декабря 2015 года
    № 509-П
    ПОЛОЖЕНИЕ
    О расчете величины ...

Номер стоит то перед видом документа, то после названия, поэтому ищется
в окне вокруг строки с видом. Акт тянется до начала следующего акта.

Колонтитулы и оформление полосы вычищаются по частоте: строка, повторяющаяся
на большинстве страниц, — это колонтитул, а не текст. Такой признак не зависит
от вёрстки и снимает разом «ВЕСТНИК», номер выпуска, дату, название раздела
и колонтитул с названием акта.
"""
from __future__ import annotations

import collections
import re
from dataclasses import dataclass

from .pdftext import _run

KINDS = ("ПОЛОЖЕНИЕ", "ИНСТРУКЦИЯ", "УКАЗАНИЕ",
         "МЕТОДИЧЕСКИЕ РЕКОМЕНДАЦИИ", "ОФИЦИАЛЬНОЕ РАЗЪЯСНЕНИЕ")
KIND_SUFFIX = {"ПОЛОЖЕНИЕ": "П", "ИНСТРУКЦИЯ": "И", "УКАЗАНИЕ": "У",
               "МЕТОДИЧЕСКИЕ РЕКОМЕНДАЦИИ": "МР", "ОФИЦИАЛЬНОЕ РАЗЪЯСНЕНИЕ": "ОР"}

# В вёрстке встречаются неразрывный дефис и разные тире. Приводить строку через
# unicodedata.NFKC нельзя: заодно она превращает № в «No» и номер перестаёт
# опознаваться. Поэтому подменяются только тире.
_DASHES = str.maketrans({c: "-" for c in "‐‑‒–—―−"})

# Вид документа бывает набран прописными («ПОЛОЖЕНИЕ») и строчными
# («Методические рекомендации»), сам по себе и с приписью «Банка России».
# Регистр не различаем; посторонние срабатывания отсекает сверка с перечнем.
_KIND_LINE = re.compile(r"^(" + "|".join(KINDS) + r")(\s+БАНКА\s+РОССИИ)?$", re.I)
_NUMBER_LINE = re.compile(r"^№\s*(?P<n>[\d/]+-[А-ЯЁ]{1,3})$")
_PAGE_NO = re.compile(r"^\d{1,4}$")
# Номер акта стоит отдельной строкой, но расстояние до заголовка непостоянно:
# он бывает и прямо над видом документа, и боковой пометкой на полсотни строк
# ниже. Поэтому номер ищется по всей странице, а при нескольких кандидатах
# берётся ближайший к заголовку.
# перенос слова в конце строки: «прово-\nдиться». Склеиваем только когда
# продолжение начинается со строчной буквы, иначе можно слить составное слово.
_SOFT_HYPHEN = re.compile(r"([а-яёa-z])-\n([а-яёa-z])")
# доля страниц, начиная с которой повторяющаяся строка считается колонтитулом
_CHROME_SHARE = 0.30
_CHROME_MAX_LEN = 120


def norm_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.translate(_DASHES)).strip()


def normalize_number(s: str) -> str:
    """Привести номер к виду «590-П», как он записан в перечне актов."""
    return re.sub(r"\s+", "", s.translate(_DASHES)).upper()


def dehyphenate(text: str) -> str:
    """Убрать переносы, оставшиеся от двухколоночной вёрстки."""
    prev = None
    while prev != text:
        prev = text
        text = _SOFT_HYPHEN.sub(r"\1\2", text)
    return text


@dataclass
class Act:
    number: str
    kind: str
    heading: str
    first_page: int
    last_page: int
    text: str

    @property
    def suffix(self) -> str:
        return KIND_SUFFIX[self.kind]


def _chrome_lines(pages: list[list[str]]) -> set[str]:
    """Строки, повторяющиеся на многих страницах, — колонтитулы и оформление."""
    seen = collections.Counter()
    for p in pages:
        for line in set(p):
            if line and len(line) <= _CHROME_MAX_LEN:
                seen[line] += 1
    limit = max(2, int(len(pages) * _CHROME_SHARE))
    return {l for l, n in seen.items() if n >= limit}


def _find_starts(pages: list[list[str]],
                 expected: set[str] | None = None) -> list[tuple[int, int, str, str, str]]:
    """Начала актов: (страница, строка, номер, вид, название).

    Ссылка на другой акт, разорванная вёрсткой по строкам, неотличима
    от начала акта:

        Инструкция
        Банка России
        № 138-И

    Такая ссылка открывает несуществующий акт и забирает себе приложения
    настоящего. Поэтому, когда известен перечень актов выпуска, начала
    сверяются с ним: номер не из перечня начала акта не открывает,
    и текст остаётся за предыдущим актом.
    """
    starts = []
    for pi, lines in enumerate(pages):
        for li, line in enumerate(lines):
            m = _KIND_LINE.match(line)
            if not m:
                continue
            cands = [(abs(j - li), normalize_number(nm.group("n")))
                     for j, l in enumerate(lines)
                     if (nm := _NUMBER_LINE.match(l))]
            if not cands:
                # номер иногда переносится на следующую полосу
                nxt = pages[pi + 1] if pi + 1 < len(pages) else []
                cands = [(len(lines) + j, normalize_number(nm.group("n")))
                         for j, l in enumerate(nxt[:20])
                         if (nm := _NUMBER_LINE.match(l))]
            if not cands:
                continue
            number = min(cands)[1]
            if expected is not None and number not in expected:
                continue
            # название идёт сразу за видом документа до пустой строки или номера
            title = []
            for nxt in lines[li + 1: li + 6]:
                if not nxt or _NUMBER_LINE.match(nxt) or _PAGE_NO.match(nxt):
                    break
                title.append(nxt)
            starts.append((pi, li, number, m.group(1).upper(), " ".join(title).strip("“”\"«» ")))
    # один и тот же акт не должен открываться дважды: оставляем первое вхождение
    out, seen = [], set()
    for s in starts:
        if s[2] in seen:
            continue
        seen.add(s[2])
        out.append(s)
    return out


def split_issue(pdf_path: str, expected: set[str] | None = None) -> list[Act]:
    """Разобрать выпуск на акты.

    expected — номера актов, которые по перечню опубликованы в этом выпуске.
    Если передан, посторонние срабатывания отбрасываются (см. _find_starts).

    Текст извлекается без -layout: «Вестник» свёрстан в две колонки, и -layout
    склеивает их построчно, перемешивая два разных текста.
    """
    raw = _run(["pdftotext", pdf_path, "-"]).split("\f")
    pages = [[norm_line(l) for l in p.split("\n")] for p in raw]
    chrome = _chrome_lines(pages)
    starts = _find_starts(pages, expected)
    if not starts:
        return []

    acts: list[Act] = []
    for k, (pi, li, number, kind, title) in enumerate(starts):
        end_pi, end_li = (starts[k + 1][0], starts[k + 1][1]) if k + 1 < len(starts) else (len(pages) - 1, None)
        body: list[str] = []
        for p in range(pi, end_pi + 1):
            lines = pages[p]
            lo = li if p == pi else 0
            hi = end_li if (p == end_pi and end_li is not None) else len(lines)
            for line in lines[lo:hi]:
                if not line or line in chrome or _PAGE_NO.match(line):
                    continue
                body.append(line)
        acts.append(Act(number=number, kind=kind, heading=title,
                        first_page=pi + 1, last_page=end_pi + 1,
                        text=dehyphenate("\n".join(body)).strip()))
    return acts
