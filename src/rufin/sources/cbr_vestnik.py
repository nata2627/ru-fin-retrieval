"""«Вестник Банка России» — основной источник корпуса.

Почему именно он. Отдельные файлы Положений и Инструкций на сайте ЦБ выложены
сканами подписанных документов с негодным OCR-слоем (см. pdftext.py). Ровно те
же акты в «Вестнике» набраны в цифре: проверка выпуска № 65-66 (1899-1900)
от 04.08.2017 — 135 страниц, 614 тыс. символов, 99,4% кириллицы, внутри полные
тексты Инструкции 180-И и Положения 590-П.

Устройство раздела:
  * поиск по актам (vestnik-search) отдаёт таблицу «номер / дата / название /
    выпуск» и покрывает публикации с 1999 года;
  * сам текст лежит в PDF выпуска, один выпуск содержит несколько актов
    (медиана 3, максимум 21), поэтому акты вырезаются из выпуска по оглавлению.

Поиск ищет подстроку и в номере, и в названии, поэтому запрос из одной буквы
возвращает почти весь индекс. Полный перечень собирается несколькими такими
запросами с последующим объединением по (номер, дата).
"""
from __future__ import annotations

import html
import re
from typing import Iterable, Iterator

from ..http import Client

SEARCH_URL = "https://www.cbr.ru/about_br/publ/vestnik-search/"
YEARS_URL = "https://www.cbr.ru/about_br/publ/vestnik/year/{year}/"

# суффиксы номеров: П — положение, У — указание, И — инструкция,
# Т — письмо, МР — методические рекомендации, ОР — официальное разъяснение
QUERIES = ("П", "У", "И", "Т", "МР", "ОР")

_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
_CELL = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S)
_LINK = re.compile(r'href="(/Queries/UniDbQuery/File/85920/-1/[^"]+)"')
# номер акта: «590-П», «4874-У», «180-И», «28-МР»
_NUMBER = re.compile(r"^[\d/]+[-–][А-ЯA-Zа-яa-z/-]{1,8}$")

AMENDMENT = re.compile(r"внесени\w*\s+изменени|признании\s+утративш", re.I)


def _txt(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


def _parse_rows(page: str) -> Iterator[dict]:
    for row in _ROW.findall(page):
        cells = [_txt(c) for c in _CELL.findall(row)]
        link = _LINK.search(row)
        if len(cells) < 4 or not link or not _NUMBER.match(cells[0]):
            continue
        yield {
            "number": cells[0],
            "date": cells[1],
            "title": cells[2],
            "issue": cells[3],
            "issue_url": "https://www.cbr.ru" + link.group(1),
        }


def fetch_acts_index(client: Client, queries: Iterable[str] = QUERIES) -> list[dict]:
    """Собрать перечень актов, опубликованных в «Вестнике».

    Возвращает записи, объединённые по паре (номер, дата): один и тот же акт
    попадает в выдачу нескольких запросов.
    """
    acts: dict[tuple[str, str], dict] = {}
    for q in queries:
        page = client.get(SEARCH_URL, **{"UniDbQuery.Posted": "True", "UniDbQuery.stext": q}).text
        for rec in _parse_rows(page):
            acts.setdefault((rec["number"], rec["date"]), rec)
    return sorted(acts.values(), key=lambda a: (a["date"][-4:], a["date"][3:5], a["date"][:2]))


def act_type(number: str) -> str:
    """Суффикс номера акта: 590-П -> П."""
    return re.sub(r"^.*?[-–]", "", number)


def is_amendment(act: dict) -> bool:
    """Правка или отмена другого акта: самостоятельного текста в ней нет.

    Такие документы состоят из указаний «в пункте 3.1 слова ... заменить
    словами ...» и как ответ на вопрос пользователя бесполезны, поэтому
    в корпус не берутся.
    """
    return bool(AMENDMENT.search(act["title"]))


def year(act: dict) -> int:
    y = act["date"][-4:]
    return int(y) if y.isdigit() else 0


def select_corpus(acts: list[dict], types: Iterable[str] = ("П", "И", "У", "МР", "ОР"),
                  since: int = 2013) -> list[dict]:
    """Отбор состава корпуса: самостоятельные акты нужных видов начиная с года."""
    types = set(types)
    return [a for a in acts
            if act_type(a["number"]) in types and not is_amendment(a) and year(a) >= since]
