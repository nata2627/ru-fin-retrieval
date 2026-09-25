"""Раздел «Правовые акты» Банка России (cbr.ru/na/).

Список документов на странице подгружается скриптом, но за ним стоит обычный
серверный обработчик, отдающий готовую разметку постранично, — его и используем.
Фильтр по виду документа задаётся параметром VidId.

Проверено: раздел содержит 2302 документа, из них содержательные акты
(Положения, Указания, Инструкции) публикуются сканами и для поиска не годятся.
Раздел оставлен в проекте как источник метаданных и как предмет измерения
доли непригодных публикаций, но не как основа корпуса. Основа — «Вестник
Банка России», см. cbr_vestnik.py.
"""
from __future__ import annotations

import html
import re
from typing import Iterator

from ..http import Client

LIST_URL = "https://www.cbr.ru/Crosscut/LawActs/Page/94917"
FILE_URL = "https://www.cbr.ru/Crosscut/LawActs/File/{doc_id}"

# виды документов, доступные в фильтре раздела
VID = {
    10: "Положение Банка России",
    11: "Указание Банка России",
    18: "Инструкция Банка России",
    20: "Приказ Банка России",
    22: "Информационное письмо",
    23: "Письмо",
    24: "Методические рекомендации Банка России",
    27: "Официальное разъяснение Банка России",
}

_BLOCK = re.compile(r'<div class="cross-result" data-doc-id="(\d+)">(.*?)(?=<div class="cross-result"|\Z)', re.S)
_NUM = re.compile(r'<span class="number[^"]*">(.*?)</span>', re.S)
_DATE = re.compile(r'<span class="date[^"]*">(.*?)</span>', re.S)
_TITLE = re.compile(r'data-zoom-title="([^"]*)"')
_SRC = re.compile(r'<div class="source">(.*?)</div>', re.S)


def _clean(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s)).replace("\xa0", " ").strip()


def iter_catalog(client: Client, vid: int | None = None, max_pages: int = 400) -> Iterator[dict]:
    """Перебрать каталог постранично. Признак конца — страница без карточек."""
    for page in range(1, max_pages + 1):
        params = {"Date.Time": "Any", "Page": page}
        if vid is not None:
            params["VidId"] = vid
        blocks = _BLOCK.findall(client.get(LIST_URL, **params).text)
        if not blocks:
            return
        for doc_id, body in blocks:
            num, date = _NUM.search(body), _DATE.search(body)
            title, src = _TITLE.search(body), _SRC.search(body)
            yield {
                "doc_id": doc_id,
                "number": _clean(num.group(1)).removeprefix("№ ") if num else "",
                "date": _clean(date.group(1)).removeprefix("от ") if date else "",
                "vid": _clean(src.group(1)) if src else "",
                "title": _clean(title.group(1)) if title else "",
                "url": FILE_URL.format(doc_id=doc_id),
            }
