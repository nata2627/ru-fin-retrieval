"""Извлечение текста из PDF и оценка пригодности текстового слоя.

Ключевая проблема источника: официальные документы публикуются двумя способами.
Часть — цифровые PDF с нормальным текстом, часть — сканы подписанных бумаг
с OCR-слоем, распознанным в латинском режиме: «Совета директоров» превращается
в «CoseTa ,li;HpeKTopoB». Формально текст есть, и обычная проверка «непустой ли
текст» такие файлы пропускает. Поэтому пригодность определяется долей кириллицы
среди букв, а не объёмом текста.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass

CYR = re.compile(r"[а-яёА-ЯЁ]")
LAT = re.compile(r"[a-zA-Z]")

# порог подобран по наблюдаемым распределениям: у цифровых документов доля
# кириллицы 0.95-1.00, у латинского OCR 0.00-0.60, промежуточных значений нет
MIN_CYRILLIC = 0.90
# нижняя граница плотности: отсекает сканы без текстового слоя и титульные листы
MIN_CHARS_PER_PAGE = 400


@dataclass
class PdfText:
    text: str
    pages: int
    chars: int
    cyrillic: float
    has_ocr_layer: bool

    @property
    def usable(self) -> bool:
        if not self.pages:
            return False
        return self.cyrillic >= MIN_CYRILLIC and self.chars / self.pages >= MIN_CHARS_PER_PAGE

    @property
    def verdict(self) -> str:
        if not self.pages:
            return "битый файл"
        if self.chars / self.pages < MIN_CHARS_PER_PAGE:
            return "скан без текстового слоя"
        if self.cyrillic < MIN_CYRILLIC:
            return "скан с латинским OCR"
        return "цифровой"


def _run(args: list[str]) -> str:
    return subprocess.run(args, capture_output=True).stdout.decode("utf-8", "replace")


def extract(path: str, layout: bool = False) -> PdfText:
    """Извлечь текст и посчитать признаки пригодности.

    layout=False намеренно по умолчанию: «Вестник Банка России» набран в две
    колонки, и -layout склеивает их построчно, перемешивая два разных текста.
    Режим по умолчанию идёт в порядке чтения и колонки не путает.
    """
    args = ["pdftotext"] + (["-layout"] if layout else []) + [path, "-"]
    text = _run(args)
    info = _run(["pdfinfo", path])
    m = re.search(r"Pages:\s+(\d+)", info)
    pages = int(m.group(1)) if m else 0
    c, l = len(CYR.findall(text)), len(LAT.findall(text))
    return PdfText(
        text=text,
        pages=pages,
        chars=len(re.sub(r"\s+", " ", text).strip()),
        cyrillic=c / (c + l) if c + l else 0.0,
        # ABBYY подкладывает скрытый слой распознавания этим шрифтом
        has_ocr_layer="HiddenHorzOCR" in _run(["pdffonts", path]),
    )
