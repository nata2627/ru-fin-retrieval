"""Сверка нарезки в опыте с номерами актов.

Прогон считает выдачи, которые сравниваются с уже посчитанными. Если
нарезка на видеокарте разойдётся с той, по которой считались прежние,
идентификатор «акт#номер» не исчезнет, а начнёт указывать на другой
текст: выдачи окажутся несравнимыми, а эталон съедет молча. Поэтому
расхождение обязано быть отказом считать, и это проверяется здесь,
а не на карте.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kaggle"))

import run_variants as V  # noqa: E402


def фрагменты(*имена: str) -> list[dict]:
    return [{"chunk_id": и, "text": "текст"} for и in имена]


def файл(tmp_path, *имена: str) -> str:
    путь = tmp_path / "ids_base.txt"
    путь.write_text("\n".join(имена) + "\n", encoding="utf-8")
    return str(путь)


def test_sovpadenie_prohodit(tmp_path):
    итог = V.сверить_идентификаторы(фрагменты("а#1", "а#2"),
                                    файл(tmp_path, "а#1", "а#2"))
    assert итог == {"фрагментов": 2, "сверено": True}


def test_sdvig_granic_otkaz(tmp_path):
    """Тот же счёт фрагментов, но границы съехали — считать нельзя."""
    with pytest.raises(SystemExit) as e:
        V.сверить_идентификаторы(фрагменты("а#1", "а#3"),
                                 файл(tmp_path, "а#1", "а#2"))
    assert "расхождение на месте 1" in str(e.value)


def test_drugoy_schet_otkaz(tmp_path):
    with pytest.raises(SystemExit) as e:
        V.сверить_идентификаторы(фрагменты("а#1"), файл(tmp_path, "а#1", "а#2"))
    assert "здесь 1 фрагментов, там 2" in str(e.value)


def test_bez_etalona_schitaem_no_govorim(tmp_path, capsys):
    """Без файла сверки прогон идёт, но молчать об этом нельзя."""
    итог = V.сверить_идентификаторы(фрагменты("а#1"), None)
    assert итог == {"фрагментов": 1, "сверено": False}
    assert "сверить нарезку не с чем" in capsys.readouterr().out
