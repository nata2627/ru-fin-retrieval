"""Конфигурации нарезки, которые сравниваются в абляциях.

Отдельный модуль без тяжёлых зависимостей: этот же файл уезжает на Kaggle
вместе с ноутбуком индексации. Разойдись параметры нарезки между локальным
прогоном и удалённым — идентификаторы фрагментов не совпадут, и матрицу
эмбеддингов нельзя будет сопоставить с корпусом.

Базовая конфигурация отличается от каждой из остальных ровно одним
параметром: иначе по результату нельзя сказать, что именно повлияло.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkConfig:
    name: str
    strategy: str
    size: int
    overlap: float
    heading: bool
    note: str


BASE = ChunkConfig("base", "structure", 512, 0.15, True, "базовая конфигурация")

GRID: tuple[ChunkConfig, ...] = (
    BASE,
    ChunkConfig("size-256",   "structure", 256,  0.15, True,  "размер чанка: 256 вместо 512"),
    ChunkConfig("size-1024",  "structure", 1024, 0.15, True,  "размер чанка: 1024 вместо 512"),
    ChunkConfig("overlap-0",  "structure", 512,  0.0,  True,  "без перекрытия"),
    ChunkConfig("by-length",  "length",    512,  0.15, True,  "нарезка по длине вместо структуры"),
    ChunkConfig("no-heading", "structure", 512,  0.15, False, "без шапки с актом и разделом"),
)

BY_NAME = {c.name: c for c in GRID}
