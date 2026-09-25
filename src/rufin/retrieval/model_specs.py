"""Описания эмбеддеров: путь, префиксы, размерность.

Вынесено отдельным модулем без тяжёлых зависимостей, потому что этот же файл
уезжает на Kaggle вместе с ноутбуком индексации. Префиксы должны совпадать
при построении индекса и при поиске — если они разойдутся, качество упадёт,
а причина будет неочевидна. Один источник правды исключает такое расхождение.

Соглашения сверены по карточкам моделей, а не взяты по памяти:
  bge-m3, USER-bge-m3   префиксы не используются;
  multilingual-e5-*     «query: » и «passage: »;
  FRIDA                 «search_query: » и «search_document: »;
  Qwen3-Embedding       запрос оборачивается в инструкцию, документ — без префикса.
"""
from __future__ import annotations

from dataclasses import dataclass

# описание задачи для Qwen3-Embedding: модель учитывает инструкцию в запросе
QWEN_TASK = ("Given a question about Russian financial regulation, retrieve the "
             "fragment of the Bank of Russia normative act that answers it")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: str
    query_prefix: str = ""
    passage_prefix: str = ""
    max_seq_length: int = 512
    dim: int = 0
    note: str = ""


MODELS: dict[str, ModelSpec] = {
    "bge-m3": ModelSpec(
        "bge-m3", "BAAI/bge-m3", dim=1024,
        note="рабочая лошадка, префиксы не нужны"),
    "e5-large": ModelSpec(
        "e5-large", "intfloat/multilingual-e5-large",
        query_prefix="query: ", passage_prefix="passage: ", dim=1024,
        note="без префиксов заметно хуже — проверяется отдельной абляцией"),
    "e5-small": ModelSpec(
        "e5-small", "intfloat/multilingual-e5-small",
        query_prefix="query: ", passage_prefix="passage: ", dim=384,
        note="118M параметров: на ней считаются абляции по нарезке, где важно "
             "сравнить конфигурации между собой, а не выжать максимум качества"),
    "qwen3-0.6b": ModelSpec(
        "qwen3-0.6b", "Qwen/Qwen3-Embedding-0.6B",
        query_prefix=f"Instruct: {QWEN_TASK}\nQuery:", passage_prefix="", dim=1024,
        note="вектор можно обрезать без переобучения — даёт абляцию по размерности"),
    "user-bge-m3": ModelSpec(
        "user-bge-m3", "deepvk/USER-bge-m3", dim=1024,
        note="bge-m3, дообученная на русском; префиксы не используются"),
    "frida": ModelSpec(
        "frida", "ai-forever/FRIDA",
        query_prefix="search_query: ", passage_prefix="search_document: ", dim=1536,
        note="русскоязычная, своя схема префиксов"),
    "rosberta": ModelSpec(
        "rosberta", "ai-forever/ru-en-RoSBERTa",
        query_prefix="search_query: ", passage_prefix="search_document: ", dim=1024,
        note="русскоязычная, запасной вариант"),
}

# модели основной таблицы и модель, на которой считаются абляции по нарезке
HEADLINE = ("bge-m3", "e5-large", "qwen3-0.6b", "user-bge-m3", "frida")
ABLATION = "e5-small"
