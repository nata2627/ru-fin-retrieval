"""Чтение и запись набора запросов в формате MTEB (BeIR).

Формат выбран не ради моды: в нём корпус, запросы и разметка лежат тремя
отдельными файлами с устоявшимися именами полей, и готовый бенчмарк можно
отдать наружу или прогнать чужим кодом, ничего не переписывая.

    corpus.jsonl     {"_id": ..., "title": ..., "text": ...}
    queries.jsonl    {"_id": ..., "text": ...}
    qrels/test.tsv   query-id <таб> corpus-id <таб> оценка

Оценки: 2 — фрагмент прямо отвечает на вопрос, 1 — относится к делу,
но ответа не содержит, 0 — не относится. Градации нужны для NDCG:
без них метрика вырождается в бинарную.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict


def write_corpus(path: str, chunks: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            title = f"{c['number']} от {c['date']}. {c['title']}"
            if c.get("section"):
                title += f" // {c['section']}"
            f.write(json.dumps({"_id": c["chunk_id"], "title": title, "text": c["text"]},
                               ensure_ascii=False) + "\n")


def write_queries(path: str, queries: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps({"_id": q["query_id"], "text": q["text"]},
                               ensure_ascii=False) + "\n")


def write_qrels(path: str, qrels: dict[str, dict[str, int]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for qid in sorted(qrels):
            for cid, score in sorted(qrels[qid].items()):
                if score > 0:
                    f.write(f"{qid}\t{cid}\t{score}\n")


def read_qrels(path: str) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        header = f.readline()
        if not header.startswith("query-id"):
            f.seek(0)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            # Столбцов может быть больше трёх: у разметки по названным пунктам
            # есть четвёртый — основание. Требовать ровно три значило бы молча
            # прочитать такой файл как пустой.
            if len(parts) < 3 or not parts[2].strip().lstrip("-").isdigit():
                continue
            out[parts[0]][parts[1]] = int(parts[2])
    return dict(out)


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]
