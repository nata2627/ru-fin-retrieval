#!/usr/bin/env python3
"""Публикация набора на HuggingFace в формате MTEB/BeIR.

Набор — главный продукт проекта, и лежать он должен там, где его можно взять
чужим кодом. Формат BeIR выбран потому, что в нём корпус, запросы и разметка
разложены тремя файлами с устоявшимися именами полей, и таск MTEB
(`mteb_task/`) читает их без переписывания.

Скрипт ничего не считает: он проверяет, что выгрузка на диске непротиворечива,
и заливает её. Проверки перед заливкой не формальность — опубликованный набор
с эталоном на несуществующий фрагмент хуже, чем ненапубликованный.

Токен берётся из переменной окружения `HF_TOKEN` или из `huggingface-cli login`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin.benchmark import read_jsonl, read_qrels  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCH = os.path.join(ROOT, "data", "benchmark")
DEFAULT_REPO = "nata2627/ru-fin-retrieval"


def check(bench: str) -> dict:
    """Сверить выгрузку саму с собой. Возвращает числа для отчёта."""
    corpus = {d["_id"] for d in read_jsonl(os.path.join(bench, "corpus.jsonl"))}
    queries = {d["_id"] for d in read_jsonl(os.path.join(bench, "queries.jsonl"))}
    беды: list[str] = []
    всего_пар = 0
    for name in sorted(os.listdir(os.path.join(bench, "qrels"))):
        qrels = read_qrels(os.path.join(bench, "qrels", name))
        всего_пар += sum(len(r) for r in qrels.values())
        нет_запроса = set(qrels) - queries
        нет_фрагмента = {c for rel in qrels.values() for c in rel} - corpus
        if нет_запроса:
            беды.append(f"{name}: разметка на {len(нет_запроса)} запросов, "
                        f"которых нет в queries.jsonl")
        if нет_фрагмента:
            беды.append(f"{name}: разметка на {len(нет_фрагмента)} фрагментов, "
                        f"которых нет в corpus.jsonl")
        пустые = [q for q, rel in qrels.items() if not any(s > 0 for s in rel.values())]
        if пустые:
            беды.append(f"{name}: {len(пустые)} запросов без единого релевантного")
    if беды:
        print("выгрузка противоречива, публиковать нельзя:")
        for b in беды:
            print("   " + b)
        raise SystemExit(1)
    return {"фрагментов": len(corpus), "запросов": len(queries), "пар": всего_пар}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--bench", default=BENCH)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="только проверки, без заливки")
    args = ap.parse_args()

    stats = check(args.bench)
    print("выгрузка непротиворечива: " +
          ", ".join(f"{k} {v}" for k, v in stats.items()))

    card = os.path.join(ROOT, "docs", "DATASET_CARD.md")
    if not os.path.exists(card):
        raise SystemExit("нет docs/DATASET_CARD.md — набор без карточки не публикуется")

    if args.dry_run:
        print("--dry-run: заливка пропущена")
        return

    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(args.repo, repo_type="dataset", private=args.private,
                    exist_ok=True)
    api.upload_folder(folder_path=args.bench, repo_id=args.repo,
                      repo_type="dataset", path_in_repo=".")
    api.upload_file(path_or_fileobj=card, path_in_repo="README.md",
                    repo_id=args.repo, repo_type="dataset")
    print(f"опубликовано: https://huggingface.co/datasets/{args.repo}")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
