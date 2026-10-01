#!/usr/bin/env python3
"""Проверка перед прогоном на Kaggle: всё ли на месте.

Каждый пункт — грабли, на которые проект уже наступал. Дешевле проверить
за полминуты, чем потерять час на прогоне, который упадёт на первой строке.
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ok_all = True


def check(name: str, ok: bool, detail: str = "") -> None:
    global ok_all
    ok_all = ok_all and ok
    print(f"  {'OK  ' if ok else 'НЕТ '} {name:<46} {detail}")


def kaggle(*args: str) -> str:
    r = subprocess.run(["kaggle", *args], text=True, capture_output=True)
    return (r.stdout or "") + (r.stderr or "")


def main() -> None:
    print("=== локально ===")
    chunks = os.path.join(ROOT, "data", "chunks", "base.jsonl")
    n = sum(1 for _ in open(chunks, encoding="utf-8")) if os.path.exists(chunks) else 0
    check("нарезка base", n == 62594, f"{n} фрагментов")

    q = os.path.join(ROOT, "data", "queries", "queries.jsonl")
    nq = sum(1 for _ in open(q, encoding="utf-8")) if os.path.exists(q) else 0
    # Число не закрепляется: набор растёт на каждом этапе (211 на этапе A,
    # 486 после разметки живых вопросов, около тысячи после генерации dev
    # и теста). Закреплённая цифра превращает проверку в ложную тревогу
    # ровно тогда, когда работа идёт, поэтому проверяется только то, что
    # набор не пустой и не усох.
    check("набор запросов", nq >= 211, f"{nq} запросов")

    runs = os.path.join(ROOT, "data", "runs")
    have = sorted(os.listdir(runs)) if os.path.isdir(runs) else []
    check("выдачи уже есть", len(have) >= 3, ", ".join(x.replace("base__", "") for x in have))

    print("\n=== код, который уедет на Kaggle ===")
    gs = open(os.path.join(ROOT, "kaggle", "gpu_search.py"), encoding="utf-8").read()
    check("будильник поверх повторов HuggingFace", "class ModelTimeout(BaseException)" in gs)
    check("ограничение ожидания HuggingFace", "HF_HUB_DOWNLOAD_TIMEOUT" in gs)
    rb = open(os.path.join(ROOT, "kaggle", "run_phase_b.py"), encoding="utf-8").read()
    check("сверка нарезки с этапом A", "check_same_chunking" in rb)
    check("выбор нарезок ключом --only", '"--only"' in rb)
    rc = open(os.path.join(ROOT, "kaggle", "run_phase_c.py"), encoding="utf-8").read()
    check("сверка нарезки в этапе C", "нарезка разошлась с канонической" in rc)
    check("сверка нарезки со сплитом", "нарезка разошлась со сплитом" in rc)
    gg = open(os.path.join(ROOT, "kaggle", "gpu_gen.py"), encoding="utf-8").read()
    check("vLLM поднимается через spawn", "VLLM_WORKER_MULTIPROC_METHOD" in gg)
    gc = open(os.path.join(ROOT, "kaggle", "gpu_common.py"), encoding="utf-8").read()
    check("поиск входов по дереву", "def find_file" in gc and "def find_embeddings" in gc)
    kr = open(os.path.join(ROOT, "scripts", "kaggle_run.py"), encoding="utf-8").read()
    check("токен HuggingFace в ноутбуке", "UserSecretsClient" in kr)
    check("источник кода ранжируется", "stale" in kr)
    check("конфигурация узла задаётся", "machine_shape" in kr)

    print("\n=== на Kaggle ===")
    ds = kaggle("datasets", "files", "naumaaaa/ru-fin-retrieval")
    check("датасет с кодом и корпусом", "acts.jsonl" in ds and "gpu_search.py" in ds)
    qs = kaggle("datasets", "files", "naumaaaa/ru-fin-queries")
    check("датасет с запросами", "queries.jsonl" in qs)
    st = kaggle("kernels", "status", "naumaaaa/ru-fin")
    check("этап A с эмбеддингами цел", "COMPLETE" in st, st.strip()[:48])

    print()
    print("ГОТОВО К ЗАПУСКУ" if ok_all else "ЕСТЬ ПРОБЛЕМЫ, ЗАПУСКАТЬ НЕЛЬЗЯ")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
