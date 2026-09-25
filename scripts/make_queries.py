#!/usr/bin/env python3
"""Генерация синтетических запросов к корпусу.

Для случайно отобранных фрагментов модель формулирует вопрос, ответ
на который содержится именно в этом фрагменте; фрагмент становится
эталонным ответом. Сгенерированное проходит через фильтры (см.
rufin/queryfilter.py): выбрасываются вопросы, списанные с фрагмента,
и вопросы, подходящие к половине корпуса.

Модель поднимается локально через llama.cpp и останавливается по окончании.
Никакой облачной генерации: у проекта одна машина, и всё должно
воспроизводиться на ней.

Это половина набора запросов. Вторая половина пишется руками — см.
docs/03-nabor-zaprosov.md.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import signal
import subprocess
import sys
import time

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin.queryfilter import IdfTable, is_good_source, judge_query, copy_score  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHUNKS = os.path.join(ROOT, "data", "chunks", "base.jsonl")
OUTDIR = os.path.join(ROOT, "data", "queries")
REPORT = os.path.join(ROOT, "docs", "raw", "make_queries.txt")

SYSTEM = (
    "Ты помогаешь составить набор проверочных вопросов к нормативным актам "
    "Банка России. Отвечай только самим вопросом, без пояснений и без кавычек."
)

PROMPT = """Ниже фрагмент нормативного акта Банка России.

--- начало фрагмента ---
{passage}
--- конец фрагмента ---

Сформулируй ОДИН вопрос, который в работе задал бы риск-аналитик, методолог
или бухгалтер банка, и ответ на который содержится именно в этом фрагменте.

Требования к вопросу:
- своими словами, без дословных кусков из фрагмента длиннее трёх слов подряд;
- конкретный: по нему должен находиться именно этот фрагмент, а не любой
  документ Банка России;
- не упоминай номер и название акта;
- одно предложение, заканчивается знаком вопроса.

Вопрос:"""


def start_server(model: str, port: int, ctx: int) -> subprocess.Popen:
    cmd = ["llama-server", "-m", model, "--port", str(port), "-c", str(ctx),
           "-ngl", "99", "--no-warmup", "--log-disable"]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(180):
        try:
            if requests.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                return proc
        except Exception:  # noqa: BLE001
            pass
        if proc.poll() is not None:
            raise RuntimeError("llama-server не запустился")
        time.sleep(1)
    proc.terminate()
    raise RuntimeError("llama-server не ответил за 180 секунд")


def ask(port: int, passage: str, temperature: float, seed: int) -> str:
    r = requests.post(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json={"messages": [{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": PROMPT.format(passage=passage)}],
              "temperature": temperature, "max_tokens": 120, "seed": seed},
        timeout=300)
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"].strip()
    text = re.sub(r"^[\s\"'«»\-–—]*", "", text).split("\n")[0].strip(" \"'«»")
    return text


def body_of(chunk: dict) -> str:
    """Текст чанка без приписанной шапки: вопрос задаётся по содержанию."""
    return chunk["text"].split("\n\n", 1)[-1] if "\n\n" in chunk["text"] else chunk["text"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=150, help="сколько запросов нужно")
    ap.add_argument("--model", default=os.path.expanduser("~/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"))
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--per-act", type=int, default=2, help="не больше вопросов на один акт")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)

    chunks = [json.loads(l) for l in open(CHUNKS, encoding="utf-8")]
    idf = IdfTable([c["text"] for c in chunks])
    good = [c for c in chunks if is_good_source(c)]
    random.Random(args.seed).shuffle(good)

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    say(f"чанков всего {len(chunks)}, пригодных как источник вопроса {len(good)} "
        f"({100 * len(good) / len(chunks):.0f}%)")
    say(f"модель: {os.path.basename(args.model)}")

    proc = start_server(args.model, args.port, args.ctx)
    say("модель поднята, генерация пошла")
    rejected = collections.Counter()
    per_act: dict[str, int] = collections.Counter()
    out: list[dict] = []
    t0 = time.monotonic()
    try:
        for i, ch in enumerate(good):
            if len(out) >= args.target:
                break
            if per_act[ch["act_id"]] >= args.per_act:
                continue
            passage = body_of(ch)[:4000]
            try:
                q = ask(args.port, passage, args.temperature, args.seed + i)
            except Exception as e:  # noqa: BLE001
                rejected["ошибка модели"] += 1
                say(f"   ошибка на фрагменте {ch['chunk_id']}: {e}")
                continue
            ok, why = judge_query(q, passage, idf)
            if not ok:
                rejected[why.split("(")[0].strip()] += 1
                continue
            run, share = copy_score(q, passage)
            per_act[ch["act_id"]] += 1
            out.append({
                "query_id": f"syn{len(out):04d}",
                "origin": "синтетический",
                "text": q,
                "gold_chunk_id": ch["chunk_id"],
                "act_id": ch["act_id"], "number": ch["number"], "date": ch["date"],
                "act_title": ch["title"], "section": ch["section"], "units": ch["units"],
                "mean_idf": round(idf.mean_idf(q), 2),
                "copy_run": run, "copy_share": round(share, 2),
            })
            if len(out) % 25 == 0:
                say(f"   принято {len(out)}/{args.target}, отклонено {sum(rejected.values())}, "
                    f"{time.monotonic() - t0:.0f} с")
    finally:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=30)

    path = os.path.join(OUTDIR, "synthetic.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for q in out:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    say()
    say(f"принято {len(out)}, отклонено {sum(rejected.values())} "
        f"({100 * sum(rejected.values()) / max(1, len(out) + sum(rejected.values())):.0f}% попыток)")
    say(f"причины отказа: {dict(rejected.most_common())}")
    say(f"актов затронуто: {len(per_act)}")
    say(f"время: {time.monotonic() - t0:.0f} с")
    say(f"файл: data/queries/synthetic.jsonl")
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
