#!/usr/bin/env python3
"""Нарезка корпуса на чанки во всех конфигурациях, которые потом сравниваются.

Конфигурация — это стратегия нарезки, размер, перекрытие и наличие шапки
с названием акта и раздела. Базовая конфигурация отличается от каждой
из остальных ровно одним параметром: иначе по результату нельзя сказать,
что именно повлияло.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin.chunking import TokenRuler, chunk_act   # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ACTS = os.path.join(ROOT, "data", "corpus", "acts.jsonl")
OUTDIR = os.path.join(ROOT, "data", "chunks")
REPORT = os.path.join(ROOT, "docs", "raw", "build_chunks.txt")

# токенизатор общий у bge-m3 и multilingual-e5: обе модели построены на XLM-RoBERTa
TOKENIZER = "xlm-roberta-base"


@dataclass(frozen=True)
class Config:
    name: str
    strategy: str
    size: int
    overlap: float
    heading: bool
    note: str


BASE = Config("base", "structure", 512, 0.15, True, "базовая конфигурация")
GRID = [
    BASE,
    Config("size-256",  "structure", 256,  0.15, True,  "размер чанка: 256 вместо 512"),
    Config("size-1024", "structure", 1024, 0.15, True,  "размер чанка: 1024 вместо 512"),
    Config("overlap-0", "structure", 512,  0.0,  True,  "без перекрытия"),
    Config("by-length",  "length",   512,  0.15, True,  "нарезка по длине вместо структуры"),
    Config("no-heading", "structure", 512, 0.15, False, "без шапки с актом и разделом"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="собрать только названные конфигурации")
    ap.add_argument("--limit", type=int, default=0, help="ограничить число актов (для пробы)")
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)

    from transformers import AutoTokenizer
    ruler = TokenRuler(AutoTokenizer.from_pretrained(TOKENIZER))

    acts = [json.loads(l) for l in open(ACTS, encoding="utf-8")]
    if args.limit:
        acts = acts[:args.limit]
    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    say(f"актов в корпусе: {len(acts)}, символов {sum(len(a['text']) for a in acts) / 1e6:.1f} млн")
    say(f"символов на токен: {ruler.calibrate([a['text'] for a in acts[:40]]):.2f}")
    say()
    say(f"{'конфигурация':<12} {'чанков':>8} {'ток.медиана':>12} {'ток.макс':>9} {'МБ':>6}  примечание")

    configs = [c for c in GRID if not args.only or c.name in args.only]
    for cfg in configs:
        t0 = time.monotonic()
        path = os.path.join(OUTDIR, f"{cfg.name}.jsonl")
        n, lens = 0, []
        with open(path, "w", encoding="utf-8") as f:
            for act in acts:
                for ch in chunk_act(act, ruler, size=cfg.size, overlap_share=cfg.overlap,
                                    strategy=cfg.strategy, add_heading=cfg.heading):
                    f.write(json.dumps(ch.as_dict(), ensure_ascii=False) + "\n")
                    n += 1
                    lens.append(ruler.count(ch.text))
        over = sum(1 for x in lens if x > cfg.size)
        say(f"{cfg.name:<12} {n:>8} {statistics.median(lens):>12.0f} {max(lens):>9} "
            f"{os.path.getsize(path) / 1048576:>6.0f}  {cfg.note}"
            + (f"  ПРЕВЫШЕНИЙ РАЗМЕРА: {over}" if over else "")
            + f"  [{time.monotonic() - t0:.0f} с]")

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    say(f"\nчанки: data/chunks/, отчёт: docs/raw/build_chunks.txt")


if __name__ == "__main__":
    main()
