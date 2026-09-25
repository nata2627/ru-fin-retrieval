#!/usr/bin/env python3
"""Индексация корпуса на Kaggle: чанки -> матрицы эмбеддингов.

Запускается в ноутбуке Kaggle с включённым GPU. На вход — acts.jsonl(.gz)
из проекта, на выход — по матрице на каждую пару «нарезка + модель».

Почему чанки пересобираются здесь, а не загружаются готовыми: файлы нарезок
весят под два гигабайта, а acts.jsonl в сжатом виде — двадцать мегабайт.
Нарезка выполняется тем же кодом (chunking.py лежит рядом) и тем же
токенизатором, поэтому идентификаторы фрагментов совпадут с локальными.

Матрицы сохраняются в float16: для косинусной близости точности хватает,
а скачивать обратно вдвое меньше.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import time

import numpy as np

from chunking import TokenRuler, chunk_act
from model_specs import ABLATION, HEADLINE, MODELS

TOKENIZER = "xlm-roberta-base"

# те же конфигурации нарезки, что и в scripts/build_chunks.py
CONFIGS = {
    "base":       dict(strategy="structure", size=512,  overlap=0.15, heading=True),
    "size-256":   dict(strategy="structure", size=256,  overlap=0.15, heading=True),
    "size-1024":  dict(strategy="structure", size=1024, overlap=0.15, heading=True),
    "overlap-0":  dict(strategy="structure", size=512,  overlap=0.0,  heading=True),
    "by-length":  dict(strategy="length",    size=512,  overlap=0.15, heading=True),
    "no-heading": dict(strategy="structure", size=512,  overlap=0.15, heading=False),
}


def load_acts(path: str) -> list[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def build_chunks(acts: list[dict], cfg: dict, ruler: TokenRuler) -> tuple[list[str], list[str]]:
    ids, texts = [], []
    for act in acts:
        for ch in chunk_act(act, ruler, size=cfg["size"], overlap_share=cfg["overlap"],
                            strategy=cfg["strategy"], add_heading=cfg["heading"]):
            ids.append(ch.chunk_id)
            texts.append(ch.text)
    return ids, texts


def pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def encode(model_name: str, texts: list[str], batch_size: int, device: str) -> np.ndarray:
    import torch
    from sentence_transformers import SentenceTransformer
    spec = MODELS[model_name]
    kwargs = {"torch_dtype": torch.float16} if device != "cpu" else {}
    model = SentenceTransformer(spec.path, device=device, trust_remote_code=True,
                                model_kwargs=kwargs)
    model.max_seq_length = spec.max_seq_length
    texts = [spec.passage_prefix + t for t in texts] if spec.passage_prefix else texts
    vec = model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=True)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return vec.astype("float16")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="/kaggle/input/ru-fin-retrieval/acts.jsonl.gz")
    ap.add_argument("--out", default="/kaggle/working/embeddings")
    ap.add_argument("--headline-models", nargs="*", default=list(HEADLINE))
    ap.add_argument("--ablation-model", default=ABLATION)
    ap.add_argument("--ablation-configs", nargs="*",
                    default=[c for c in CONFIGS if c != "base"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit-acts", type=int, default=0, help="для пробного прогона")
    ap.add_argument("--device", default=None, help="cuda на Kaggle; определяется само")
    args = ap.parse_args()

    device = args.device or pick_device()
    print(f"устройство: {device}", flush=True)

    from transformers import AutoTokenizer
    ruler = TokenRuler(AutoTokenizer.from_pretrained(TOKENIZER))

    acts = load_acts(args.acts)
    if args.limit_acts:
        acts = acts[:args.limit_acts]
    print(f"актов: {len(acts)}, символов {sum(len(a['text']) for a in acts) / 1e6:.1f} млн",
          flush=True)

    # что считаем: основная таблица на базовой нарезке + абляции на одной модели
    plan = [("base", m) for m in args.headline_models]
    plan += [(c, args.ablation_model) for c in args.ablation_configs]
    plan += [("base", args.ablation_model)]      # опорная точка для абляций

    chunk_cache: dict[str, tuple[list[str], list[str]]] = {}
    report = []
    for cfg_name, model_name in plan:
        if cfg_name not in chunk_cache:
            t0 = time.time()
            chunk_cache[cfg_name] = build_chunks(acts, CONFIGS[cfg_name], ruler)
            print(f"[нарезка {cfg_name}] {len(chunk_cache[cfg_name][0])} чанков "
                  f"за {time.time() - t0:.0f} с", flush=True)
        ids, texts = chunk_cache[cfg_name]
        out_dir = os.path.join(args.out, cfg_name, model_name)
        if os.path.exists(os.path.join(out_dir, "vectors.npy")):
            print(f"[{cfg_name} / {model_name}] уже посчитано, пропуск", flush=True)
            continue
        print(f"[{cfg_name} / {model_name}] {len(ids)} чанков, пошло", flush=True)
        t0 = time.time()
        try:
            vec = encode(model_name, texts, args.batch_size, device)
        except Exception as e:  # noqa: BLE001
            print(f"[{cfg_name} / {model_name}] ОШИБКА: {e}", flush=True)
            report.append({"config": cfg_name, "model": model_name, "error": str(e)[:200]})
            continue
        seconds = time.time() - t0
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "vectors.npy"), vec)
        with open(os.path.join(out_dir, "ids.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(ids) + "\n")
        meta = {"model": model_name, "chunks_config": cfg_name, "chunks": len(ids),
                "dim": int(vec.shape[1]), "dtype": "float16", "device": device,
                "seconds": round(seconds, 1), "per_second": round(len(ids) / seconds, 1),
                "size_mb": round(vec.nbytes / 1048576, 1), "batch_size": args.batch_size}
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        print(f"[{cfg_name} / {model_name}] готово: {seconds:.0f} с "
              f"({meta['per_second']} чанк/с), {meta['size_mb']:.0f} МБ", flush=True)
        report.append(meta)

    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    total = sum(r.get("size_mb", 0) for r in report)
    print(f"\nсчитано {len([r for r in report if 'error' not in r])} матриц, "
          f"суммарно {total:.0f} МБ -> {args.out}")


if __name__ == "__main__":
    main()
