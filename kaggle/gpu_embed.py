"""Эмбеддинги корпуса на видеокорте.

Матрица сохраняется в float16: для косинусной близости точности хватает,
а скачивать вдвое меньше. Посчитанное не пересчитывается, поэтому после
обрыва сессии достаточно запустить ячейку заново.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from rufin.retrieval.model_specs import MODELS


def encode(model_name: str, texts: list[str], batch_size: int, device: str) -> np.ndarray:
    import torch
    from gpu_common import load_encoder
    spec = MODELS[model_name]
    model = load_encoder(spec.path, device, spec.max_seq_length)
    if spec.passage_prefix:
        texts = [spec.passage_prefix + t for t in texts]
    vec = model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=True)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return vec.astype("float16")


def embed_config(chunks: list[dict], config: str, model_name: str, out_root: str,
                 batch_size: int, device: str) -> dict:
    out_dir = os.path.join(out_root, config, model_name)
    marker = os.path.join(out_dir, "vectors.npy")
    if os.path.exists(marker):
        print(f"[{config} / {model_name}] уже посчитано, пропуск", flush=True)
        return json.load(open(os.path.join(out_dir, "meta.json"), encoding="utf-8"))

    ids = [c["chunk_id"] for c in chunks]
    print(f"[{config} / {model_name}] {len(ids)} фрагментов, пошло", flush=True)
    t0 = time.time()
    vec = encode(model_name, [c["text"] for c in chunks], batch_size, device)
    seconds = time.time() - t0

    os.makedirs(out_dir, exist_ok=True)
    np.save(marker, vec)
    with open(os.path.join(out_dir, "ids.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(ids) + "\n")
    meta = {"model": model_name, "chunks_config": config, "chunks": len(ids),
            "dim": int(vec.shape[1]), "dtype": "float16", "device": device,
            "seconds": round(seconds, 1), "per_second": round(len(ids) / seconds, 1),
            "size_mb": round(vec.nbytes / 1048576, 1), "batch_size": batch_size}
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"[{config} / {model_name}] готово: {seconds:.0f} с "
          f"({meta['per_second']} фрагм./с), {meta['size_mb']:.0f} МБ", flush=True)
    return meta
