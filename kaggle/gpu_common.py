"""Общее для всех тяжёлых шагов на Kaggle: загрузка корпуса и нарезка.

Нарезка выполняется тем же кодом (`rufin.chunking`) и тем же токенизатором,
что локально, поэтому идентификаторы фрагментов совпадают и матрицу
эмбеддингов можно сопоставить с локальным корпусом.

Чанки не передаются между этапами файлами: все нарезки вместе весят под два
гигабайта, а пересобрать их из acts.jsonl — вопрос нескольких минут.
"""
from __future__ import annotations

import gzip
import json
import time

from rufin.chunk_configs import BY_NAME
from rufin.chunking import TokenRuler, chunk_act

TOKENIZER = "xlm-roberta-base"


def make_ruler() -> TokenRuler:
    from transformers import AutoTokenizer
    return TokenRuler(AutoTokenizer.from_pretrained(TOKENIZER))


def load_acts(path: str, limit: int = 0) -> list[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        acts = [json.loads(l) for l in f if l.strip()]
    return acts[:limit] if limit else acts


def build_chunks(acts: list[dict], config: str, ruler: TokenRuler) -> list[dict]:
    cfg = BY_NAME[config]
    t0 = time.time()
    out: list[dict] = []
    for act in acts:
        for ch in chunk_act(act, ruler, size=cfg.size, overlap_share=cfg.overlap,
                            strategy=cfg.strategy, add_heading=cfg.heading):
            out.append(ch.as_dict())
    print(f"[нарезка {config}] {len(out)} фрагментов за {time.time() - t0:.0f} с", flush=True)
    return out


def pick_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def half_kwargs(device: str) -> dict:
    """Параметры половинной точности.

    Имя параметра у transformers менялось: раньше `torch_dtype`, теперь `dtype`.
    Версия на Kaggle заранее не известна, поэтому подходящее подбирается пробой,
    а не угадывается.
    """
    if device == "cpu":
        return {}
    import inspect
    import torch
    from transformers import AutoModel
    name = "dtype" if "dtype" in inspect.signature(
        AutoModel.from_pretrained).parameters else "torch_dtype"
    return {name: torch.float16}


def load_encoder(path: str, device: str, max_seq_length: int = 512):
    """Загрузить би-энкодер, при неудаче с половинной точностью — без неё."""
    from sentence_transformers import SentenceTransformer
    try:
        model = SentenceTransformer(path, device=device, trust_remote_code=True,
                                    model_kwargs=half_kwargs(device))
    except TypeError:
        model = SentenceTransformer(path, device=device, trust_remote_code=True)
    model.max_seq_length = max_seq_length
    return model


def load_cross_encoder(path: str, device: str, max_length: int = 512):
    """Загрузить кросс-энкодер, при неудаче с половинной точностью — без неё."""
    from sentence_transformers import CrossEncoder
    try:
        return CrossEncoder(path, device=device, max_length=max_length,
                            model_kwargs=half_kwargs(device))
    except TypeError:
        return CrossEncoder(path, device=device, max_length=max_length)
