"""Плотный поиск: эмбеддинги чанков и поиск по ним через FAISS.

Про префиксы. Модели семейства e5 обучены с пометками «query:» и «passage:»,
и без них качество проседает на ровном месте — это самая частая ошибка при
их использовании. bge-m3 префиксов не требует, ru-en-RoSBERTa использует свои
(«search_query:» / «search_document:»). Поэтому префиксы описаны в свойствах
модели, а не проставляются руками на месте вызова.

Индекс — FAISS с внутренним произведением по нормированным векторам, то есть
косинус. На 26 тысячах чанков это точный перебор, никаких приближений:
матрица в сотню мегабайт, поиск занимает единицы миллисекунд.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: str
    query_prefix: str = ""
    passage_prefix: str = ""
    max_seq_length: int = 512
    note: str = ""


MODELS = {
    "bge-m3": ModelSpec(
        "bge-m3", "BAAI/bge-m3",
        note="префиксы не нужны"),
    "e5-large": ModelSpec(
        "e5-large", "intfloat/multilingual-e5-large",
        query_prefix="query: ", passage_prefix="passage: ",
        note="без префиксов заметно хуже, проверяется абляцией"),
    "rosberta": ModelSpec(
        "rosberta", "ai-forever/ru-en-RoSBERTa",
        query_prefix="search_query: ", passage_prefix="search_document: ",
        note="русскоязычная, меньше и быстрее"),
}


def pick_device() -> str:
    """MPS на маке, иначе CPU. Устройство пишется в отчёт: замеры времени
    без указания железа сравнивать нельзя."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


class Encoder:
    def __init__(self, spec: ModelSpec, device: str | None = None, batch_size: int = 8):
        from sentence_transformers import SentenceTransformer
        self.spec = spec
        self.device = device or pick_device()
        self.batch_size = batch_size
        self.model = SentenceTransformer(spec.path, device=self.device)
        self.model.max_seq_length = spec.max_seq_length

    def encode(self, texts: list[str], *, is_query: bool, show_progress: bool = False) -> np.ndarray:
        prefix = self.spec.query_prefix if is_query else self.spec.passage_prefix
        if prefix:
            texts = [prefix + t for t in texts]
        vec = self.model.encode(texts, batch_size=self.batch_size, convert_to_numpy=True,
                                normalize_embeddings=True, show_progress_bar=show_progress)
        return vec.astype("float32")


@dataclass
class DenseIndex:
    ids: list[str]
    vectors: np.ndarray
    spec: ModelSpec
    index: object = field(default=None, repr=False)
    build_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.index is None:
            import faiss
            self.index = faiss.IndexFlatIP(self.vectors.shape[1])
            self.index.add(self.vectors)

    @classmethod
    def from_files(cls, spec: ModelSpec, vectors_path: str, ids_path: str) -> "DenseIndex":
        vec = np.load(vectors_path)
        ids = [l.rstrip("\n") for l in open(ids_path, encoding="utf-8")]
        if len(ids) != vec.shape[0]:
            raise ValueError(f"идентификаторов {len(ids)}, векторов {vec.shape[0]}")
        return cls(ids=ids, vectors=vec, spec=spec)

    @property
    def size_mb(self) -> float:
        return self.vectors.nbytes / 1048576

    def search_vectors(self, qvec: np.ndarray, k: int = 10) -> list[list[tuple[str, float]]]:
        scores, idx = self.index.search(qvec, k)
        return [[(self.ids[j], float(s)) for j, s in zip(row_i, row_s) if j >= 0]
                for row_i, row_s in zip(idx, scores)]


def encode_corpus(spec: ModelSpec, texts: list[str], ids: list[str], out_dir: str,
                  batch_size: int = 8, device: str | None = None) -> dict:
    """Посчитать эмбеддинги корпуса и сохранить матрицу рядом с идентификаторами.

    Матрица кладётся файлом, чтобы поиск потом работал без пересчёта
    и без видеокарты: индексация — разовая операция.
    """
    os.makedirs(out_dir, exist_ok=True)
    enc = Encoder(spec, device=device, batch_size=batch_size)
    t0 = time.monotonic()
    vec = enc.encode(texts, is_query=False, show_progress=True)
    seconds = time.monotonic() - t0
    np.save(os.path.join(out_dir, "vectors.npy"), vec)
    with open(os.path.join(out_dir, "ids.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(ids) + "\n")
    return {"model": spec.name, "device": enc.device, "chunks": len(ids),
            "dim": int(vec.shape[1]), "seconds": round(seconds, 1),
            "per_second": round(len(ids) / seconds, 1) if seconds else 0.0,
            "size_mb": round(vec.nbytes / 1048576, 1)}
