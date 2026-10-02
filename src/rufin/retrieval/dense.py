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

from dataclasses import dataclass, field

import numpy as np

from .model_specs import MODELS, ModelSpec  # noqa: F401  (переэкспорт)


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
    """Обёртка над моделью. Половинная точность включена по умолчанию на MPS:
    замер на этой машине дал 4.9 чанка в секунду против 2.9 при float32,
    а размер батча выше 4 только замедляет — восемь гигабайт памяти общие
    с системой, и крупный батч упирается в них (docs/raw/build_index.txt).
    """

    def __init__(self, spec: ModelSpec, device: str | None = None, batch_size: int = 4,
                 half: bool | None = None):
        import torch
        from sentence_transformers import SentenceTransformer
        self.spec = spec
        self.device = device or pick_device()
        self.batch_size = batch_size
        use_half = (self.device in ("mps", "cuda")) if half is None else half
        kwargs = {"torch_dtype": torch.float16} if use_half else {}
        self.model = SentenceTransformer(spec.path, device=self.device, model_kwargs=kwargs)
        self.model.max_seq_length = spec.max_seq_length
        self.half = use_half

    def encode(self, texts: list[str], *, is_query: bool, show_progress: bool = False) -> np.ndarray:
        prefix = self.spec.query_prefix if is_query else self.spec.passage_prefix
        if prefix:
            texts = [prefix + t for t in texts]
        vec = self.model.encode(texts, batch_size=self.batch_size, convert_to_numpy=True,
                                normalize_embeddings=True, show_progress_bar=show_progress)
        return truncate(vec.astype("float32"), self.spec.truncate_dim)


def truncate(vectors: np.ndarray, dim: int) -> np.ndarray:
    """Обрезать вектор до `dim` измерений и нормировать заново.

    Так работает матрёшка: модель обучена так, что первые 128 измерений
    сами по себе осмысленный вектор. Повторная нормировка обязательна —
    после обрезки длина уже не единица, а индекс считает косинус
    внутренним произведением. Без неё выдача изменится не из-за обрезки,
    а из-за разной длины векторов, и это будет тихо.
    """
    if not dim or dim >= vectors.shape[1]:
        return vectors
    cut = vectors[:, :dim]
    norm = np.linalg.norm(cut, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return (cut / norm).astype("float32")


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
