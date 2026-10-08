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


class MeanPoolEncoder:
    """То же самое через `transformers` напрямую: токенизатор, усреднение
    по токенам с маской, нормировка.

    Нужен для весов, сохранённых обучением на Kaggle. В `modules.json`
    там записаны имена классов той версии `sentence-transformers`, которая
    стояла в сессии («sentence_transformers.base.modules.transformer»),
    и локальная версия 3.0.1 их не находит: ModuleNotFoundError на пустом
    месте. Обёртка над моделью при этом не делает ничего, кроме среднего
    по маске и нормировки, — ровно то, что здесь написано руками.

    Второй довод тот же, что в `kaggle/gpu_traineval.py`: загрузка через
    `sentence-transformers` дважды уводила прогон в молчаливое зависание.
    Путь без неё проверен и на видеокарте, и здесь.
    """

    def __init__(self, spec: ModelSpec, device: str | None = None,
                 batch_size: int = 8):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.spec = spec
        self.device = device or pick_device()
        self.batch_size = batch_size
        self.half = False
        self.tok = AutoTokenizer.from_pretrained(spec.path, use_fast=True,
                                                 local_files_only=True)
        self.model = AutoModel.from_pretrained(spec.path, local_files_only=True)
        self.model = self.model.to(self.device).eval()
        self.torch = torch

    def encode(self, texts: list[str], *, is_query: bool,
               show_progress: bool = False) -> np.ndarray:
        torch = self.torch
        prefix = self.spec.query_prefix if is_query else self.spec.passage_prefix
        if prefix:
            texts = [prefix + t for t in texts]
        out = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = self.tok(texts[start:start + self.batch_size],
                                 padding=True, truncation=True,
                                 max_length=self.spec.max_seq_length,
                                 return_tensors="pt").to(self.device)
                hidden = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                out.append(torch.nn.functional.normalize(vec, p=2, dim=1)
                           .float().cpu().numpy())
                if show_progress:
                    print(f"   {start + len(batch['input_ids'])}/{len(texts)}",
                          flush=True)
        return truncate(np.vstack(out), self.spec.truncate_dim)


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


def top_k(vectors: np.ndarray, qvec: np.ndarray, k: int) -> list[list[tuple[int, float]]]:
    """Точный поиск по нормированным векторам на numpy.

    Тот же косинус, что в FAISS, но без FAISS. Нужен там, где в одном
    процессе работают и модель, и индекс: на маке FAISS и torch приносят
    каждый свою сборку libomp, и процесс падает по SIGSEGV — без ошибки
    и без traceback, просто код выхода 139. На шестидесяти тысячах
    фрагментов и сотнях запросов перебор занимает доли секунды, так что
    терять тут нечего.
    """
    scores = (qvec.astype("float32") @ vectors.astype("float32").T)
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    out = []
    for row, idx in zip(scores, part):
        order = idx[np.argsort(-row[idx])]
        out.append([(int(j), float(row[j])) for j in order])
    return out


@dataclass
class DenseIndex:
    ids: list[str]
    vectors: np.ndarray
    spec: ModelSpec
    index: object = field(default=None, repr=False)
    build_seconds: float = 0.0
    backend: str = "faiss"

    def __post_init__(self) -> None:
        if self.backend == "numpy":
            return
        if self.index is None:
            import faiss
            self.index = faiss.IndexFlatIP(self.vectors.shape[1])
            self.index.add(self.vectors)

    @classmethod
    def from_files(cls, spec: ModelSpec, vectors_path: str, ids_path: str,
                   backend: str = "faiss") -> "DenseIndex":
        vec = np.load(vectors_path)
        ids = [l.rstrip("\n") for l in open(ids_path, encoding="utf-8")]
        if len(ids) != vec.shape[0]:
            raise ValueError(f"идентификаторов {len(ids)}, векторов {vec.shape[0]}")
        return cls(ids=ids, vectors=vec, spec=spec, backend=backend)

    @property
    def size_mb(self) -> float:
        return self.vectors.nbytes / 1048576

    def search_vectors(self, qvec: np.ndarray, k: int = 10) -> list[list[tuple[str, float]]]:
        if self.index is None:
            return [[(self.ids[j], s) for j, s in row]
                    for row in top_k(self.vectors, qvec, k)]
        scores, idx = self.index.search(qvec, k)
        return [[(self.ids[j], float(s)) for j, s in zip(row_i, row_s) if j >= 0]
                for row_i, row_s in zip(idx, scores)]
