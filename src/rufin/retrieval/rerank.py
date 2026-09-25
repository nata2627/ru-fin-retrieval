"""Переранжирование кросс-энкодером.

Кросс-энкодер читает запрос и документ вместе, поэтому качество выше
би-энкодера, но считать его можно только на коротком списке: пара «запрос —
чанк» прогоняется через модель целиком. Отсюда схема «достать 50 дешёвым
поиском, переупорядочить дорогой моделью».

Замеры задержки снимаются на маке через MPS — там, где система и работает.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from .dense import pick_device

# кросс-энкодер той же серии, что и bge-m3: многоязычный, знает русский
DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"


@dataclass
class Reranker:
    model_path: str = DEFAULT_MODEL
    device: str = ""
    batch_size: int = 8
    max_length: int = 512

    def __post_init__(self) -> None:
        from sentence_transformers import CrossEncoder
        self.device = self.device or pick_device()
        self.model = CrossEncoder(self.model_path, device=self.device, max_length=self.max_length)

    def rerank(self, query: str, candidates: list[tuple[str, float]],
               texts: dict[str, str], top: int = 10) -> tuple[list[tuple[str, float]], float]:
        """Переупорядочить кандидатов. Возвращает выдачу и затраченное время."""
        if not candidates:
            return [], 0.0
        t0 = time.monotonic()
        pairs = [(query, texts[c]) for c, _ in candidates]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        order = sorted(zip((c for c, _ in candidates), scores), key=lambda x: -float(x[1]))
        return [(c, float(s)) for c, s in order[:top]], time.monotonic() - t0
