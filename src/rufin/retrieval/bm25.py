"""Базовая линия: BM25 по фрагментам корпуса.

Без неё любые цифры плотного поиска не с чем сравнивать.

Реализация своя, на разреженной матрице, а не на rank_bm25. Причина —
память: rank_bm25 хранит все токенизированные документы списками строк,
и на шестидесяти тысячах фрагментов это два-три гигабайта. На машине
с восемью гигабайтами, где заодно надо держать матрицу эмбеддингов,
этого достаточно, чтобы система ушла в подкачку. Разреженная матрица
частот на тех же данных занимает около ста мегабайт, а поиск идёт быстрее:
вместо перебора документов складываются вклады только тех, где слово
запроса вообще встречается.

"""
from __future__ import annotations

import time
from array import array
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csc_matrix

from .text import tokenize

K1 = 1.5
B = 0.75
# Доля среднего IDF, которой заменяются отрицательные значения. У Okapi BM25
# IDF слова, встречающегося более чем в половине документов, отрицателен,
# и без замены такое слово начинает работать против документов, где оно есть.
EPSILON = 0.25


@dataclass
class BM25Index:
    ids: list[str]
    vocab: dict[str, int]
    matrix: csc_matrix               # слова x документы, значения — частоты
    idf: np.ndarray
    doc_norm: np.ndarray             # k1 * (1 - b + b * |d| / avgdl)
    build_seconds: float = 0.0
    _scores: np.ndarray = field(default=None, repr=False)

    @classmethod
    def build(cls, chunk_ids: list[str], texts: list[str], k1: float = K1,
              b: float = B) -> "BM25Index":
        t0 = time.monotonic()
        vocab: dict[str, int] = {}
        # Компактные массивы, а не списки питона: у корпуса около двенадцати
        # миллионов ненулевых частот, и списки целых заняли бы под гигабайт
        # только на время сборки. В array.array числа лежат по четыре байта.
        rows = array("i")
        cols = array("i")
        vals = array("f")
        lengths = np.zeros(len(texts), dtype=np.float32)

        for col, text in enumerate(texts):
            counts: dict[int, int] = {}
            n = 0
            for tok in tokenize(text):
                idx = vocab.get(tok)
                if idx is None:
                    idx = vocab[tok] = len(vocab)
                counts[idx] = counts.get(idx, 0) + 1
                n += 1
            lengths[col] = n
            for idx, cnt in counts.items():
                rows.append(idx)
                cols.append(col)
                vals.append(cnt)

        matrix = csc_matrix((np.frombuffer(vals, dtype=np.float32),
                             (np.frombuffer(rows, dtype=np.int32),
                              np.frombuffer(cols, dtype=np.int32))),
                            shape=(len(vocab), len(texts)))
        # число документов, содержащих слово
        df = np.diff(matrix.tocsr().indptr).astype(np.float32)
        n_docs = len(texts)
        # Формула Okapi в том виде, в каком её считает rank_bm25: без прибавления
        # единицы под логарифмом. Вариант Lucene (log(...+1)) даёт другие числа,
        # и базовая линия перестала бы совпадать с общепринятой.
        idf = (np.log(n_docs - df + 0.5) - np.log(df + 0.5)).astype(np.float32)
        if idf.size:
            idf = np.where(idf < 0, EPSILON * float(idf.mean()), idf).astype(np.float32)
        avgdl = float(lengths.mean()) if n_docs else 0.0
        doc_norm = (k1 * (1.0 - b + b * lengths / (avgdl or 1.0))).astype(np.float32)

        return cls(ids=list(chunk_ids), vocab=vocab, matrix=matrix.tocsr(), idf=idf,
                   doc_norm=doc_norm, build_seconds=time.monotonic() - t0)

    def scores(self, query: str, k1: float = K1) -> np.ndarray:
        """Оценки по всем документам. Буфер переиспользуется между запросами."""
        if self._scores is None or self._scores.shape[0] != len(self.ids):
            self._scores = np.zeros(len(self.ids), dtype=np.float32)
        out = self._scores
        out.fill(0.0)
        for tok in tokenize(query):
            idx = self.vocab.get(tok)
            if idx is None:
                continue
            start, end = self.matrix.indptr[idx], self.matrix.indptr[idx + 1]
            docs = self.matrix.indices[start:end]
            tf = self.matrix.data[start:end]
            out[docs] += self.idf[idx] * tf * (k1 + 1.0) / (tf + self.doc_norm[docs])
        return out

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        s = self.scores(query)
        if k >= s.shape[0]:
            order = np.argsort(-s)
        else:
            part = np.argpartition(-s, k)[:k]
            order = part[np.argsort(-s[part])]
        return [(self.ids[i], float(s[i])) for i in order[:k]]

    @property
    def size_mb(self) -> float:
        return (self.matrix.data.nbytes + self.matrix.indices.nbytes
                + self.matrix.indptr.nbytes + self.idf.nbytes
                + self.doc_norm.nbytes) / 1048576
