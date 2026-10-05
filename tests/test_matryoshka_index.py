"""Индексы урезанных размерностей собираются из полной матрицы.

Скрипт исполняется на маке после того, как с видеокарты приехала матрица
эмбеддингов, и до замера задержки. Ошибка в нём означает, что задержку
померили не по той матрице, а понять это по самим миллисекундам нельзя.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from rufin.retrieval.model_specs import MODELS, TRAINED  # noqa: E402


@pytest.fixture
def матрица(tmp_path):
    """Полная матрица в раскладке проекта."""
    папка = tmp_path / "base" / "rufin"
    папка.mkdir(parents=True)
    rng = np.random.default_rng(13)
    vec = rng.normal(size=(50, 384)).astype("float32")
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    np.save(папка / "vectors.npy", vec.astype("float16"))
    (папка / "ids.txt").write_text(
        "\n".join(f"590-П_28062017#{i:04d}" for i in range(50)) + "\n",
        encoding="utf-8")
    return tmp_path


def прогнать(embdir, *ключи):
    код = [sys.executable, os.path.join(ROOT, "scripts", "matryoshka_index.py"),
           "--embeddings", str(embdir), *ключи]
    return subprocess.run(код, capture_output=True, text=True)


def test_урезанные_матрицы_собираются_и_нормированы(матрица):
    р = прогнать(матрица)
    assert р.returncode == 0, р.stderr
    for имя in TRAINED:
        spec = MODELS[имя]
        if not spec.truncate_dim:
            continue
        путь = матрица / "base" / имя
        vec = np.load(путь / "vectors.npy")
        assert vec.shape == (50, spec.truncate_dim)
        # после обрезки длина обязана быть снова единицей, иначе индекс
        # считает не косинус
        assert np.allclose(np.linalg.norm(vec.astype("float32"), axis=1), 1.0,
                           atol=1e-3)
        meta = json.loads((путь / "meta.json").read_text(encoding="utf-8"))
        assert meta["dim"] == spec.truncate_dim
        assert len((путь / "ids.txt").read_text(encoding="utf-8").split()) == 50


def test_без_полной_матрицы_отказ(tmp_path):
    р = прогнать(tmp_path)
    assert р.returncode != 0
    assert "нет матрицы" in (р.stdout + р.stderr)
