"""Сквозной прогон финального шага на крошечной модели, без видеокарты.

Зачем. Шаг `final` — единственное касание теста за весь этап, и он
не исполнялся ещё ни разу нигде. Проверять его на видеокарте значит
платить часами за каждую опечатку, а опечатка там уже стоила шестнадцати
минут обучения (`sorted` по словарю со смешанными ключами).

Поэтому здесь собирается настоящая, но крошечная модель: два слоя,
скрытый размер 32, словарь на две сотни токенов. Она ничего не умеет,
но она настоящая — через неё проходят ровно те же вызовы, что и через
обученную. Весь шаг исполняется целиком, с записью выдач, отчёта
и матрицы эмбеддингов.

Сеть и токенизатор собираются из описания, ничего не качается.
"""
from __future__ import annotations

import json
import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kaggle"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import run_phase_d as R  # noqa: E402

from rufin.training.journal import Journal, entry_from_per_query  # noqa: E402


@pytest.fixture(scope="module")
def модель(tmp_path_factory) -> str:
    """Крошечная модель на диске, собранная из описания."""
    from transformers import BertConfig, BertModel, BertTokenizerFast

    папка = tmp_path_factory.mktemp("модель")
    словарь = папка / "vocab.txt"
    токены = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + \
             [f"т{i}" for i in range(195)]
    словарь.write_text("\n".join(токены) + "\n", encoding="utf-8")
    tok = BertTokenizerFast(vocab_file=str(словарь))
    tok.save_pretrained(str(папка))

    cfg = BertConfig(vocab_size=len(токены), hidden_size=32, num_hidden_layers=2,
                     num_attention_heads=2, intermediate_size=64,
                     max_position_embeddings=64)
    BertModel(cfg).save_pretrained(str(папка))
    return str(папка)


def окружение(tmp_path, фрагментов: int = 40, запросов: int = 12):
    """Окружение, собранное вручную.

    Настоящий конструктор читает корпус, сплит и разметку с диска и строит
    нарезку; здесь всё это подменено, потому что проверяется не он,
    а то, что делает шаг `final` с уже собранным окружением.
    """
    env = object.__new__(R.Окружение)
    env.device = "cpu"
    env.chunks = [{"chunk_id": f"590-П_28062017#{i:04d}", "act_id": "590-П_28062017",
                   "text": f"текст фрагмента номер {i} про резервы и ссуды"}
                  for i in range(фрагментов)]
    env.texts = {c["chunk_id"]: c["text"] for c in env.chunks}
    env.queries = [{"query_id": f"q{i:04d}", "text": f"вопрос номер {i}"}
                   for i in range(запросов)]
    env.qrels = {f"q{i:04d}": {f"590-П_28062017#{i:04d}": 2} for i in range(запросов)}
    env.dev = {f"q{i:04d}" for i in range(запросов // 2)}
    env.dev_queries = [q for q in env.queries if q["query_id"] in env.dev]
    env.dev_qrels = {k: v for k, v in env.qrels.items() if k in env.dev}
    env.student = ""
    return env


def ключи(tmp_path, веса: str, **прочее):
    значения = dict(weights=веса, runs=str(tmp_path / "runs"),
                    embeddings=str(tmp_path / "embeddings"),
                    journal=str(tmp_path / "journal.json"),
                    batch_size=8, tags=None, keep_only_best=False,
                    forget_task="RuBQRetrieval", forget_queries=10)
    значения.update(прочее)
    return types.SimpleNamespace(**значения)


def журнал_с(меткой: str, config: dict) -> Journal:
    j = Journal()
    pq = {n: [0.5] * 6 for n in ("Recall@1", "Recall@5", "Recall@10",
                                 "MRR@10", "NDCG@10")}
    qids = [f"q{i:04d}" for i in range(6)]
    j.add(entry_from_per_query("базовая", "базовая", pq, qids))
    pq2 = {n: [0.7] * 6 for n in pq}
    j.add(entry_from_per_query(меткой, "A", pq2, qids, config=config))
    return j


# ---- сам сквозной прогон ----

def test_финал_проходит_целиком_и_пишет_всё_что_обещал(модель, tmp_path):
    веса = tmp_path / "weights"
    (веса / "a-3ep").mkdir(parents=True)
    for имя in os.listdir(модель):
        os.link(os.path.join(модель, имя), веса / "a-3ep" / имя)

    env = окружение(tmp_path)
    args = ключи(tmp_path, str(веса))
    j = журнал_с("a-3ep", {"tag": "a-3ep", "stage": "A", "matryoshka": []})

    R.шаг_final(env, args, j)

    выдача = tmp_path / "runs" / "base__dense-rufin.jsonl"
    assert выдача.exists(), "выдача не записана"
    строки = [json.loads(l) for l in выдача.read_text(encoding="utf-8").splitlines()]
    assert len(строки) == len(env.queries)
    assert set(строки[0]) == {"query_id", "ranked"}

    отчёт = json.loads((tmp_path / "runs" / "report_phase_d.json").read_text(encoding="utf-8"))
    assert "dense-rufin" in отчёт["финал"]

    матрица = tmp_path / "embeddings" / "base" / "rufin"
    assert (матрица / "vectors.npy").exists()
    assert (матрица / "ids.txt").exists()
    meta = json.loads((матрица / "meta.json").read_text(encoding="utf-8"))
    assert meta["chunks"] == len(env.chunks)


def test_финал_с_матрёшкой_пишет_три_выдачи(модель, tmp_path):
    """Размерности берутся из журнала, а не из рецепта: состав этапа D
    зависел от того, что оказалось принятым."""
    веса = tmp_path / "weights"
    (веса / "d-matryoshka").mkdir(parents=True)
    for имя in os.listdir(модель):
        os.link(os.path.join(модель, имя), веса / "d-matryoshka" / имя)

    env = окружение(tmp_path)
    args = ключи(tmp_path, str(веса))
    j = журнал_с("d-matryoshka", {"tag": "d-matryoshka", "stage": "D",
                                  "matryoshka": [32, 16, 8]})

    R.шаг_final(env, args, j)

    for имя in ("base__dense-rufin.jsonl", "base__dense-rufin-16.jsonl",
                "base__dense-rufin-8.jsonl"):
        assert (tmp_path / "runs" / имя).exists(), f"нет выдачи {имя}"


def test_финал_отказывается_без_весов(tmp_path):
    env = окружение(tmp_path)
    args = ключи(tmp_path, str(tmp_path / "нет"))
    j = журнал_с("a-3ep", {"tag": "a-3ep", "stage": "A", "matryoshka": []})
    with pytest.raises(SystemExit, match="нет весов"):
        R.шаг_final(env, args, j)


def test_финал_отказывается_когда_нечего_прогонять(tmp_path):
    """Ни один этап не принят. Молча взять любой нельзя: на тест смотрят
    один раз, и смотреть на него непонятно чем — хуже, чем не смотреть."""
    env = окружение(tmp_path)
    args = ключи(tmp_path, str(tmp_path))
    with pytest.raises(SystemExit, match="нечего прогонять"):
        R.шаг_final(env, args, Journal())


def test_уборка_весов_оставляет_только_лучшее(tmp_path):
    веса = tmp_path / "weights"
    for имя in ("a-1ep", "a-3ep", "b-neg"):
        (веса / имя).mkdir(parents=True)
        (веса / имя / "model.safetensors").write_bytes(b"0" * 10)
    args = ключи(tmp_path, str(веса))
    j = журнал_с("a-3ep", {})
    R.прибрать_веса(args, j)
    assert sorted(os.listdir(веса)) == ["a-3ep"]


def test_забывание_честно_пропускается_без_сети(tmp_path):
    """Посторонний набор качается из сети, а сеть запрещена. Шаг обязан
    сказать это и вернуть управление, а не падать и не висеть."""
    env = окружение(tmp_path)
    args = ключи(tmp_path, str(tmp_path), tags=[])
    j = журнал_с("a-3ep", {})
    R.шаг_forget(env, args, j)
    итог = json.loads((tmp_path / "runs" / "zabyvanie.json").read_text(encoding="utf-8"))
    assert "пропущено" in итог or "ошибка" in итог


# ---- контроль воспроизводимости ----

def постоянный(значение: float, n: int = 6):
    return {к: [значение] * n for к in ("Recall@1", "Recall@5", "Recall@10",
                                        "MRR@10", "NDCG@10")}


def test_контроль_сходится_когда_число_то_же(модель, tmp_path, monkeypatch):
    """Журнал склеивается из прогонов на разных образах Kaggle. Перед тем
    как дописывать, считается заново то, что уже есть."""
    env = окружение(tmp_path)
    args = ключи(tmp_path, str(tmp_path))
    j = Journal()
    qids = [f"q{i:04d}" for i in range(6)]
    j.add(entry_from_per_query("базовая", "базовая", постоянный(0.45), qids))

    monkeypatch.setattr(R, "оценить_dev", lambda *a, **k: {
        384: {"per_query": постоянный(0.45), "qids": qids},
        "секунд на корпус": 1.0})
    R.шаг_контроль(env, args, j)
    итог = json.loads((tmp_path / "runs" / "kontrol.json").read_text(encoding="utf-8"))
    assert итог["сошлось"] is True


def test_контроль_отказывается_когда_число_уехало(модель, tmp_path, monkeypatch):
    """Расхождение означает, что часть разницы между этапами окажется
    разницей образов, а не рецепта. Дописывать в такой журнал нельзя."""
    env = окружение(tmp_path)
    args = ключи(tmp_path, str(tmp_path))
    j = Journal()
    qids = [f"q{i:04d}" for i in range(6)]
    j.add(entry_from_per_query("базовая", "базовая", постоянный(0.45), qids))

    monkeypatch.setattr(R, "оценить_dev", lambda *a, **k: {
        384: {"per_query": постоянный(0.60), "qids": qids},
        "секунд на корпус": 1.0})
    with pytest.raises(SystemExit, match="не воспроизвелась"):
        R.шаг_контроль(env, args, j)


def test_контроль_без_журнала_говорит_что_проверять_нечего(tmp_path):
    env = окружение(tmp_path)
    args = ключи(tmp_path, str(tmp_path))
    with pytest.raises(SystemExit, match="нет нулевой точки"):
        R.шаг_контроль(env, args, Journal())


def test_контроль_ничего_не_пишет_в_журнал(модель, tmp_path, monkeypatch):
    """Это проверка, а не этап. Строки в таблице рецепта он не добавляет."""
    env = окружение(tmp_path)
    args = ключи(tmp_path, str(tmp_path))
    j = Journal()
    qids = [f"q{i:04d}" for i in range(6)]
    j.add(entry_from_per_query("базовая", "базовая", постоянный(0.45), qids))
    monkeypatch.setattr(R, "оценить_dev", lambda *a, **k: {
        384: {"per_query": постоянный(0.45), "qids": qids},
        "секунд на корпус": 1.0})
    R.шаг_контроль(env, args, j)
    assert [e.tag for e in j.entries] == ["базовая"]
