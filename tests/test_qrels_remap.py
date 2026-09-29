"""Перенос эталонов между нарезками.

Ошибка здесь не падает и не видна по метрикам: она превращается в ровный
ноль или, наоборот, в лишние правильные ответы, и абляция по нарезке
начинает мерить не нарезку, а качество переноса. Поэтому правило проверяется
на отрезках, выписанных руками, а тождественность переноса нарезки в себя —
ещё и на настоящей разметке.
"""
from __future__ import annotations

import json

import pytest

from rufin.qrels_remap import Span, overlap_chars, remap, share_of_gold, transfer_one

ПОРОГ = 0.5


def отрезок(name: str, start: int, end: int, act: str = "A") -> Span:
    return Span(name, act, start, end)


def test_перекрытие_считается_по_символам():
    a, b = отрезок("a", 0, 100), отрезок("b", 60, 200)
    assert overlap_chars(a, b) == 40
    assert share_of_gold(a, b) == pytest.approx(0.4)
    # доля считается от эталона, поэтому у обратной пары она другая
    assert share_of_gold(b, a) == pytest.approx(40 / 140)


def test_фрагменты_разных_актов_не_пересекаются():
    a = отрезок("a", 0, 100, act="A")
    b = отрезок("b", 0, 100, act="B")
    assert overlap_chars(a, b) == 0


def test_соприкосновение_границ_не_пересечение():
    # [0; 100) и [100; 200) общих символов не имеют
    assert overlap_chars(отрезок("a", 0, 100), отрезок("b", 100, 200)) == 0


def test_оценку_два_получает_наибольшее_перекрытие():
    gold = отрезок("g", 0, 100)
    кандидаты = [отрезок("t0", 0, 30), отрезок("t1", 30, 100)]
    t = transfer_one(gold, 2, кандидаты, ПОРОГ)
    assert t.target["t1"] == 2
    assert "t0" not in t.target        # 0,30 эталона — ниже порога


def test_кандидат_выше_порога_получает_единицу():
    gold = отрезок("g", 0, 100)
    кандидаты = [отрезок("t0", 0, 60), отрезок("t1", 40, 100)]
    t = transfer_one(gold, 2, кандидаты, ПОРОГ)
    assert t.target == {"t0": 2, "t1": 1}


def test_из_единицы_единицы_не_плодятся():
    # эталон с оценкой 1 — уже «рядом лежащий пункт»; сосед соседа был бы догадкой
    gold = отрезок("g", 0, 100)
    кандидаты = [отрезок("t0", 0, 60), отрезок("t1", 40, 100)]
    t = transfer_one(gold, 1, кандидаты, ПОРОГ)
    assert t.target == {"t0": 1}


def test_эталон_без_пересечений_теряется_и_это_видно():
    qrels = {"q1": {"g": 2}}
    source = {"g": отрезок("g", 0, 100)}
    target = [отрезок("t", 500, 600)]
    moved, stats = remap(qrels, source, target, ПОРОГ)
    assert moved == {}
    assert stats.lost == 1 and stats.lost_ids == ["q1\tg"]


def test_эталон_которого_нет_в_исходной_нарезке_учтён_отдельно():
    moved, stats = remap({"q1": {"нет-такого": 2}}, {}, [отрезок("t", 0, 10)], ПОРОГ)
    assert moved == {} and stats.unknown == 1


def test_число_двоек_сохраняется():
    # иначе у нарезок окажется разное число правильных ответов на запрос
    # и сравнивать их будет нельзя
    qrels = {"q1": {"g1": 2, "g2": 2}}
    source = {"g1": отрезок("g1", 0, 100), "g2": отрезок("g2", 100, 200)}
    target = [отрезок("t0", 0, 50), отрезок("t1", 50, 100),
              отрезок("t2", 100, 150), отрезок("t3", 150, 200)]
    moved, stats = remap(qrels, source, target, ПОРОГ)
    assert stats.grade_two == 2
    assert sorted(k for k, v in moved["q1"].items() if v == 2) == ["t0", "t2"]


def test_совпадение_границ_переносит_оценку_и_не_добавляет_соседей():
    gold = отрезок("g", 0, 100)
    # сосед перекрывает 0,60 эталона и при обычном правиле получил бы единицу
    кандидаты = [отрезок("тот же", 0, 100), отрезок("сосед", 40, 140)]
    t = transfer_one(gold, 2, кандидаты, ПОРОГ)
    assert t.target == {"тот же": 2} and t.exact


def test_перенос_нарезки_в_себя_тождественен():
    qrels = {"q1": {"c0": 2, "c1": 1}, "q2": {"c2": 2}}
    spans = {"c0": отрезок("c0", 0, 100), "c1": отрезок("c1", 85, 200),
             "c2": отрезок("c2", 180, 300)}
    moved, stats = remap(qrels, spans, list(spans.values()), ПОРОГ)
    assert moved == qrels
    assert stats.exact == 3 and stats.lost == 0


@pytest.mark.parametrize("порог", [0.0, 0.3, 0.5, 0.9, 1.0])
def test_тождественность_не_зависит_от_порога(порог):
    # при нулевом пороге единицу получил бы каждый пересекающийся сосед —
    # спасает именно правило о совпадении границ
    qrels = {"q1": {"c0": 2}}
    spans = {"c0": отрезок("c0", 0, 100), "c1": отрезок("c1", 50, 150)}
    moved, _ = remap(qrels, spans, list(spans.values()), порог)
    assert moved == qrels


@pytest.mark.data
def test_перенос_базовой_нарезки_в_себя_даёт_исходную_разметку(data_dir):
    """То же самое, но на настоящих данных: 185 эталонов, 62 594 фрагмента."""
    chunks = data_dir / "chunks" / "base.jsonl"
    qrels_path = data_dir / "queries" / "qrels.tsv"
    if not chunks.exists() or not qrels_path.exists():
        pytest.skip("нет нарезки или разметки")

    qrels: dict[str, dict[str, int]] = {}
    with qrels_path.open(encoding="utf-8") as f:
        next(f)
        for line in f:
            qid, cid, score = line.rstrip("\n").split("\t")
            qrels.setdefault(qid, {})[cid] = int(score)

    нужные_акты = {cid.split("#")[0] for d in qrels.values() for cid in d}
    spans: dict[str, Span] = {}
    with chunks.open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c["act_id"] in нужные_акты:
                spans[c["chunk_id"]] = Span.from_chunk(c)

    moved, stats = remap(qrels, spans, list(spans.values()), ПОРОГ)
    assert moved == qrels
    assert stats.lost == 0 and stats.unknown == 0
    assert stats.exact == stats.entries

