"""Запись и чтение набора запросов в формате MTEB (BeIR).

Формат нужен, чтобы бенчмарк можно было отдать наружу и прогнать чужим
кодом. Значит, он должен переживать круг «записали — прочитали» без потерь.
"""
from __future__ import annotations

from rufin.benchmark import read_jsonl, read_qrels, write_corpus, write_qrels, write_queries


def test_разметка_переживает_запись_и_чтение(tmp_path):
    qrels = {"q1": {"c1": 2, "c2": 1}, "q2": {"c3": 2}}
    path = tmp_path / "qrels" / "test.tsv"
    write_qrels(str(path), qrels)
    assert read_qrels(str(path)) == qrels


def test_нулевая_оценка_не_попадает_в_файл(tmp_path):
    path = tmp_path / "qrels.tsv"
    write_qrels(str(path), {"q1": {"c1": 2, "c2": 0}})
    assert read_qrels(str(path)) == {"q1": {"c1": 2}}


def test_у_разметки_есть_шапка(tmp_path):
    path = tmp_path / "qrels.tsv"
    write_qrels(str(path), {"q1": {"c1": 2}})
    assert path.read_text(encoding="utf-8").splitlines()[0] == "query-id\tcorpus-id\tscore"


def test_файл_без_шапки_тоже_читается(tmp_path):
    path = tmp_path / "qrels.tsv"
    path.write_text("q1\tc1\t2\n", encoding="utf-8")
    assert read_qrels(str(path)) == {"q1": {"c1": 2}}


def test_битая_строка_пропускается(tmp_path):
    path = tmp_path / "qrels.tsv"
    path.write_text("query-id\tcorpus-id\tscore\nq1\tc1\t2\nмусор\n", encoding="utf-8")
    assert read_qrels(str(path)) == {"q1": {"c1": 2}}


def test_корпус_несет_опознавательные_данные_акта(tmp_path):
    path = tmp_path / "corpus.jsonl"
    write_corpus(str(path), [{
        "chunk_id": "590-П_28062017#0001", "number": "590-П", "date": "28.06.2017",
        "title": "О резервах", "section": "Глава 1", "text": "текст фрагмента",
    }])
    row = read_jsonl(str(path))[0]
    assert row["_id"] == "590-П_28062017#0001"
    assert row["title"] == "590-П от 28.06.2017. О резервах // Глава 1"
    assert row["text"] == "текст фрагмента"


def test_фрагмент_без_раздела_не_получает_пустой_хвост(tmp_path):
    path = tmp_path / "corpus.jsonl"
    write_corpus(str(path), [{
        "chunk_id": "c1", "number": "1-У", "date": "01.01.2020",
        "title": "О чем-то", "section": "", "text": "текст",
    }])
    assert "//" not in read_jsonl(str(path))[0]["title"]


def test_запросы_пишутся_в_формате_beir(tmp_path):
    path = tmp_path / "queries.jsonl"
    write_queries(str(path), [{"query_id": "syn0000", "text": "Какой порядок?", "origin": "x"}])
    row = read_jsonl(str(path))[0]
    assert row == {"_id": "syn0000", "text": "Какой порядок?"}


def test_кириллица_не_экранируется(tmp_path):
    path = tmp_path / "queries.jsonl"
    write_queries(str(path), [{"query_id": "q1", "text": "резерв"}])
    assert "резерв" in path.read_text(encoding="utf-8")
