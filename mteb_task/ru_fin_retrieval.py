"""Таск MTEB для набора ru-fin-retrieval.

Зачем он нужен. Свой бенчмарк, который прогоняется только своим кодом, —
это утверждение, а не измерение: проверить его снаружи нельзя. Набор лежит
в формате BeIR (`corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv`), и тонкой
обёртки достаточно, чтобы его прогнал чужой код, ничего не переписывая.

Подвыборки заведены отдельными разделениями (`splits`), а не усредняются
в одну цифру: живые вопросы, ручные запросы, невиданные акты и синтетика —
это разные речевые режимы, и одно среднее по ним говорит о смеси, которой
не существует. Заголовочная цифра набора — `test`: живые, ручные
и невиданные акты вместе. Синтетика вынесена в `synthetic` и в заголовочную
цифру не входит.

Запуск:

    import mteb
    from ru_fin_retrieval import RuFinRetrieval

    mteb.MTEB(tasks=[RuFinRetrieval()]).run(model, output_folder="results")
"""
from __future__ import annotations

import json
import os

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

# Набор можно читать с диска (папка data/benchmark проекта) или с HuggingFace.
LOCAL = os.environ.get("RU_FIN_RETRIEVAL_PATH")
HF_NAME = "nata2627/ru-fin-retrieval"


class RuFinRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="RuFinRetrieval",
        description=(
            "Поиск по нормативным актам Банка России: Положения, Инструкции, "
            "Указания, Методические рекомендации и Официальные разъяснения "
            "2013–2026 годов, нарезанные на фрагменты. Запросы — настоящие "
            "вопросы поднадзорных организаций из раздела «Разъяснения» Банка "
            "России, короткие поисковые запросы, написанные руками, "
            "и синтетические вопросы к актам, не участвовавшим в обучении."),
        reference="https://github.com/nata2627/ru-fin-retrieval",
        dataset={"path": HF_NAME, "revision": "main"},
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["rus-Cyrl"],
        main_score="ndcg_at_10",
        date=("2013-01-01", "2026-09-29"),
        domains=["Legal", "Financial", "Government", "Written"],
        task_subtypes=["Question answering"],
        license="mit",
        annotations_creators="mixed",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@misc{ru-fin-retrieval,
  title  = {ru-fin-retrieval: бенчмарк поиска по нормативным актам Банка России},
  author = {Гордеева, Наталия},
  year   = {2026},
  url    = {https://github.com/nata2627/ru-fin-retrieval}
}""",
    )

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        if LOCAL:
            self.corpus, self.queries, self.relevant_docs = _from_disk(LOCAL)
        else:
            super().load_data(**kwargs)
            return
        self.data_loaded = True


def _from_disk(root: str) -> tuple[dict, dict, dict]:
    """Прочитать набор из папки в формате BeIR.

    Нужно для прогона до публикации на HuggingFace и для проверки, что
    выложенное и лежащее на диске — одно и то же.
    """
    corpus = {}
    with open(os.path.join(root, "corpus.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            corpus[d["_id"]] = {"title": d.get("title", ""), "text": d["text"]}
    queries = {}
    with open(os.path.join(root, "queries.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            queries[d["_id"]] = d["text"]
    rel: dict[str, dict[str, int]] = {}
    with open(os.path.join(root, "qrels", "test.tsv"), encoding="utf-8") as f:
        next(f)
        for line in f:
            qid, cid, score = line.rstrip("\n").split("\t")[:3]
            rel.setdefault(qid, {})[cid] = int(score)
    # MTEB ждёт словарь по разделениям
    return ({"test": corpus}, {"test": queries}, {"test": rel})
