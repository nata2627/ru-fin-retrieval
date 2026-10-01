"""Отбор фрагментов для вопросов и отбраковка негодных вопросов.

Две проверки из плана работ.

1. Вопрос не должен дословно повторять фрагмент. Иначе поиск решает задачу
   поиска подстроки, а не смысла, и BM25 получает незаслуженное преимущество,
   которое потом выглядит как «плотный поиск не нужен».
2. Вопрос не должен подходить к половине корпуса. «Какие требования
   устанавливает Банк России?» формально корректен, но правильного ответа
   у него нет, и такой запрос только зашумляет метрики.

Первая проверка — по длине самой длинной общей цепочки слов. Вторая —
по средней обратной частоте слов запроса: у общих слов она низкая,
и запрос из одних общих слов ничего не выделяет.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from .retrieval.text import tokenize

# самая длинная общая цепочка слов, при которой вопрос ещё считается
# переформулировкой, а не копией
MAX_COMMON_RUN = 5
# доля слов вопроса, встречающихся во фрагменте подряд
MAX_COPY_SHARE = 0.6
# минимальная средняя обратная частота слов запроса
MIN_MEAN_IDF = 3.0
MIN_QUERY_TOKENS = 4


def longest_common_run(a: list[str], b: list[str]) -> int:
    """Длина самой длинной общей подпоследовательности подряд идущих слов."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


class IdfTable:
    """Обратная частота слов по корпусу чанков."""

    def __init__(self, texts: list[str]):
        df = Counter()
        for t in texts:
            df.update(set(tokenize(t)))
        self.n = len(texts)
        self.df = df

    def idf(self, token: str) -> float:
        return math.log((self.n + 1) / (self.df.get(token, 0) + 1))

    def mean_idf(self, query: str) -> float:
        toks = tokenize(query)
        return sum(self.idf(t) for t in toks) / len(toks) if toks else 0.0


def copy_score(query: str, passage: str) -> tuple[int, float]:
    """Насколько вопрос списан с фрагмента: длина общей цепочки и её доля."""
    q, p = tokenize(query), tokenize(passage)
    run = longest_common_run(q, p)
    return run, run / len(q) if q else 0.0


_КИРИЛЛИЦА = re.compile(r"[а-яёА-ЯЁ]")
_ЛАТИНИЦА = re.compile(r"[a-zA-Z]")
# Букв, которых в русском нет вовсе. Одна такая — верный признак, что модель
# сползла в соседний язык, и вопрос надо выбрасывать целиком.
_НЕ_РУССКИЕ_БУКВЫ = re.compile(r"[іїєґІЇЄҐ]")


def po_russki(query: str) -> tuple[bool, str]:
    """Написан ли вопрос по-русски.

    Проверка не теоретическая. Модель-генератор для теста берётся из другого
    семейства нарочно — иначе тест померил бы, насколько ученик выучил стиль
    своего же генератора, — но другое семейство хуже держит русский.
    Mistral-7B-Instruct на пробе выдал по-английски **одиннадцать вопросов
    из двадцати** в тесте и шесть из двадцати в dev, а один написал
    по-украински. Набор для поиска по актам Банка России, где половина
    вопросов не на русском, мерил бы неизвестно что, и по метрикам это
    выглядело бы прилично.

    Остальные фильтры этого не ловят: они смотрят на списывание и на общность
    слов, а английский вопрос не списан и вполне конкретен.

    Латиница сама по себе не криминал: USD, МСФО (IFRS), SWIFT и LEI
    встречаются в законных вопросах. Поэтому сравниваются доли, а не
    наличие.
    """
    кир = len(_КИРИЛЛИЦА.findall(query))
    лат = len(_ЛАТИНИЦА.findall(query))
    if not кир and not лат:
        return False, "без букв"
    if лат >= кир:
        return False, "не по-русски"
    if _НЕ_РУССКИЕ_БУКВЫ.search(query):
        return False, "не по-русски (соседний язык)"
    return True, ""


# Отсылки к самому фрагменту. Вопрос с ними бессмыслен как поисковый запрос:
# «этого фрагмента» у пользователя перед глазами нет, и найти по такому
# запросу нечего. Модель пишет их, когда сбивается с задачи «составь вопрос»
# на задачу «перескажи текст».
_НА_СЕБЯ = re.compile(
    r"\b(?:в|из|по|согласно|исходя\s+из)?\s*(?:этом|этого|этому|данном|данного|"
    r"данному|настоящем|настоящего|приведённом|приведенном|указанном)\s+"
    r"(?:фрагмент\w*|текст\w*|отрывк\w*|пункт\w*|документ\w*|акт\w*|положени\w*)",
    re.IGNORECASE)
_НА_СЕБЯ_КОРОТКО = re.compile(
    r"\b(?:этот|этого|данный|данного|настоящий|настоящего)\s+фрагмент\w*",
    re.IGNORECASE)


def ssylaetsya_na_sebya(query: str) -> bool:
    """Ссылается ли вопрос на сам фрагмент вместо того, чтобы его искать."""
    return bool(_НА_СЕБЯ.search(query) or _НА_СЕБЯ_КОРОТКО.search(query))


def judge_query(query: str, passage: str, idf: IdfTable,
                require_question_mark: bool = True) -> tuple[bool, str]:
    """Годится ли вопрос. Возвращает решение и причину отказа.

    `require_question_mark` выключается для одного стиля — короткого
    поискового запроса. В строку поиска набирают «резерв по ссуде третьей
    категории», без вопросительного знака и без глагола, и это не брак
    генерации, а третий речевой режим, ради которого стиль и заведён.
    Для остальных стилей проверка остаётся: модель, сбившаяся с задачи,
    первым делом начинает писать утверждения и пересказ фрагмента.
    """
    ладно, почему = po_russki(query)
    if not ладно:
        return False, почему
    if ssylaetsya_na_sebya(query):
        return False, "ссылается на сам фрагмент"
    toks = tokenize(query)
    if len(toks) < MIN_QUERY_TOKENS:
        return False, "слишком короткий"
    if require_question_mark and not query.strip().endswith("?"):
        return False, "не вопрос"
    run, share = copy_score(query, passage)
    if run > MAX_COMMON_RUN or share > MAX_COPY_SHARE:
        return False, f"списан с фрагмента (цепочка {run} слов)"
    if idf.mean_idf(query) < MIN_MEAN_IDF:
        return False, "подходит к слишком многим фрагментам"
    return True, ""


# фрагменты, на которых нельзя задать осмысленный вопрос
_DIGITS = re.compile(r"\d")


def is_good_source(chunk: dict, min_chars: int = 500, max_digit_share: float = 0.18) -> bool:
    """Пригоден ли фрагмент как источник вопроса.

    Отсекаются перечни ссылок на законы, таблицы и формы отчётности: текст
    там есть, но содержательного правила, о котором можно спросить, нет.
    """
    text = chunk["text"]
    body = text.split("\n\n", 1)[-1] if "\n\n" in text else text
    if len(body) < min_chars:
        return False
    if sum(bool(_DIGITS.match(c)) for c in body) / len(body) > max_digit_share:
        return False
    if body.count("ст. ") + body.count("№ ") > len(body) / 200:
        return False                       # сплошные ссылки на законы
    if chunk.get("section", "").startswith("Приложение"):
        return False
    return body.count(".") >= 3            # хотя бы несколько предложений
