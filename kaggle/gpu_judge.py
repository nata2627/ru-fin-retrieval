"""Разметка пула кандидатов языковой моделью-судьёй.

Пул — это 600 запросов по четырнадцать кандидатов, восемь тысяч пар. Руками
столько не размечает никто, и разметка моделью здесь не отговорка, а принятая
практика. Но у неё есть цена, и цена называется числом: человек размечает
стратифицированную выборку в 150 пар, и по ней считается каппа Коэна
(`scripts/kappa.py`). Каппа идёт в README и в карточку набора.

Три правила, без которых разметка судьёй подыгрывает.

**Судья не видит, какая конфигурация нашла кандидата.** В листе этого нет
вовсе — `pool_candidates.py` его не записывает и порядок перемешивает.
Иначе оценка подстроилась бы под знакомый метод.

**Инструкция судье и инструкция человеку — один текст.** Он лежит
константой `RUBRIC` в `rufin.annotation`, оттуда попадает и в подсказку,
и в `docs/ANNOTATION_GUIDE.md`. Разойдись они — каппа померила бы разницу
в инструкциях.

**Судья видит фрагмент целиком.** В листе для человека фрагмент обрезан
до трёхсот знаков ради обозримости; подавать судье обрезанный текст значило
бы спрашивать его о другом.

Температура нулевая: разметка обязана воспроизводиться. Ответ ожидается
одной цифрой, и всё, что цифрой не оказалось, попадает в отдельный счётчик,
а не в оценку по умолчанию.
"""
from __future__ import annotations

import collections
import csv
import json
import os
import re
import time

from rufin.annotation import RUBRIC

DEFAULT_JUDGE = "Qwen/Qwen2.5-14B-Instruct-AWQ"

SYSTEM = ("Ты оцениваешь релевантность фрагмента нормативного акта Банка России "
          "запросу. Отвечай одной цифрой: 0, 1 или 2. Ничего больше.")

PROMPT = """{rubric}
--- запрос ---
{query}

--- фрагмент ---
{passage}

Оценка (одна цифра: 0, 1 или 2):"""

_DIGIT = re.compile(r"[012]")


def read_pool(path: str) -> list[dict]:
    """Лист кандидатов: пары «запрос — фрагмент».

    Текст вопроса в листе стоит только в первой строке группы, остальные
    пустые — так лист читается человеком. Здесь он раздаётся всей группе.
    """
    rows: list[dict] = []
    current = ""
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("vopros"):
                current = row["vopros"]
            rows.append({"query_id": row["query_id"], "chunk_id": row["chunk_id"],
                         "vopros": row.get("vopros") or current})
    return rows


def judge(llm, pairs: list[dict], texts: dict[str, str], out_path: str,
          batch: int = 512) -> dict:
    """Разметить пары. Возвращает статистику; оценки пишутся в TSV."""
    from vllm import SamplingParams

    tok = llm.get_tokenizer()
    params = SamplingParams(temperature=0.0, max_tokens=4)

    marks: list[dict] = []
    stats: collections.Counter = collections.Counter()
    t0 = time.time()
    for start in range(0, len(pairs), batch):
        part = [p for p in pairs[start:start + batch] if p["chunk_id"] in texts]
        stats["фрагмент не найден"] += len(pairs[start:start + batch]) - len(part)
        if not part:
            continue
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": PROMPT.format(
                 rubric=RUBRIC, query=p["vopros"], passage=texts[p["chunk_id"]])}],
            tokenize=False, add_generation_prompt=True) for p in part]
        outs = llm.generate(prompts, params, use_tqdm=False)
        for p, o in zip(part, outs):
            raw = o.outputs[0].text.strip()
            m = _DIGIT.search(raw)
            if m is None:
                stats["ответ не цифра"] += 1
                continue
            score = int(m.group())
            stats[f"оценка {score}"] += 1
            marks.append({"query_id": p["query_id"], "chunk_id": p["chunk_id"],
                          "otsenka_0_1_2": score, "syroy_otvet": raw[:40]})
        done = min(start + batch, len(pairs))
        print(f"   размечено {done}/{len(pairs)}, {time.time() - t0:.0f} с", flush=True)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query_id", "chunk_id", "otsenka_0_1_2",
                                          "syroy_otvet"], delimiter="\t")
        w.writeheader()
        w.writerows(marks)

    with_two = {m["query_id"] for m in marks if m["otsenka_0_1_2"] == 2}
    queries = {m["query_id"] for m in marks}
    result = {"пар размечено": len(marks), "распределение": dict(stats),
              "запросов в пуле": len(queries),
              "запросов, у которых есть двойка": len(with_two),
              "секунд": round(time.time() - t0, 1)}
    print(f"\nпар размечено {len(marks)}; запросов {len(queries)}, "
          f"с найденным прямым ответом {len(with_two)} "
          f"({100 * len(with_two) / max(1, len(queries)):.0f}%)", flush=True)
    print(f"распределение оценок: {dict(stats)}", flush=True)
    print("запрос, у которого ни одному кандидату не поставлена 2, из набора "
          "выбывает: ответа на него в корпусе поиск не нашёл", flush=True)
    return result


def load_texts(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        return {c["chunk_id"]: c["text"] for c in (json.loads(l) for l in f if l.strip())}
