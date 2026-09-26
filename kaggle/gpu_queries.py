"""Генерация синтетических запросов на видеокарте.

Модель читает фрагмент нормативного акта и формулирует вопрос, ответ
на который в этом фрагменте содержится; фрагмент становится эталонным
ответом. Сгенерированное проходит два фильтра (`rufin.queryfilter`):
выбрасываются вопросы, списанные с фрагмента, и вопросы, подходящие
к половине корпуса.

Про выбор модели. Взята Qwen2.5-14B-Instruct в четырёхбитном квантовании:
в половинной точности она заняла бы 29,5 ГБ и на T4 не поместилась бы вовсе,
а в четырёх битах занимает 8,1 ГБ из 16 доступных. Именно квантование
и позволяет взять на этой карте модель вдвое крупнее, а не ускоряет счёт.
"""
from __future__ import annotations

import collections
import json
import os
import random
import re
import time

from rufin.queryfilter import IdfTable, copy_score, is_good_source, judge_query

DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct"

SYSTEM = ("Ты помогаешь составить набор проверочных вопросов к нормативным актам "
          "Банка России. Отвечай только самим вопросом, без пояснений и без кавычек.")

PROMPT = """Ниже фрагмент нормативного акта Банка России.

--- начало фрагмента ---
{passage}
--- конец фрагмента ---

Сформулируй ОДИН вопрос, который в работе задал бы риск-аналитик, методолог
или бухгалтер банка, и ответ на который содержится именно в этом фрагменте.

Требования к вопросу:
- своими словами, без дословных кусков из фрагмента длиннее трёх слов подряд;
- конкретный: по нему должен находиться именно этот фрагмент, а не любой
  документ Банка России;
- не упоминай номер и название акта;
- одно предложение, заканчивается знаком вопроса.

Вопрос:"""


def load_model(model_path: str, four_bit: bool = True, device: str = "cuda"):
    """Загрузить модель-генератор.

    Имя параметра типа данных у transformers менялось (`torch_dtype` -> `dtype`),
    версия на Kaggle заранее не известна, поэтому подходящее подбирается пробой.
    """
    import inspect

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    tok.padding_side = "left"          # для батчевой генерации дополнять слева
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype_key = "dtype" if "dtype" in inspect.signature(
        AutoModelForCausalLM.from_pretrained).parameters else "torch_dtype"
    kwargs = {dtype_key: torch.float16 if device != "cpu" else torch.float32}
    if device == "cuda":
        kwargs["device_map"] = "cuda:0"
    if four_bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    if device != "cuda":
        model = model.to(device)
    model.eval()
    return tok, model


def generate(tok, model, passages: list[str], max_new_tokens: int = 96,
             temperature: float = 0.7) -> list[str]:
    import torch
    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": PROMPT.format(passage=p)}],
        tokenize=False, add_generation_prompt=True) for p in passages]
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
              max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                             temperature=temperature, top_p=0.9,
                             pad_token_id=tok.pad_token_id)
    answers = []
    for i in range(out.shape[0]):
        text = tok.decode(out[i][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        text = re.sub(r"^[\s\"'«»\-–—]*", "", text).split("\n")[0].strip(" \"'«»")
        answers.append(text)
    return answers


def body_of(chunk: dict) -> str:
    """Текст фрагмента без приписанной шапки: вопрос задаётся по содержанию."""
    return chunk["text"].split("\n\n", 1)[-1] if "\n\n" in chunk["text"] else chunk["text"]


def make_queries(chunks: list[dict], out_path: str, target: int = 150, per_act: int = 2,
                 model_path: str = DEFAULT_MODEL, batch_size: int = 8, seed: int = 13,
                 four_bit: bool = True, device: str = "cuda") -> dict:
    idf = IdfTable([c["text"] for c in chunks])
    good = [c for c in chunks if is_good_source(c)]
    random.Random(seed).shuffle(good)
    print(f"фрагментов всего {len(chunks)}, пригодных как источник вопроса {len(good)} "
          f"({100 * len(good) / len(chunks):.0f}%)", flush=True)

    tok, model = load_model(model_path, four_bit=four_bit, device=device)
    print(f"модель поднята: {model_path}" + (" (4 бита)" if four_bit else ""), flush=True)

    out: list[dict] = []
    rejected = collections.Counter()
    per_act_count = collections.Counter()
    t0 = time.time()
    pos = 0
    while len(out) < target and pos < len(good):
        batch = []
        while len(batch) < batch_size and pos < len(good):
            ch = good[pos]
            pos += 1
            if per_act_count[ch["act_id"]] >= per_act:
                continue
            batch.append(ch)
        if not batch:
            break
        questions = generate(tok, model, [body_of(c)[:4000] for c in batch])
        for ch, q in zip(batch, questions):
            passage = body_of(ch)
            ok, why = judge_query(q, passage, idf)
            if not ok:
                rejected[why.split("(")[0].strip()] += 1
                continue
            run, share = copy_score(q, passage)
            per_act_count[ch["act_id"]] += 1
            out.append({
                "query_id": f"syn{len(out):04d}", "origin": "синтетический", "text": q,
                "gold_chunk_id": ch["chunk_id"], "act_id": ch["act_id"],
                "number": ch["number"], "date": ch["date"], "act_title": ch["title"],
                "section": ch["section"], "units": ch["units"],
                "mean_idf": round(idf.mean_idf(q), 2),
                "copy_run": run, "copy_share": round(share, 2),
            })
            if len(out) >= target:
                break
        print(f"   принято {len(out)}/{target}, отклонено {sum(rejected.values())}, "
              f"{time.time() - t0:.0f} с", flush=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for q in out:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    attempts = len(out) + sum(rejected.values())
    stats = {"model": model_path, "four_bit": four_bit, "accepted": len(out),
             "rejected": sum(rejected.values()),
             "reject_share": round(sum(rejected.values()) / max(1, attempts), 2),
             "reasons": dict(rejected.most_common()),
             "acts_touched": len(per_act_count), "seconds": round(time.time() - t0, 1)}
    print(f"\nпринято {stats['accepted']}, отклонено {stats['rejected']} "
          f"({100 * stats['reject_share']:.0f}% попыток)", flush=True)
    print(f"причины отказа: {stats['reasons']}", flush=True)
    print(f"актов затронуто: {stats['acts_touched']}, время {stats['seconds']:.0f} с", flush=True)
    return stats
