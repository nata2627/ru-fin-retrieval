"""Генерация вопросов на видеокарте: vLLM, три речевых режима, три фильтра.

Чем это отличается от `gpu_queries.py`, который тут уже был. Тот поднимал
Qwen2.5-14B в четырёх битах и генерировал пачками по восемь через
`model.generate`; так было получено 150 вопросов. На пятнадцати тысячах
это десяток часов видеокарты — треть недельной квоты на один прогон,
и двенадцатичасовая сессия может не успеть. Поэтому здесь vLLM и модель
поменьше: вопрос по фрагменту рассуждений не требует, а непрерывная пачка
и общий кэш внимания дают порядок разницы в скорости.

**Про T4 и типы данных.** T4 — это Turing, bf16 на нём нет. Запуск vLLM
с `dtype="bfloat16"` либо падает, либо тихо деградирует, поэтому тип
задаётся явно и по умолчанию `float16`. FlashAttention тоже требует Ampere;
vLLM сам откатится на другой бэкенд, и это нормально на длине 4096.

**Про объём.** Сначала замер на сотне фрагментов, потом назначение объёма
по замеру. Десять тысяч вопросов за три часа лучше пятнадцати, которые
не досчитаются.

**Три речевых режима.** Вопрос, написанный моделью по фрагменту, наследует
его словарь, и набор из одного такого стиля меряет, как поиск сопоставляет
канцелярит с канцеляритом. Поэтому стилей три, и они чередуются: письмо
практика, бытовая ситуация и короткий поисковый запрос — тот самый, который
набирают в строку и у которого нет ни глагола, ни вопросительного знака.

**Фильтры.** Два прежних (`rufin.queryfilter`): вопрос, списанный
с фрагмента, сводит задачу к поиску подстроки и дарит преимущество BM25;
вопрос, подходящий к половине корпуса, правильного ответа не имеет.
Третий добавлен здесь: если учитель-кросс-энкодер не ставит исходный
фрагмент в первые пятьдесят, вопрос выбрасывается. Он ловит то, чего
не ловят первые два, — вопрос, формально безупречный, но отвечает на него
не этот фрагмент.
"""
from __future__ import annotations

import collections
import json
import os
import random
import re
import time
from dataclasses import dataclass

from rufin.queryfilter import IdfTable, copy_score, is_good_source, judge_query

# Генератор обучающей выборки и генератор теста обязаны быть из разных
# семейств: иначе тест меряет, насколько ученик выучил стиль своего же
# генератора. Обе модели записываются в DATASET_CARD.md.
TRAIN_MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
# Модель для dev и теста — обязательно из другого семейства, чем TRAIN_MODEL:
# иначе тест померит не качество поиска, а то, насколько ученик выучил стиль
# своего же генератора. Но семейство мало выбрать — оно должно ещё и держать
# русский. Mistral-7B-Instruct-v0.2 на пробе 30.09 выдал одиннадцать английских
# вопросов из двадцати в тесте, а среди русских попадались «заполнение графф»
# и «привести в информированное состояние». Vikhr-Nemo — Mistral-Nemo,
# дообученный на русском: семейство по-прежнему не Qwen, а язык родной.
# Запасная сборка, если эта не заведётся:
#   mandanya/Vikhr-Nemo-12B-Instruct-R-21-09-24-AWQ
TEST_MODEL = "NiGuLa/Vikhr-Nemo-12B-Instruct-R-21-09-24-awq-4bit"

# Требование писать по-русски стоит первым и повторяется в требованиях
# к каждому вопросу. Это не перестраховка: модель для теста берётся из
# другого семейства нарочно, а другое семейство хуже держит русский —
# Mistral-7B-Instruct на пробе выдал одиннадцать английских вопросов
# из двадцати. Фильтр такие отбрасывает, но каждый отброшенный оплачен
# временем видеокарты, и дешевле их не порождать.
SYSTEM = ("Ты помогаешь составить набор проверочных вопросов к нормативным актам "
          "Банка России. Пиши ТОЛЬКО по-русски. Отвечай только самим вопросом, "
          "без пояснений и без кавычек.")

HEAD = """Ниже фрагмент нормативного акта Банка России.

--- начало фрагмента ---
{passage}
--- конец фрагмента ---
"""

COMMON = """
Требования:
- вопрос по-русски, целиком; латиница допустима только в обозначениях
  вроде USD, IFRS, SWIFT;
- своими словами, без дословных кусков из фрагмента длиннее трёх слов подряд;
- конкретно: по запросу должен находиться именно этот фрагмент, а не любой
  документ Банка России;
- не упоминай номер и название акта.
"""


@dataclass(frozen=True)
class Style:
    name: str
    tail: str
    question_mark: bool = True
    max_tokens: int = 96


STYLES = (
    Style("вопрос профессионала", """
Сформулируй ОДИН вопрос, который задал бы риск-аналитик, методолог или
бухгалтер банка, и ответ на который содержится именно в этом фрагменте.
Одно предложение, заканчивается знаком вопроса.
""" + COMMON),
    Style("бытовая ситуация", """
Опиши ОДНУ рабочую ситуацию в двух-трёх предложениях и закончи вопросом,
ответ на который содержится именно в этом фрагменте. Пиши так, как пишут
в банке коллеге: без ссылок на пункты, с обычными словами.
""" + COMMON, max_tokens=160),
    Style("короткий поисковый запрос", """
Напиши ОДИН короткий поисковый запрос — так, как его набрали бы в строку
поиска: три-восемь слов, без глагола, без вопросительного знака, без
вежливых оборотов. Он должен находить именно этот фрагмент.
""" + COMMON, question_mark=False, max_tokens=32),
)


def body_of(chunk: dict) -> str:
    """Текст фрагмента без приписанной шапки: вопрос задаётся по содержанию."""
    return chunk["text"].split("\n\n", 1)[-1] if "\n\n" in chunk["text"] else chunk["text"]


def load_llm(model_path: str, dtype: str = "float16", quantization: str | None = None,
             tensor_parallel: int = 1, max_model_len: int = 4096,
             gpu_memory_utilization: float = 0.90):
    """Поднять vLLM.

    Тип данных задаётся явно: на Turing нет bf16, и «auto» у модели,
    обученной в bfloat16, приводит либо к падению, либо к молчаливой
    деградации — а деградацию видно только по качеству вопросов, то есть
    когда квота уже потрачена.
    """
    # vLLM поднимает движок отдельным процессом и по умолчанию делает это
    # через fork. Форк не годится, если CUDA уже тронута в родителе, —
    # а она тронута всегда: `pick_device()` спрашивает `torch.cuda
    # .is_available()` первой же строкой, до всякой генерации. Падает это
    # не там, где причина: «Cannot re-initialize CUDA in forked subprocess»
    # прилетает из недр движка через полминуты после старта. Переменная
    # ставится до импорта vllm — после него её уже не читают.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from vllm import LLM
    kwargs = dict(model=model_path, dtype=dtype, max_model_len=max_model_len,
                  gpu_memory_utilization=gpu_memory_utilization,
                  tensor_parallel_size=tensor_parallel, trust_remote_code=True)
    if quantization:
        kwargs["quantization"] = quantization
    print(f"поднимаю vLLM: {model_path} ({dtype}"
          + (f", {quantization}" if quantization else "")
          + f", карт {tensor_parallel})", flush=True)
    return LLM(**kwargs)


def _clean(text: str) -> str:
    text = re.sub(r"^[\s\"'«»\-–—]*", "", text)
    return text.split("\n")[0].strip(" \"'«»")


def _chat(tok, задание: str) -> str:
    """Собрать запрос по шаблону модели, пережив отсутствие системной роли.

    Шаблоны разных семейств несовместимы, и это не мелочь оформления.
    Qwen принимает системную роль, а Mistral-7B-Instruct требует строгого
    чередования user/assistant и на системное сообщение отвечает
    `TemplateError: Conversation roles must alternate`. Модель другого
    семейства для теста обязательна — иначе тест померит, насколько ученик
    выучил стиль своего же генератора, — поэтому подстраиваться приходится
    здесь, а не выбором модели.

    Запасной путь не выбрасывает наставление, а приклеивает его к вопросу:
    выбросить — значит менять задачу, и вопросы Mistral оказались бы
    несопоставимы с вопросами Qwen.
    """
    try:
        return tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": задание}],
            tokenize=False, add_generation_prompt=True)
    except Exception:
        return tok.apply_chat_template(
            [{"role": "user", "content": SYSTEM + "\n\n" + задание}],
            tokenize=False, add_generation_prompt=True)


def generate(llm, style: Style, passages: list[str], temperature: float = 0.7) -> list[str]:
    from vllm import SamplingParams
    tok = llm.get_tokenizer()
    prompts = [_chat(tok, HEAD.format(passage=p) + style.tail) for p in passages]
    params = SamplingParams(temperature=temperature, top_p=0.9,
                            max_tokens=style.max_tokens)
    out = llm.generate(prompts, params, use_tqdm=False)
    return [_clean(o.outputs[0].text) for o in out]


def probe_speed(llm, chunks: list[dict], sample: int = 100, seed: int = 13) -> dict:
    """Замер скорости: сколько вопросов в секунду и во что это выльется.

    Ставится до всякой генерации. Без него объём обучающей выборки
    назначается на глаз, и узнать, что он не помещается в сессию, можно
    только через несколько часов.
    """
    rnd = random.Random(seed)
    picked = rnd.sample(chunks, min(sample, len(chunks)))
    stats = {"sample": len(picked), "по стилям": {}}
    total = 0.0
    for style in STYLES:
        t0 = time.time()
        got = generate(llm, style, [body_of(c)[:4000] for c in picked])
        dt = time.time() - t0
        total += dt
        stats["по стилям"][style.name] = {
            "секунд": round(dt, 1),
            "вопросов в секунду": round(len(got) / dt, 2),
            "средняя длина": round(sum(len(g) for g in got) / max(1, len(got))),
        }
        print(f"   [{style.name}] {len(got)} за {dt:.0f} с "
              f"= {len(got) / dt:.2f} вопр./с", flush=True)
    rate = len(picked) * len(STYLES) / total
    stats["вопросов в секунду"] = round(rate, 2)
    stats["часов на 10000"] = round(10000 / rate / 3600, 2)
    stats["часов на 15000"] = round(15000 / rate / 3600, 2)
    print(f"\nитого {rate:.2f} вопр./с (до фильтров): "
          f"10 000 за {stats['часов на 10000']:.1f} ч, "
          f"15 000 за {stats['часов на 15000']:.1f} ч", flush=True)
    print("объём обучающей выборки назначается по этому числу, а не по плану",
          flush=True)
    return stats


def make_queries(llm, chunks: list[dict], out_path: str, target: int,
                 prefix: str = "syn", per_act: int = 12, seed: int = 13,
                 batch: int = 256, idf_texts: list[str] | None = None) -> dict:
    """Сгенерировать и отфильтровать вопросы. Возвращает статистику.

    Обратная частота слов считается по всему корпусу, а не по обучающей
    части: фильтр «подходит к слишком многим фрагментам» должен мерить
    общность слова в языке актов, а не в отобранной половине.
    """
    idf = IdfTable(idf_texts if idf_texts is not None else [c["text"] for c in chunks])
    good = [c for c in chunks if is_good_source(c)]
    random.Random(seed).shuffle(good)
    print(f"фрагментов дано {len(chunks)}, пригодных как источник {len(good)} "
          f"({100 * len(good) / max(1, len(chunks)):.0f}%)", flush=True)

    out: list[dict] = []
    rejected: collections.Counter = collections.Counter()
    per_act_count: collections.Counter = collections.Counter()
    by_style: collections.Counter = collections.Counter()
    t0 = time.time()
    pos = 0
    while len(out) < target and pos < len(good):
        batch_chunks: list[dict] = []
        while len(batch_chunks) < batch and pos < len(good):
            ch = good[pos]
            pos += 1
            if per_act_count[ch["act_id"]] >= per_act:
                continue
            batch_chunks.append(ch)
        if not batch_chunks:
            break
        style = STYLES[next_style(by_style)]
        questions = generate(llm, style, [body_of(c)[:4000] for c in batch_chunks])
        for ch, q in zip(batch_chunks, questions):
            passage = body_of(ch)
            ok, why = judge_query(q, passage, idf,
                                  require_question_mark=style.question_mark)
            if not ok:
                rejected[why.split("(")[0].strip()] += 1
                continue
            run, share = copy_score(q, passage)
            per_act_count[ch["act_id"]] += 1
            by_style[style.name] += 1
            out.append({
                "query_id": f"{prefix}{len(out):05d}", "origin": "синтетический",
                "style": style.name, "text": q,
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

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for q in out:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    attempts = len(out) + sum(rejected.values())
    stats = {"accepted": len(out), "rejected": sum(rejected.values()),
             "reject_share": round(sum(rejected.values()) / max(1, attempts), 2),
             "reasons": dict(rejected.most_common()), "by_style": dict(by_style),
             "acts_touched": len(per_act_count), "seconds": round(time.time() - t0, 1)}
    print(f"\nпринято {stats['accepted']}, отклонено {stats['rejected']} "
          f"({100 * stats['reject_share']:.0f}% попыток)", flush=True)
    print(f"причины отказа: {stats['reasons']}", flush=True)
    print(f"по стилям: {stats['by_style']}", flush=True)
    print(f"актов затронуто: {stats['acts_touched']}, "
          f"время {stats['seconds'] / 60:.0f} мин", flush=True)
    return stats


def next_style(by_style: collections.Counter) -> int:
    """Следующий стиль — тот, которого принято меньше всего.

    Не по кругу: стили отсеиваются фильтрами по-разному (короткий запрос
    чаще ловится на «подходит к слишком многим»), и ровное чередование
    пачек дало бы перекос в принятом.
    """
    counts = [by_style[s.name] for s in STYLES]
    return counts.index(min(counts))


def filter_by_teacher(queries: list[dict], chunks: list[dict], depth: int = 100,
                      keep_top: int | None = None, batch_size: int = 64,
                      device: str = "cuda") -> tuple[list[dict], dict]:
    """Третий фильтр: учитель обязан видеть исходный фрагмент в первых
    `keep_top` из `depth` кандидатов BM25.

    **Порог считается от глубины, а не задан числом, и это не мелочь.**
    Эталон всегда среди кандидатов: если BM25 его не нашёл, его вставляют
    силой. Значит после переранжирования его место — число от 1 до `depth`.
    При жёстком пороге 50 и глубине 30 условие «место не хуже 50-го»
    выполняется всегда, фильтр не отбрасывает ничего и стоит при этом час
    с лишним видеокарты, а отчёт показывает бодрое «принято 12000 из 12000».
    Ровно это и было заложено в план урезания: глубину предлагалось снизить
    со 100 до 30, оставив порог 50, то есть выключить фильтр, думая,
    что он лишь подешевел.

    Поэтому по умолчанию порог — половина глубины, как было при исходных
    100 и 50. Отношение и есть смысл: эталон обязан попасть в лучшую
    половину того, что показали учителю.

    Зачем он нужен помимо двух прежних. Первые два смотрят на пару
    «вопрос и фрагмент» и ловят списывание и общие слова. Они пропускают
    вопрос, на который в корпусе лучше отвечает другой фрагмент: такой
    вопрос формально хорош, а как обучающий пример он вреден — модель
    учится притягивать к нему не тот текст.

    Цена честно считается заранее: `len(queries) * depth` проходов
    кросс-энкодера. На пятнадцати тысячах вопросов при глубине 100 это
    полтора миллиона пар, то есть часы. Глубину имеет смысл уменьшать,
    а не отказываться от фильтра.
    """
    import gpu_search as S

    from rufin.retrieval.bm25 import BM25Index

    if keep_top is None:
        keep_top = max(1, depth // 2)
    if keep_top >= depth:
        raise ValueError(
            f"порог {keep_top} не меньше глубины {depth}: эталон всегда "
            f"попадает в кандидатов, поэтому такой фильтр не отбрасывает "
            f"ничего и только тратит время. Уменьшите порог или уберите "
            f"--teacher-filter совсем.")
    print(f"третий фильтр: глубина {depth}, порог {keep_top} "
          f"(эталон обязан попасть в первые {keep_top} из {depth})", flush=True)

    texts = {c["chunk_id"]: c["text"] for c in chunks}
    t0 = time.time()
    index = BM25Index.build([c["chunk_id"] for c in chunks],
                            [c["text"] for c in chunks])
    print(f"BM25 для фильтра построен за {time.time() - t0:.0f} с", flush=True)
    print(f"проходов кросс-энкодера: {len(queries) * depth}", flush=True)

    candidates: dict[str, list[tuple[str, float]]] = {}
    for q in queries:
        found = [c for c, _ in index.search(q["text"], depth)]
        if q["gold_chunk_id"] not in found:
            found = found[:depth - 1] + [q["gold_chunk_id"]]
        candidates[q["query_id"]] = [(c, 0.0) for c in found]

    ranked = S.rerank_runs([{"query_id": q["query_id"], "text": q["text"]} for q in queries],
                           candidates, texts, batch_size=batch_size, device=device)

    kept, positions = [], []
    for q in queries:
        order = ranked.get(q["query_id"], [])
        place = order.index(q["gold_chunk_id"]) + 1 if q["gold_chunk_id"] in order else 0
        if place and place <= keep_top:
            q = dict(q, teacher_rank=place)
            kept.append(q)
            positions.append(place)
    stats = {"на входе": len(queries), "принято": len(kept),
             "отброшено": len(queries) - len(kept),
             "глубина": depth, "порог": keep_top,
             "медиана места эталона": sorted(positions)[len(positions) // 2] if positions else 0}
    print(f"третий фильтр: принято {stats['принято']} из {stats['на входе']}, "
          f"медиана места эталона {stats['медиана места эталона']}", flush=True)
    return kept, stats
