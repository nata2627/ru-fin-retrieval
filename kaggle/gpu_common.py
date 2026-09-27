"""Общее для всех тяжёлых шагов на Kaggle: загрузка корпуса и нарезка.

Нарезка выполняется тем же кодом (`rufin.chunking`) и тем же токенизатором,
что локально, поэтому идентификаторы фрагментов совпадают и матрицу
эмбеддингов можно сопоставить с локальным корпусом.

Чанки не передаются между этапами файлами: все нарезки вместе весят под два
гигабайта, а пересобрать их из acts.jsonl — вопрос нескольких минут.
"""
from __future__ import annotations

import glob
import gzip
import json
import os
import time

from rufin.chunk_configs import BY_NAME
from rufin.chunking import TokenRuler, chunk_act

TOKENIZER = "xlm-roberta-base"


def make_ruler() -> TokenRuler:
    from transformers import AutoTokenizer
    return TokenRuler(AutoTokenizer.from_pretrained(TOKENIZER))


def resolve_acts(path: str | None = None) -> str:
    """Найти файл корпуса.

    Kaggle иногда разжимает архивы при создании датасета, и `acts.jsonl.gz`
    превращается в `acts.jsonl`. Поэтому путь не задаётся жёстко: проверяются
    оба варианта, а если не указан вовсе — корпус ищется среди подключённых
    входов и в рабочей папке.
    """
    candidates: list[str] = []
    if path:
        candidates += [path, path[:-3] if path.endswith(".gz") else path + ".gz"]
    candidates += sorted(glob.glob("/kaggle/input/*/acts.jsonl*"))
    candidates += sorted(glob.glob("/kaggle/working/acts.jsonl*"))
    candidates += sorted(glob.glob("acts.jsonl*"))
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        "корпус не найден. Ожидался acts.jsonl или acts.jsonl.gz среди "
        f"подключённых входов. Проверено: {candidates}")


def load_acts(path: str | None = None, limit: int = 0) -> list[dict]:
    resolved = resolve_acts(path)
    if path and resolved != path:
        print(f"корпус найден как {resolved}", flush=True)
    opener = gzip.open if resolved.endswith(".gz") else open
    with opener(resolved, "rt", encoding="utf-8") as f:
        acts = [json.loads(l) for l in f if l.strip()]
    return acts[:limit] if limit else acts


def build_chunks(acts: list[dict], config: str, ruler: TokenRuler) -> list[dict]:
    cfg = BY_NAME[config]
    t0 = time.time()
    out: list[dict] = []
    for act in acts:
        for ch in chunk_act(act, ruler, size=cfg.size, overlap_share=cfg.overlap,
                            strategy=cfg.strategy, add_heading=cfg.heading):
            out.append(ch.as_dict())
    print(f"[нарезка {config}] {len(out)} фрагментов за {time.time() - t0:.0f} с", flush=True)
    return out


def find_embeddings(root: str = "/kaggle/input") -> str | None:
    """Найти папку с матрицами эмбеддингов среди подключённых входов.

    Вывод ноутбука Kaggle монтирует не там же, где датасеты, и глубина пути
    заранее не известна. Поэтому ищем не по имени папки, а по содержимому:
    рядом с каждой матрицей лежит ids.txt, и его дед по дереву — искомый
    корень «эмбеддинги / нарезка / модель».
    """
    marks = glob.glob(os.path.join(root, "**", "ids.txt"), recursive=True)
    roots: dict[str, int] = {}
    for m in marks:
        candidate = os.path.dirname(os.path.dirname(os.path.dirname(m)))
        roots[candidate] = roots.get(candidate, 0) + 1
    if not roots:
        return None
    # если корней несколько, берём тот, где матриц больше
    return max(roots.items(), key=lambda kv: kv[1])[0]


def find_file(name: str, root: str = "/kaggle/input") -> str | None:
    """Найти файл среди подключённых входов.

    Два правила Kaggle, на которых проект терял время не раз, и потому
    ни один путь здесь не задаётся жёстко.

    Первое: входы монтируются на разной глубине. Вывод ядра лежит
    в /kaggle/input/<ядро>/, а датасет — в /kaggle/input/datasets/<кто>/<что>/.
    Поэтому обход идёт по дереву, а не по известному пути.

    Второе: при создании датасета Kaggle распаковывает архивы. Файл,
    загруженный как chunks_base.jsonl.gz, окажется chunks_base.jsonl,
    а загруженный папкой может приехать и распакованным zip-ом. Поэтому
    имя ищется в обоих видах — со сжатием и без.
    """
    names = [name]
    if name.endswith(".gz"):
        names.append(name[:-3])
    else:
        names.append(name + ".gz")
    for candidate in names:
        hits = glob.glob(os.path.join(root, "**", candidate), recursive=True)
        if hits:
            return sorted(hits, key=len)[0]
    return None


def build_chunks_cached(acts: list[dict], config: str, ruler: TokenRuler,
                        cache_dir: str | None) -> list[dict]:
    """Нарезка с сохранением на диск.

    Нарезка детерминирована и зависит только от корпуса и версии токенизатора,
    но занимает минуты на каждую конфигурацию, и на видеокарте это время
    простоя. Готовую кладём рядом: повторный прогон в той же сессии,
    а при подключении вывода — и в следующей, её не пересчитывает.
    """
    if not cache_dir:
        return build_chunks(acts, config, ruler)
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{config}.jsonl")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            out = [json.loads(l) for l in f if l.strip()]
        print(f"[нарезка {config}] взята готовая: {len(out)} фрагментов", flush=True)
        return out
    # уже посчитанная в другом прогоне и подключённая входом
    ready = find_file(f"{config}.jsonl")
    if ready and os.path.basename(os.path.dirname(ready)) == "chunks":
        with open(ready, encoding="utf-8") as f:
            out = [json.loads(l) for l in f if l.strip()]
        print(f"[нарезка {config}] взята из входов: {ready}, {len(out)} фрагментов",
              flush=True)
        return out
    out = build_chunks(acts, config, ruler)
    with open(path, "w", encoding="utf-8") as f:
        for c in out:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return out


def pick_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def half_kwargs(device: str) -> dict:
    """Параметры половинной точности.

    Имя параметра у transformers менялось: раньше `torch_dtype`, теперь `dtype`.
    Версия на Kaggle заранее не известна, поэтому подходящее подбирается пробой,
    а не угадывается.
    """
    if device == "cpu":
        return {}
    import inspect
    import torch
    from transformers import AutoModel
    name = "dtype" if "dtype" in inspect.signature(
        AutoModel.from_pretrained).parameters else "torch_dtype"
    return {name: torch.float16}


def load_encoder(path: str, device: str, max_seq_length: int = 512):
    """Загрузить би-энкодер, при неудаче с половинной точностью — без неё."""
    from sentence_transformers import SentenceTransformer
    try:
        model = SentenceTransformer(path, device=device, trust_remote_code=True,
                                    model_kwargs=half_kwargs(device))
    except TypeError:
        model = SentenceTransformer(path, device=device, trust_remote_code=True)
    model.max_seq_length = max_seq_length
    return model


def load_cross_encoder(path: str, device: str, max_length: int = 512):
    """Загрузить кросс-энкодер, при неудаче с половинной точностью — без неё."""
    from sentence_transformers import CrossEncoder
    try:
        return CrossEncoder(path, device=device, max_length=max_length,
                            model_kwargs=half_kwargs(device))
    except TypeError:
        return CrossEncoder(path, device=device, max_length=max_length)
