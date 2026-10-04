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

# Запрет на обращения к HuggingFace. Ставится до первого импорта
# transformers, иначе библиотека успеет прочитать настройки.
#
# Это не предпочтение, а вывод из потери суток. Секрет с токеном через API
# к ядру не прицепить, значит любое обращение к хабу идёт неавторизованным,
# а такое обращение не падает, а молча встаёт. Будильник его не снимает:
# процесс стоит внутри чужого кода, куда сигнал не доходит. Два прогона
# по двенадцать часов простояли ровно так.
#
# С этими переменными случайное обращение падает сразу и называет себя.
# Отказ дешевле зависания в двенадцать часов на три порядка.
for _ключ in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ.setdefault(_ключ, "1")

# Видна одна карта, и это тоже не предпочтение.
#
# Узел даёт две, но библиотека обучения сама заворачивает модель
# в `DataParallel`, как только видит больше одной. Тот же `DataParallel`
# уже уводил прогон в зависание на первой же партии, а в обучении он
# вдобавок падает по памяти, потому что держит копию на каждой карте.
#
# Прятать вторую карту надёжнее, чем просить библиотеку её не трогать:
# просить пришлось бы в каждом месте, где модель попадает в чужие руки.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from rufin.chunk_configs import BY_NAME  # noqa: E402
from rufin.chunking import TokenRuler, chunk_act  # noqa: E402

TOKENIZER = "xlm-roberta-base"


def make_ruler(path: str | None = None) -> TokenRuler:
    """Токенизатор для нарезки.

    Нужен только там, где нарезка считается заново. Когда она берётся
    готовой, а в этапе D она берётся готовой всегда, вызывать это нельзя:
    имя `xlm-roberta-base` потянуло бы загрузку из хаба, а хаб запрещён.
    """
    return TokenRuler(_токенизатор(path or TOKENIZER))


def _токенизатор(path: str):
    from transformers import AutoTokenizer
    if "/" not in path.strip("/") or path.startswith("/"):
        return AutoTokenizer.from_pretrained(path, local_files_only=True)
    найдено = resolve_model(path, обязательно=False)
    if найдено == path:
        raise SystemExit(
            f"токенизатор {path} пришлось бы качать из хаба, а хаб запрещён. "
            f"Подключите его моделью Kaggle или не пересчитывайте нарезку: "
            f"готовая кладётся в кэш и токенизатора не требует.")
    return AutoTokenizer.from_pretrained(найдено, local_files_only=True)


def resolve_acts(path: str | None = None) -> str:
    """Найти файл корпуса.

    Kaggle иногда разжимает архивы при создании датасета, и `acts.jsonl.gz`
    превращается в `acts.jsonl`. Поэтому путь не задаётся жёстко: проверяются
    оба варианта, а если не указан вовсе — корпус ищется среди подключённых
    входов и в рабочей папке.

    Глубина монтирования тоже не задаётся. Датасет приезжает то как
    `/kaggle/input/<слаг>/`, то как `/kaggle/input/datasets/<кто>/<слаг>/`,
    и зависит это не от нас. Поиск по одному уровню находил корпус годами,
    а потом перестал — обход по дереву не ломается от переезда.
    """
    candidates: list[str] = []
    if path:
        candidates += [path, path[:-3] if path.endswith(".gz") else path + ".gz"]
    candidates += sorted(glob.glob("/kaggle/input/**/acts.jsonl*", recursive=True),
                         key=len)
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


def build_chunks_cached(acts: list[dict], config: str, ruler,
                        cache_dir: str | None) -> list[dict]:
    """Нарезка с сохранением на диск.

    Нарезка детерминирована и зависит только от корпуса и версии токенизатора,
    но занимает минуты на каждую конфигурацию, и на видеокарте это время
    простоя. Готовую кладём рядом: повторный прогон в той же сессии,
    а при подключении вывода — и в следующей, её не пересчитывает.
    """
    # Токенизатор может прийти функцией, а не объектом: тогда он
    # поднимается только если нарезку правда надо считать. Готовая нарезка
    # токенизатора не требует вовсе, а его загрузка пошла бы в хаб.
    def линейка():
        return ruler() if callable(ruler) else ruler

    if not cache_dir:
        return build_chunks(acts, config, линейка())
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
    out = build_chunks(acts, config, линейка())
    with open(path, "w", encoding="utf-8") as f:
        for c in out:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return out


def pick_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_model(hf_id: str, root: str = "/kaggle/input",
                  обязательно: bool = True) -> str:
    """Путь к весам: сначала модель, подключённая входом, потом HuggingFace.

    Зачем вообще искать локально. Скачивание с HuggingFace без токена
    режется по скорости, а секреты Kaggle через API не прицепить вовсе:
    ядро, созданное командой, токена не увидит никогда. Модели же
    подключаются входом как есть, мгновенно и без сети.

    Ищется не по известному пути, а по содержимому: каталог с `config.json`,
    в пути которого встречается имя модели. Раскладка у входов разная
    (`<модель>/<каркас>/<вариант>/<версия>`), и задавать её жёстко —
    тот же способ потерять час, что и с датасетами.

    Если ничего не нашлось, возвращается имя на HuggingFace: это рабочий
    запасной путь, а не ошибка. Какой из двух сработал, печатается —
    молча подменять источник весов нельзя.
    """
    имя = hf_id.split("/")[-1].lower()
    находки = []
    for path in glob.glob(os.path.join(root, "**", "config.json"), recursive=True):
        каталог = os.path.dirname(path)
        if имя in каталог.lower().replace("_", "-"):
            находки.append(каталог)
    if not находки:
        if not обязательно:
            return hf_id
        raise SystemExit(
            f"модель {hf_id} не подключена входом, а качать её из хаба нельзя. "
            f"Хаб запрещён не из принципа: токен к ядру не прицепить, "
            f"неавторизованное обращение не падает, а молча встаёт, и два "
            f"прогона по двенадцать часов уже простояли так. Подключите "
            f"модель через + Add Input -> Models.")
    # самый короткий путь: корень модели, а не вложенная папка вроде onnx/
    выбран = sorted(находки, key=lambda p: (len(p.split(os.sep)), len(p)))[0]
    print(f"модель {hf_id}: взята из входов, {выбран}", flush=True)
    return выбран


def half_kwargs(device: str) -> dict:
    """Параметры половинной точности.

    Имя параметра у transformers менялось: было `torch_dtype`, стало `dtype`,
    и в пятой версии старое имя не действует вовсе.

    **Разбором подписи это определить нельзя, и прежняя попытка была
    бесполезной.** У `AutoModel.from_pretrained` в подписи стоит только
    `*model_args, **kwargs`, поэтому проверка «есть ли среди параметров
    dtype» всегда отвечала «нет» и всегда выбирала старое имя. На старых
    версиях это работало по совпадению. На пятой — перестало, и перестало
    молча: модель грузится в fp32, ошибки нет, а счёт идёт втрое дольше.

    Поэтому имя выбирается по версии библиотеки, а результат ещё и
    проверяется на самой модели (`ensure_half`): предполагать здесь нечего,
    цена ошибки — часы.
    """
    if device == "cpu":
        return {}
    import torch
    import transformers
    старшая = int(transformers.__version__.split(".")[0])
    return {("dtype" if старшая >= 5 else "torch_dtype"): torch.float16}


def ensure_half(model, device: str):
    """Убедиться, что модель действительно в половинной точности.

    Проверка на самой модели, а не на намерении: параметр мог быть
    проигнорирован, переименован или просто не доехать. На T4 разница
    между fp16 и fp32 — втрое по времени, и узнавать о ней по длительности
    прогона слишком дорого.
    """
    if device == "cpu":
        return model
    import torch
    было = next(model.parameters()).dtype
    if было is torch.float32:
        print(f"   точность приехала {было}, перевожу в float16: "
              f"на T4 это втрое по времени", flush=True)
        model = model.half()
    else:
        print(f"   точность: {было}", flush=True)
    return model


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
