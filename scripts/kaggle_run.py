#!/usr/bin/env python3
"""Управление вычислениями на Kaggle из командной строки.

Тяжёлые шаги считаются на чужой видеокарте, и до сих пор это означало
ручные действия в браузере: загрузить датасет, создать ноутбук, подключить
входы, включить ускоритель, дождаться, скачать. Каждый такой шаг — место,
где легко ошибиться и трудно повторить.

Официальный Kaggle CLI умеет всё то же самое, поэтому прогон описывается
файлом и запускается одной командой. Ноутбук собирается из скрипта этапа:
на Kaggle уезжает не рукописная тетрадь, а тот же код, что лежит в проекте.

Команды:
  dataset   обновить датасет с корпусом и кодом
  queries   загрузить набор запросов отдельным датасетом
  weights   загрузить обученные веса отдельным датасетом
  run       собрать ноутбук, запустить и дождаться
  fetch     забрать результат последнего прогона
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DIST = os.path.join(ROOT, "dist")
PKG = os.path.join(DIST, "kaggle")

DATASET = "ru-fin-retrieval"
QUERIES_DATASET = "ru-fin-queries"
WEIGHTS_DATASET = "ru-fin-weights"

# Модели, подключаемые входом вместо скачивания с HuggingFace. Секреты Kaggle
# через API не прицепить вовсе — ядро, созданное командой, токена не увидит, —
# а без токена скачивание режется по скорости. Входом модель приезжает
# мгновенно и без сети.
#
# Это чужие зеркала, поэтому `gpu_prepare.py` их сверяет: устройство модели
# против ожидаемого и, главное, воспроизведение уже посчитанной выдачи
# `base__hybrid-rerank.jsonl`. Совпали первые места — тот самый учитель,
# которым измерено 0,690. Не совпали — прогон отказывается считать.
MODELS = {
    # зеркало intfloat/multilingual-e5-small, закреплено за коммитом
    # 614241f622f53c4eeff9890bdc4f31cfecc418b3
    "e5-small": "dangkhoa2016/intfloat-multilingual-e5-small/transformers/default/1",
    # зеркало BAAI/bge-reranker-v2-m3, 2,29 ГБ — размер весов в fp32
    "reranker": "andreasbis/baai-bge-reranker-v2-m3/transformers/default/1",
}


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("  $ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, text=True, capture_output=True, **kw)


def require_credentials() -> str:
    """Имя пользователя Kaggle; заодно проверка, что ключ на месте.

    Ключ бывает в двух видах: старый kaggle.json с парой «имя и ключ»
    и новый access_token, где имени нет вовсе. Поэтому имя спрашивается
    у самого клиента, а не вычитывается из файла.
    """
    home = os.path.expanduser("~/.kaggle")
    if not any(os.path.exists(os.path.join(home, n))
               for n in ("kaggle.json", "access_token")):
        raise SystemExit(
            "ключ Kaggle не найден. Взять: kaggle.com -> Settings -> API -> "
            "Create New Token. Положить в ~/.kaggle/access_token "
            "(или ~/.kaggle/kaggle.json) и выставить права 600.")
    probe = subprocess.run(["kaggle", "config", "view"], text=True, capture_output=True)
    for line in (probe.stdout or "").splitlines():
        if line.strip().startswith("- username:"):
            name = line.split(":", 1)[1].strip()
            if name and name != "None":
                return name
    raise SystemExit("клиент Kaggle не сообщает имя пользователя: проверьте ключ")


def push_dataset(user: str, folder: str, slug: str, title: str, message: str) -> None:
    meta = {"title": title, "id": f"{user}/{slug}", "licenses": [{"name": "unknown"}]}
    with open(os.path.join(folder, "dataset-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    probe = run(["kaggle", "datasets", "status", f"{user}/{slug}"])
    exists = probe.returncode == 0 and "error" not in probe.stdout.lower()
    if exists:
        r = run(["kaggle", "datasets", "version", "-p", folder, "-m", message,
                 "-r", "zip", "--dir-mode", "zip"])
    else:
        r = run(["kaggle", "datasets", "create", "-p", folder, "-r", "zip",
                 "--dir-mode", "zip"])
    print((r.stdout or r.stderr).strip())
    if r.returncode != 0:
        raise SystemExit("не удалось загрузить датасет")


def notebook_from_script(script: str, args_line: str) -> dict:
    """Ноутбук из двух ячеек: подготовка окружения и запуск скрипта этапа."""
    setup = [
        "import glob, os, shutil, subprocess, sys, zipfile\n",
        "\n",
        "# Без токена HuggingFace режет скорость скачивания, и на больших моделях\n",
        "# это превращается в многочасовое ожидание. Токен кладётся в секреты\n",
        "# ноутбука под именем HF_TOKEN; переменная наследуется дочерним процессом.\n",
        "# Имён два: у библиотеки старое HUGGING_FACE_HUB_TOKEN и новое HF_TOKEN.\n",
        "try:\n",
        "    from kaggle_secrets import UserSecretsClient\n",
        "    _tok = UserSecretsClient().get_secret('HF_TOKEN')\n",
        "    os.environ['HF_TOKEN'] = _tok\n",
        "    os.environ['HUGGING_FACE_HUB_TOKEN'] = _tok\n",
        "    print('токен HuggingFace подключён')\n",
        "except Exception as e:\n",
        "    print('токен HuggingFace недоступен, скачивание будет медленным:', e)\n",
        "\n",
        "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U',\n",
        "                'sentence-transformers'], check=False)\n",
        "\n",
        "# Вывод предыдущего этапа тоже лежит среди входов и содержит копию кода:\n",
        "# ноутбук этапа A копировал датасет в рабочую папку, и всё это стало его\n",
        "# выводом. Если взять код оттуда, приедет старая версия. Поэтому источники\n",
        "# ранжируются: каталог со следами прогона (__notebook__.ipynb, embeddings)\n",
        "# идёт последним.\n",
        "def find_source(root='/kaggle/input', max_depth=4):\n",
        "    found, archives = [], []\n",
        "    for cur, dirs, files in os.walk(root):\n",
        "        if cur.count('/') - root.count('/') >= max_depth:\n",
        "            dirs.clear()\n",
        "        if 'rufin' in dirs:\n",
        "            stale = ('__notebook__.ipynb' in files or 'embeddings' in dirs\n",
        "                     or '__output__.json' in files)\n",
        "            found.append((1 if stale else 0, cur))\n",
        "        archives += [os.path.join(cur, f) for f in files if f.endswith('.zip')]\n",
        "    if found:\n",
        "        found.sort()\n",
        "        return found[0][1], None\n",
        "    return (os.path.dirname(archives[0]), archives[0]) if archives else (None, None)\n",
        "\n",
        "src, archive = find_source()\n",
        "assert src, 'датасет с кодом не подключён'\n",
        "print('источник:', src)\n",
        "if archive:\n",
        "    with zipfile.ZipFile(archive) as z:\n",
        "        z.extractall('/kaggle/working')\n",
        "else:\n",
        "    for item in glob.glob(os.path.join(src, '*')):\n",
        "        dst = os.path.join('/kaggle/working', os.path.basename(item))\n",
        "        if os.path.isdir(item):\n",
        "            shutil.copytree(item, dst, dirs_exist_ok=True)\n",
        "        else:\n",
        "            shutil.copy2(item, dst)\n",
        "os.chdir('/kaggle/working')\n",
        "print('в рабочей папке:', sorted(os.listdir('.'))[:12])\n",
        "\n",
        "# Каноническая нарезка кладётся туда, откуда скрипты берут готовую.\n",
        "# Пересчитывать её здесь нельзя: нарезка зависит от версии токенизатора,\n",
        "# а установка vLLM тянет свой transformers и сдвигает границы — 61 922\n",
        "# фрагмента вместо 62 594 на том же корпусе. Идентификатор позиционный\n",
        "# («акт#номер»), поэтому при сдвиге он не исчезает, а начинает указывать\n",
        "# на другой текст, и эталон съезжает молча. Кэш сносится: упавший прогон\n",
        "# успевает записать туда свою нарезку, а готовая берётся раньше новой.\n",
        "готовая = (glob.glob('/kaggle/input/**/chunks_base.jsonl', recursive=True)\n",
        "           + glob.glob('/kaggle/input/**/chunks_base.jsonl.gz', recursive=True))\n",
        "if готовая:\n",
        "    import gzip\n",
        "    os.makedirs('/kaggle/working/chunks', exist_ok=True)\n",
        "    цель = '/kaggle/working/chunks/base.jsonl'\n",
        "    if os.path.exists(цель):\n",
        "        os.remove(цель)\n",
        "    opener = gzip.open if готовая[0].endswith('.gz') else open\n",
        "    with opener(готовая[0], 'rb') as fi, open(цель, 'wb') as fo:\n",
        "        shutil.copyfileobj(fi, fo, length=1 << 20)\n",
        "    print('нарезка из входов:', sum(1 for _ in open(цель, encoding='utf-8')),\n",
        "          'фрагментов')\n",
        "else:\n",
        "    print('ВНИМАНИЕ: канонической нарезки среди входов нет, будет посчитана '\n",
        "          'заново — для этапа C это недопустимо')\n",
        "\n",
        "# vLLM поднимает движок отдельным процессом и по умолчанию форкается,\n",
        "# а форк не годится, когда CUDA уже тронута в родителе.\n",
        "os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')\n",
    ]
    launch = [
        "import os, subprocess, sys\n",
        f"cmd = [sys.executable, {script!r}] + {args_line!r}.split()\n",
        "print('запуск:', ' '.join(cmd), flush=True)\n",
        "env = dict(os.environ, VLLM_WORKER_MULTIPROC_METHOD='spawn')\n",
        "p = subprocess.run(cmd, cwd='/kaggle/working', env=env)\n",
        "raise SystemExit(p.returncode)\n",
    ]
    cells = [{"cell_type": "code", "metadata": {}, "source": s,
              "outputs": [], "execution_count": None} for s in (setup, launch)]
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                        "name": "python3"},
                         "language_info": {"name": "python", "version": "3.11"}},
            "nbformat": 4, "nbformat_minor": 5}


def push_kernel(user: str, slug: str, title: str, script: str, script_args: str,
                dataset_sources: list[str], kernel_sources: list[str],
                gpu: bool = True, machine: str = "", notebook: str = "",
                model_sources: list[str] | None = None) -> str:
    folder = os.path.join(DIST, "kernel_" + slug)
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder)
    if notebook:
        # Готовая тетрадь из проекта вместо собранной на лету. Нужна там, где
        # прогон — не одна команда, а порядок шагов с проверками между ними:
        # этап C ставит библиотеки, тянет веса под предохранителем, гоняет
        # пробу и только потом считает. Повторять это генератором из двух
        # ячеек значило бы держать одну логику в двух местах, и они разойдутся.
        shutil.copyfile(notebook, os.path.join(folder, "run.ipynb"))
        print(f"  тетрадь: {os.path.relpath(notebook, ROOT)}")
    else:
        with open(os.path.join(folder, "run.ipynb"), "w", encoding="utf-8") as f:
            json.dump(notebook_from_script(script, script_args), f,
                      ensure_ascii=False, indent=1)
    # machine_shape задаёт конфигурацию узла. Пустое значение означает выбор
    # по умолчанию — узел с двумя T4, который дефицитнее и потому дольше ждёт
    # очереди. Наш код работает с одной картой, поэтому просим одиночную.
    meta = {
        "id": f"{user}/{slug}", "title": title, "code_file": "run.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": gpu, "enable_tpu": False, "enable_internet": True,
        "machine_shape": machine,
        "dataset_sources": dataset_sources, "kernel_sources": kernel_sources,
        "competition_sources": [], "model_sources": model_sources or [],
    }
    if model_sources:
        for m in model_sources:
            print(f"  модель входом: {m}")
    with open(os.path.join(folder, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    # Новая версия не отменяет запуск предыдущей: Kaggle оставляет обе считаться,
    # и обе расходуют квоту. Отмены в клиенте нет, остановить можно только
    # в браузере, поэтому хотя бы предупреждаем.
    probe = run(["kaggle", "kernels", "status", f"{user}/{slug}"])
    state = (probe.stdout or "").lower()
    # очередь опаснее счёта: версия, стоящая в очереди, всё равно запустится
    if "running" in state or "queued" in state:
        print(f"  ВНИМАНИЕ: предыдущий запуск {user}/{slug} ещё не завершён "
              f"(в очереди или считается) и после нового пуша не отменится. "
              f"Остановить можно только в браузере: меню «...» у БОЛЕЕ РАННЕЙ "
              f"версии -> Stop.")

    r = run(["kaggle", "kernels", "push", "-p", folder])
    print((r.stdout or r.stderr).strip())
    if r.returncode != 0:
        raise SystemExit("не удалось запустить ноутбук")
    return f"{user}/{slug}"


def wait(ref: str, poll: int = 60, limit_hours: float = 12.0) -> str:
    print(f"\nжду завершения {ref}; опрос раз в {poll} с", flush=True)
    started = time.monotonic()
    last = ""
    while time.monotonic() - started < limit_hours * 3600:
        r = run(["kaggle", "kernels", "status", ref])
        out = (r.stdout or r.stderr).strip()
        if out != last:
            print("  " + out.replace("\n", " "), flush=True)
            last = out
        low = out.lower()
        if "complete" in low:
            return "complete"
        if "error" in low or "cancel" in low:
            return "error"
        time.sleep(poll)
    return "timeout"


def fetch(ref: str, dest: str) -> None:
    os.makedirs(dest, exist_ok=True)
    r = run(["kaggle", "kernels", "output", ref, "-p", dest])
    print((r.stdout or r.stderr).strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("dataset", help="обновить датасет с корпусом и кодом")
    p.add_argument("-m", "--message", default="обновление кода")

    p = sub.add_parser("queries", help="загрузить набор запросов")

    p = sub.add_parser("weights", help="загрузить обученные веса")
    p.add_argument("метки", nargs="*", default=None,
                   help="какие этапы заливать; по умолчанию все из data/weights")

    p = sub.add_parser("run", help="собрать ноутбук, запустить и дождаться")
    p.add_argument("stage", choices=["a", "b", "c", "d0", "d", "export", "rerank"])
    p.add_argument("--slug", default=None)
    p.add_argument("--source", default="ru-fin",
                   help="ядро, чей вывод подключается: там лежат матрицы этапа A")
    p.add_argument("--args", default="", help="аргументы скрипта этапа, строкой")
    p.add_argument("--no-wait", action="store_true")
    p.add_argument("--machine", default="p100",
                   help="конфигурация узла: p100 — одна карта, быстрее получить; "
                        "пусто — по умолчанию T4 x2, дефицитнее")
    p.add_argument("--notebook", default=None,
                   help="запустить готовую тетрадь из notebooks/ вместо собранной "
                        "из скрипта этапа")
    p.add_argument("--kernels", nargs="*", default=None,
                   help="вывод каких ядер подключить входом, помимо обычного. "
                        "Нужно продолжению рецепта: журнал прежнего прогона "
                        "лежит в выводе предыдущего ядра")
    p.add_argument("--no-gpu", action="store_true",
                   help="считать без видеокарты: квота GPU не тратится, а ядер "
                        "процессора сессии достаётся больше")

    p = sub.add_parser("fetch", help="забрать результат")
    p.add_argument("slug")
    p.add_argument("--dest", default=os.path.join(ROOT, "data", "kaggle_out"))

    args = ap.parse_args()
    user = require_credentials()
    print(f"пользователь Kaggle: {user}\n")

    if args.cmd == "dataset":
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts",
                                                     "make_kaggle_package.py")], check=True)
        push_dataset(user, PKG, DATASET, "ru-fin-retrieval", args.message)

    elif args.cmd == "queries":
        # Вместе с запросами уезжает то, что нужно маленьким прогонам:
        # готовые выдачи и фрагменты базовой нарезки. Так переранжирование
        # обходится без повторной нарезки корпуса.
        folder = os.path.join(DIST, "queries")
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder)
        # Сплит и лист пула — не «на всякий случай», а условие запуска: без
        # сплита этап C откажется генерировать (вопросы ушли бы на акты теста,
        # и утечку потом не отследить), без пула судье нечего размечать.
        # Отсутствие любого из них — не молчаливый пропуск, а предупреждение:
        # иначе оно всплывёт на видеокарте, когда квота уже пошла.
        # Этапу D нужны ещё обучающая выборка, dev, разметка и состав
        # подвыборок: dev на видеокарте обязан считаться тем же составом
        # и тем же эталоном, что в локальном отчёте.
        for name in ("queries.jsonl", "split.json", "pool_candidates.tsv",
                     "synthetic_train.jsonl", "synthetic_dev.jsonl",
                     "qrels.tsv", "podvyborki.json"):
            src = os.path.join(ROOT, "data", "queries", name)
            if os.path.exists(src):
                shutil.copy2(src, folder)
            elif name == "queries.jsonl":
                raise SystemExit("нет data/queries/queries.jsonl: `make queries`")
            else:
                print(f"  ВНИМАНИЕ: {name} нет, уезжает датасет без него")
        # Переранжированная выдача едет не для счёта, а для сверки: ею
        # проверяется, что подключённое зеркало учителя — та самая модель,
        # которой измерено 0,690.
        for name in ("base__bm25.jsonl", "base__dense-bge-m3.jsonl",
                     "base__hybrid.jsonl", "base__hybrid-rerank.jsonl"):
            src = os.path.join(ROOT, "data", "runs", name)
            if os.path.exists(src):
                shutil.copy2(src, folder)
        # Журнал прежнего прогона и подготовка едут датасетом, а не входом
        # от ядра. Kaggle не отдаёт вывод упавшего ядра как источник, и
        # продолжение рецепта на этом спотыкается ровно тогда, когда оно
        # нужнее всего — после падения.
        for rel in ("docs/raw/journal.json", "data/queries/train_prepared.jsonl"):
            src = os.path.join(ROOT, rel)
            if os.path.exists(src):
                shutil.copy2(src, folder)
                print(f"  в датасет: {os.path.basename(rel)}")

        chunks = os.path.join(ROOT, "data", "chunks", "base.jsonl")
        if os.path.exists(chunks):
            import gzip
            with open(chunks, "rb") as fi, gzip.open(
                    os.path.join(folder, "chunks_base.jsonl.gz"), "wb", compresslevel=6) as fo:
                shutil.copyfileobj(fi, fo, length=1 << 20)
        total = sum(os.path.getsize(os.path.join(folder, f)) for f in os.listdir(folder))
        print(f"в датасет уходит {len(os.listdir(folder))} файлов, "
              f"{total / 1048576:.0f} МБ")
        push_dataset(user, folder, QUERIES_DATASET, "ru-fin-queries",
                     "запросы, готовые выдачи и фрагменты")

    elif args.cmd == "weights":
        # Веса едут датасетом, а не выводом ядра. Вывод ядра не переживает
        # ни падения этого ядра, ни переход на другой аккаунт, а финальный
        # прогон без весов лучшего этапа сделать нельзя.
        источник = os.path.join(ROOT, "data", "weights")
        if not os.path.isdir(источник):
            raise SystemExit(f"нет {os.path.relpath(источник, ROOT)}: "
                             f"привезите веса с Kaggle")
        метки = args.метки or sorted(os.listdir(источник))
        folder = os.path.join(DIST, "weights")
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder)
        всего = 0
        for метка in метки:
            откуда = os.path.join(источник, метка)
            if not os.path.isdir(откуда):
                raise SystemExit(f"нет весов этапа {метка}")
            shutil.copytree(откуда, os.path.join(folder, метка))
            размер = sum(os.path.getsize(os.path.join(б, и))
                         for б, _, ф in os.walk(откуда) for и in ф)
            всего += размер
            print(f"   {метка:<16} {размер / 1048576:>7.0f} МБ")
        print(f"   {'итого':<16} {всего / 1048576:>7.0f} МБ")
        push_dataset(user, folder, WEIGHTS_DATASET, "ru-fin-weights",
                     "веса этапов рецепта: " + ", ".join(метки))

    elif args.cmd == "run":
        # вывод этапа A подключается как источник: там лежат матрицы эмбеддингов
        source = [f"{user}/{args.source}"] if args.source else []
        stages = {
            "a": ("run_phase_a.py", "ru-fin-phase-a", "ru-fin phase A", [], True),
            "export": ("export_chunks.py", "ru-fin-export-chunks", "ru-fin export chunks",
                       source, True),
            "b": ("run_phase_b.py", "ru-fin-phase-b", "ru-fin phase B", source, True),
            # Переранжирование отдельным маленьким прогоном: всё остальное
            # уже посчитано, и повторять его незачем.
            "rerank": ("gpu_rerank.py", "ru-fin-rerank-only", "ru fin rerank only", [], True),
            # Этап C поднимает языковую модель сам и матрицы этапа A не трогает:
            # вывод прежнего ядра ему не нужен, а подключённый — только лишние
            # гигабайты на монтирование и лишний источник старого кода.
            "c": ("run_phase_c.py", "ru-fin-phase-c", "ru-fin phase C", [], True),
            # D0 считается отдельным прогоном: он самый дорогой в этапе,
            # его файл нужен сразу двум этапам рецепта, и повторять его
            # ради перезапуска обучения незачем.
            "d0": ("gpu_prepare.py", "ru-fin-d0", "ru fin d0", [], True),
            # Рецепту нужен вывод подготовки: трудные негативы и оценки
            # учителя считаются один раз и подключаются входом.
            "d": ("run_phase_d.py", "ru-fin-phase-d", "ru-fin phase D",
                  [f"{user}/ru-fin-d0"], True),
        }
        script, slug, title, kernels, gpu = stages[args.stage]
        if args.slug:
            # Kaggle требует, чтобы заголовок приводился к слагу, иначе
            # отказывается принимать ядро
            slug = args.slug
            title = args.slug.replace("-", " ")
        if args.no_gpu:
            gpu = False
            # Отдельное имя: иначе прогон без карты перезапишет версию с картой.
            # Заголовок правим вместе со слагом — Kaggle требует, чтобы одно
            # выводилось из другого, и иначе отказывается принимать ядро.
            if not args.slug:
                slug = f"{slug}-cpu"
                title = f"{title} cpu"
        datasets = [f"{user}/{DATASET}"]
        if args.stage in ("b", "c", "d0", "d", "rerank"):
            datasets.append(f"{user}/{QUERIES_DATASET}")
        # Веса подключаются, если датасет с ними есть. Финалу они нужны,
        # остальным шагам безразличны: каждый этап обучается с исходных.
        if args.stage == "d":
            проба = run(["kaggle", "datasets", "status",
                         f"{user}/{WEIGHTS_DATASET}"])
            if проба.returncode == 0 and "error" not in (проба.stdout or "").lower():
                datasets.append(f"{user}/{WEIGHTS_DATASET}")
                print(f"  веса входом: {user}/{WEIGHTS_DATASET}")
        # Этапу D обе модели нужны обязательно: ученик считает плотную часть
        # выдачи, учитель — оценки. Остальным этапам модели входом не нужны,
        # у них свои источники весов.
        models = list(MODELS.values()) if args.stage in ("d0", "d") else []
        if args.kernels:
            kernels = list(kernels) + [k if "/" in k else f"{user}/{k}"
                                       for k in args.kernels]
        ref = push_kernel(user, slug, title, script, args.args, datasets, kernels,
                          gpu, args.machine, args.notebook or "", models)
        if not args.no_wait:
            state = wait(ref)
            print(f"\nсостояние: {state}")
            if state == "complete":
                fetch(ref, os.path.join(ROOT, "data", "kaggle_out", slug.split("/")[-1]))

    elif args.cmd == "fetch":
        ref = args.slug if "/" in args.slug else f"{user}/{args.slug}"
        fetch(ref, args.dest)


if __name__ == "__main__":
    main()
