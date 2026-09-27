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
        "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U',\n",
        "                'sentence-transformers'], check=False)\n",
        "\n",
        "def find_source(root='/kaggle/input', max_depth=4):\n",
        "    archives = []\n",
        "    for cur, dirs, files in os.walk(root):\n",
        "        if cur.count('/') - root.count('/') >= max_depth:\n",
        "            dirs.clear()\n",
        "        if 'rufin' in dirs or any(f.startswith('acts.jsonl') for f in files):\n",
        "            return cur, None\n",
        "        archives += [os.path.join(cur, f) for f in files if f.endswith('.zip')]\n",
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
    ]
    launch = [
        "import subprocess, sys\n",
        f"cmd = [sys.executable, {script!r}] + {args_line!r}.split()\n",
        "print('запуск:', ' '.join(cmd), flush=True)\n",
        "p = subprocess.run(cmd, cwd='/kaggle/working')\n",
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
                gpu: bool = True) -> str:
    folder = os.path.join(DIST, "kernel_" + slug)
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder)
    with open(os.path.join(folder, "run.ipynb"), "w", encoding="utf-8") as f:
        json.dump(notebook_from_script(script, script_args), f, ensure_ascii=False, indent=1)
    meta = {
        "id": f"{user}/{slug}", "title": title, "code_file": "run.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": gpu, "enable_internet": True,
        "dataset_sources": dataset_sources, "kernel_sources": kernel_sources,
        "competition_sources": [], "model_sources": [],
    }
    with open(os.path.join(folder, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    # Новая версия не отменяет запуск предыдущей: Kaggle оставляет обе считаться,
    # и обе расходуют квоту. Отмены в клиенте нет, остановить можно только
    # в браузере, поэтому хотя бы предупреждаем.
    probe = run(["kaggle", "kernels", "status", f"{user}/{slug}"])
    if "running" in (probe.stdout or "").lower():
        print(f"  ВНИМАНИЕ: предыдущий запуск {user}/{slug} ещё идёт и после нового "
              f"пуша продолжит считаться. Остановить можно только в браузере: "
              f"меню «...» у нужной версии -> Stop.")

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

    p = sub.add_parser("run", help="собрать ноутбук, запустить и дождаться")
    p.add_argument("stage", choices=["a", "b", "export"])
    p.add_argument("--slug", default=None)
    p.add_argument("--source", default="ru-fin",
                   help="ядро, чей вывод подключается: там лежат матрицы этапа A")
    p.add_argument("--args", default="")
    p.add_argument("--no-wait", action="store_true")

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
        folder = os.path.join(DIST, "queries")
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder)
        shutil.copy2(os.path.join(ROOT, "data", "queries", "queries.jsonl"), folder)
        push_dataset(user, folder, QUERIES_DATASET, "ru-fin-queries", "набор запросов")

    elif args.cmd == "run":
        # вывод этапа A подключается как источник: там лежат матрицы эмбеддингов
        source = [f"{user}/{args.source}"] if args.source else []
        stages = {
            "a": ("run_phase_a.py", "ru-fin-phase-a", "ru-fin phase A", [], True),
            "export": ("export_chunks.py", "ru-fin-export-chunks", "ru-fin export chunks",
                       source, True),
            "b": ("run_phase_b.py", "ru-fin-phase-b", "ru-fin phase B", source, True),
        }
        script, slug, title, kernels, gpu = stages[args.stage]
        slug = args.slug or slug
        datasets = [f"{user}/{DATASET}"]
        if args.stage == "b":
            datasets.append(f"{user}/{QUERIES_DATASET}")
        ref = push_kernel(user, slug, title, script, args.args, datasets, kernels, gpu)
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
