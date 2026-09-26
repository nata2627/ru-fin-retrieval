#!/usr/bin/env python3
"""Этап A на видеокарте: синтетические запросы и эмбеддинги корпуса.

Порядок важен: сначала запросы, потом эмбеддинги. Запросы генерирует
языковая модель на 14 миллиардов параметров, и держать её в памяти
одновременно с эмбеддерами незачем — она выгружается до начала индексации.

Что получается на выходе:
  queries/synthetic.jsonl                       — маленький, скачать обязательно
  embeddings/<нарезка>/<модель>/vectors.npy     — большие, остаются на Kaggle
  report_phase_a.json                           — что посчиталось и за сколько

Всё посчитанное не пересчитывается: после обрыва сессии достаточно запустить
ячейку заново.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gpu_common as common              # noqa: E402
from gpu_embed import embed_config       # noqa: E402
from rufin.chunk_configs import GRID     # noqa: E402
from rufin.retrieval.model_specs import ABLATION, HEADLINE  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="/kaggle/input/ru-fin-retrieval/acts.jsonl.gz")
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--target-queries", type=int, default=150)
    ap.add_argument("--generator", default=None, help="модель для генерации вопросов")
    ap.add_argument("--headline-models", nargs="*", default=list(HEADLINE))
    ap.add_argument("--ablation-model", default=ABLATION)
    ap.add_argument("--ablation-configs", nargs="*",
                    default=[c.name for c in GRID if c.name != "base"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit-acts", type=int, default=0, help="для пробного прогона")
    ap.add_argument("--skip-queries", action="store_true")
    ap.add_argument("--no-four-bit", action="store_true",
                    help="без квантования: нужно для проверки на малой модели")
    ap.add_argument("--skip-embeddings", action="store_true")
    args = ap.parse_args()

    device = common.pick_device()
    print(f"устройство: {device}", flush=True)
    if device != "cuda":
        print("ВНИМАНИЕ: видеокарта не подключена. В настройках ноутбука "
              "Accelerator должен быть GPU T4 x2.", flush=True)

    ruler = common.make_ruler()
    acts = common.load_acts(args.acts, limit=args.limit_acts)
    print(f"актов: {len(acts)}, символов {sum(len(a['text']) for a in acts) / 1e6:.1f} млн",
          flush=True)

    report: dict = {"acts": len(acts), "device": device}
    base_chunks = common.build_chunks(acts, "base", ruler)
    report["base_chunks"] = len(base_chunks)

    # ---- синтетические запросы ----
    queries_path = os.path.join(args.out, "queries", "synthetic.jsonl")
    if args.skip_queries:
        print("генерация запросов пропущена", flush=True)
    elif os.path.exists(queries_path):
        print(f"запросы уже есть ({queries_path}), генерация пропущена", flush=True)
    else:
        from gpu_queries import DEFAULT_MODEL, make_queries
        report["queries"] = make_queries(
            base_chunks, queries_path, target=args.target_queries,
            model_path=args.generator or DEFAULT_MODEL,
            four_bit=not args.no_four_bit, device=device)

    # ---- эмбеддинги ----
    if not args.skip_embeddings:
        emb_root = os.path.join(args.out, "embeddings")
        plan = [("base", m) for m in args.headline_models]
        plan.append(("base", args.ablation_model))       # опорная точка для абляций
        plan += [(c, args.ablation_model) for c in args.ablation_configs]
        plan = list(dict.fromkeys(plan))                 # без повторов, порядок сохраняется

        report["embeddings"] = []
        chunks_by_config = {"base": base_chunks}
        for config, model in plan:
            if config not in chunks_by_config:
                chunks_by_config = {config: common.build_chunks(acts, config, ruler)}
            try:
                meta = embed_config(chunks_by_config[config], config, model, emb_root,
                                    args.batch_size, device)
            except Exception as e:  # noqa: BLE001
                print(f"[{config} / {model}] ОШИБКА: {e}", flush=True)
                report["embeddings"].append({"config": config, "model": model,
                                             "error": str(e)[:300]})
                continue
            report["embeddings"].append(meta)

    with open(os.path.join(args.out, "report_phase_a.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)

    print("\n==== итог этапа A ====", flush=True)
    if "queries" in report:
        q = report["queries"]
        print(f"запросов принято {q['accepted']}, отклонено {q['rejected']} "
              f"({100 * q['reject_share']:.0f}% попыток)", flush=True)
    ok = [e for e in report.get("embeddings", []) if "error" not in e]
    bad = [e for e in report.get("embeddings", []) if "error" in e]
    print(f"матриц посчитано: {len(ok)}, сбоев: {len(bad)}", flush=True)
    for e in ok:
        print(f"   {e['chunks_config']:<12} {e['model']:<12} {e['chunks']:>7} фрагм. "
              f"dim={e['dim']:<5} {e['per_second']:>7} фрагм./с {e['size_mb']:>8} МБ", flush=True)
    for e in bad:
        print(f"   СБОЙ {e['config']} / {e['model']}: {e['error'][:120]}", flush=True)
    print(f"\nсуммарный объём матриц: {sum(e['size_mb'] for e in ok) / 1024:.1f} ГБ", flush=True)
    print("скачать обязательно: queries/synthetic.jsonl (маленький файл)", flush=True)


if __name__ == "__main__":
    main()
