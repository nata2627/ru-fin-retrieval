#!/usr/bin/env python3
"""Этап B на видеокарте: выдачи всех поисковых конфигураций.

Запускается после того, как набор запросов собран целиком: полторы сотни
синтетических с этапа A плюс пятьдесят написанных руками, с проверенной
разметкой. Нужен файл queries.jsonl — он загружается отдельным маленьким
датасетом.

Конфигурации: BM25, плотный поиск каждой моделью, гибрид через RRF,
гибрид с кросс-энкодером на первых пятидесяти. Плюс те же BM25 и плотный
поиск на каждой нарезке для абляций.

Обратно едут только списки идентификаторов: сотни килобайт. Метрики,
доверительные интервалы и разбор ошибок считаются локально.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gpu_common as common                                   # noqa: E402
import gpu_search as S                                        # noqa: E402
from rufin.chunk_configs import GRID                          # noqa: E402
from rufin.retrieval.hybrid import rrf                        # noqa: E402
from rufin.retrieval.model_specs import ABLATION, HEADLINE    # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default=None,
                    help="путь к корпусу; по умолчанию ищется сам")
    ap.add_argument("--queries", default="/kaggle/input/ru-fin-queries/queries.jsonl")
    ap.add_argument("--embeddings", default="/kaggle/input/ru-fin-embeddings/embeddings")
    ap.add_argument("--out", default="/kaggle/working/runs")
    ap.add_argument("--headline-models", nargs="*", default=list(HEADLINE))
    ap.add_argument("--hybrid-dense", default="bge-m3")
    ap.add_argument("--ablation-model", default=ABLATION)
    ap.add_argument("--ablation-configs", nargs="*",
                    default=[c.name for c in GRID if c.name != "base"])
    ap.add_argument("--limit-acts", type=int, default=0)
    args = ap.parse_args()

    device = common.pick_device()
    print(f"устройство: {device}", flush=True)
    ruler = common.make_ruler()
    acts = common.load_acts(args.acts, limit=args.limit_acts)
    queries = [json.loads(l) for l in open(args.queries, encoding="utf-8") if l.strip()]
    print(f"актов {len(acts)}, запросов {len(queries)}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    report: dict = {"queries": len(queries), "device": device, "runs": [], "index": {}}

    def emb_dir(config: str, model: str) -> str | None:
        d = os.path.join(args.embeddings, config, model)
        return d if os.path.exists(os.path.join(d, "vectors.npy")) else None

    configs = ["base"] + list(args.ablation_configs)
    for config in configs:
        print(f"\n=== нарезка {config} ===", flush=True)
        chunks = common.build_chunks(acts, config, ruler)
        texts = {c["chunk_id"]: c["text"] for c in chunks}

        bm25, build_seconds = S.bm25_runs(chunks, queries)
        S.save_run(args.out, config, "bm25", bm25)
        report["runs"].append({"config": config, "retrieval": "bm25"})
        report["index"][f"{config}/bm25"] = {"chunks": len(chunks),
                                             "build_seconds": round(build_seconds, 1)}

        models = args.headline_models if config == "base" else [args.ablation_model]
        if config == "base" and args.ablation_model not in models:
            models = list(models) + [args.ablation_model]

        dense: dict[str, dict] = {}
        for model in models:
            d = emb_dir(config, model)
            if d is None:
                print(f"   {model}: матрицы нет, пропуск", flush=True)
                continue
            try:
                runs, meta = S.dense_runs(d, model, queries, device=device)
            except Exception as e:  # noqa: BLE001
                print(f"   {model}: ОШИБКА {e}", flush=True)
                continue
            dense[model] = runs
            S.save_run(args.out, config, f"dense-{model}", runs)
            report["runs"].append({"config": config, "retrieval": f"dense:{model}"})
            report["index"][f"{config}/{model}"] = meta
            print(f"   {model}: выдача готова (матрица {meta['size_mb']} МБ, "
                  f"dim={meta['dim']})", flush=True)

        # гибрид и реранкер считаются только для базовой нарезки:
        # абляции по нарезке сравниваются на BM25 и одном плотном поиске
        if config != "base" or args.hybrid_dense not in dense:
            continue

        hybrid = {q["query_id"]: rrf([[(c, 0.0) for c in bm25[q["query_id"]]],
                                      dense[args.hybrid_dense][q["query_id"]]], top=S.TOP)
                  for q in queries}
        S.save_run(args.out, config, "hybrid", hybrid)
        report["runs"].append({"config": config, "retrieval": "hybrid"})
        print("   гибрид: выдача готова", flush=True)

        reranked = S.rerank_runs(queries, hybrid, texts, device=device)
        S.save_run(args.out, config, "hybrid-rerank", reranked)
        report["runs"].append({"config": config, "retrieval": "hybrid+rerank"})
        report["index"]["reranker"] = {"model": S.RERANKER, "candidates": S.TOP}
        print("   гибрид с реранкером: выдача готова", flush=True)

    with open(os.path.join(args.out, "report_phase_b.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)

    total = sum(os.path.getsize(os.path.join(args.out, f)) for f in os.listdir(args.out))
    print(f"\n==== итог этапа B ====", flush=True)
    print(f"выдач сохранено: {len(report['runs'])}, суммарно {total / 1048576:.1f} МБ", flush=True)
    print(f"скачать всю папку runs и положить в проект как data/runs", flush=True)


if __name__ == "__main__":
    main()
