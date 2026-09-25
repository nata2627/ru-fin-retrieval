# Воспроизведение проекта по шагам. Каждая цель самодостаточна и
# перезапускается без ручных действий.

PY ?= python3
SAMPLE ?= 12
PAUSE ?= 1.5

.PHONY: help probe corpus index eval clean-raw

help:
	@echo "probe   — разведка источников: доступность, объём, качество текстового слоя"
	@echo "corpus  — сбор и нарезка корпуса              (этап 2)"
	@echo "index   — построение индексов BM25 и плотных  (этап 3)"
	@echo "eval    — прогон конфигураций и метрики       (этап 3)"
	@echo ""
	@echo "переменные: SAMPLE=$(SAMPLE) PAUSE=$(PAUSE)"

# Разведка. Перечни актов берутся из data/index, если они уже собраны;
# make probe REFRESH=1 перекачивает их заново.
probe:
	$(PY) scripts/probe_sources.py --sample $(SAMPLE) --pause $(PAUSE) $(if $(REFRESH),--refresh,)

corpus index eval:
	@echo "цель '$@' появится на следующем этапе; сейчас доступна только 'probe'" && exit 1

clean-raw:
	rm -rf data/raw/*
