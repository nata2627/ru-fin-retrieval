# Воспроизведение проекта по шагам. Каждая цель перезапускается
# без ручных действий; скачанное не выкачивается заново.

PY ?= python3
SAMPLE ?= 12
ISSUES ?= 25
PAUSE ?= 1.5
CHUNKS ?= base
MODEL ?= bge-m3

.PHONY: help probe check-split corpus chunks index queries eval ablations kaggle clean-raw

help:
	@echo "probe        разведка источников: доступность, объём, качество текстового слоя"
	@echo "check-split  проверка нарезки выпусков на отдельные акты"
	@echo "corpus       сбор корпуса: выпуски «Вестника» -> акты"
	@echo "chunks       нарезка актов на чанки во всех конфигурациях абляций"
	@echo "index        эмбеддинги корпуса      (MODEL=$(MODEL) CHUNKS=$(CHUNKS))"
	@echo "queries      генерация синтетических запросов"
	@echo "eval         прогон конфигураций и метрики (CHUNKS=$(CHUNKS))"
	@echo "ablations    прогон всех абляций"
	@echo "kaggle       пакет для индексации на чужой видеокарте"
	@echo ""
	@echo "переменные: SAMPLE=$(SAMPLE) ISSUES=$(ISSUES) PAUSE=$(PAUSE) CHUNKS=$(CHUNKS) MODEL=$(MODEL)"

probe:
	$(PY) scripts/probe_sources.py --sample $(SAMPLE) --pause $(PAUSE) $(if $(REFRESH),--refresh,)

check-split:
	$(PY) scripts/validate_split.py --issues $(ISSUES) --pause $(PAUSE)

corpus:
	$(PY) scripts/build_corpus.py --pause $(PAUSE)

chunks:
	$(PY) scripts/build_chunks.py

index:
	$(PY) scripts/build_index.py --chunks $(CHUNKS) --model $(MODEL)

queries:
	$(PY) scripts/make_queries.py

eval:
	$(PY) scripts/evaluate.py --chunks $(CHUNKS)

# Абляции: по одному изменённому параметру за раз, поэтому каждая нарезка
# индексируется и меряется отдельно, на тех же запросах и той же разметке.
ablations:
	@for c in size-256 size-1024 overlap-0 by-length no-heading; do \
		$(PY) scripts/build_index.py --chunks $$c --model $(MODEL) && \
		$(PY) scripts/evaluate.py --chunks $$c --configs bm25 dense:$(MODEL) --dense $(MODEL) --no-latency; \
	done
	$(PY) scripts/evaluate.py --chunks base --configs bm25 dense:$(MODEL) --dense $(MODEL) \
		--no-latency --normalize-queries

# Индексация на видеокарте: пакет кладётся в dist/kaggle и грузится
# на Kaggle отдельным датасетом, см. kaggle/README.md.
kaggle:
	$(PY) scripts/make_kaggle_package.py

clean-raw:
	rm -rf data/raw/*
