# Воспроизведение проекта по шагам.
#
# Тяжёлые вычисления вынесены на видеокарту Kaggle: на маке с восемью
# гигабайтами памяти инференс языковой модели и подсчёт эмбеддингов
# по шестидесяти тысячам фрагментов приводят к уходу системы в подкачку.
# Локально остаётся то, что памяти не требует: сбор корпуса, нарезка,
# метрики, разбор ошибок — и замеры задержки, которые по условию задачи
# снимаются там, где система работает.

PY ?= python3
SAMPLE ?= 12
ISSUES ?= 25
PAUSE ?= 1.5
CHUNKS ?= base
DENSE ?= bge-m3

.PHONY: help probe check-split check-bm25 check-alignment corpus chunks use-chunks kaggle bench metrics latency errors clean-raw

help:
	@echo "Локально (памяти не требует):"
	@echo "  probe        разведка источников: доступность, объём, качество текстового слоя"
	@echo "  check-split  проверка нарезки выпусков «Вестника» на отдельные акты"
	@echo "  check-bm25   сверка своей реализации BM25 с rank_bm25"
	@echo "  check-alignment  сверка: тот ли текст под эталонным фрагментом"
	@echo "  corpus       сбор корпуса: выпуски «Вестника» -> акты"
	@echo "  chunks       нарезка актов на фрагменты (CONFIGS=base ...)"
	@echo "  use-chunks   поставить нарезку, выгруженную с видеокарты (FILE=...)"
	@echo "  bench        сборка набора запросов и выгрузка в формате MTEB"
	@echo "  metrics      метрики по выдачам, посчитанным на Kaggle (CHUNKS=$(CHUNKS))"
	@echo "  errors       разбор провальных запросов"
	@echo "  latency      замеры задержки на этой машине (DENSE=$(DENSE))"
	@echo ""
	@echo "На видеокарте Kaggle (см. kaggle/README.md):"
	@echo "  kaggle       собрать пакет для загрузки (dist/kaggle, ~20 МБ)"
	@echo "               этап A: синтетические запросы и эмбеддинги"
	@echo "               этап B: выдачи всех поисковых конфигураций"

probe:
	$(PY) scripts/probe_sources.py --sample $(SAMPLE) --pause $(PAUSE) $(if $(REFRESH),--refresh,)

check-split:
	$(PY) scripts/validate_split.py --issues $(ISSUES) --pause $(PAUSE)

check-bm25:
	$(PY) scripts/check_bm25.py

check-alignment:
	$(PY) scripts/check_alignment.py --chunks $(CHUNKS)

corpus:
	$(PY) scripts/build_corpus.py --pause $(PAUSE)

# Нарезки занимают около двух гигабайт на диске. Для работы они локально
# не нужны: базовую отдаёт этап B на видеокарте (chunks_base.jsonl.gz),
# и брать надо именно её — граница фрагмента считается в токенах, а версия
# токенизатора у себя и на Kaggle может отличаться, и тогда под прежним
# именем окажется другой текст. Цель оставлена для разработки.
chunks:
	$(PY) scripts/build_chunks.py $(if $(CONFIGS),--only $(CONFIGS),--only base)

# Нарезка, по которой считались эмбеддинги, ставится как рабочая: повторять
# её локально нельзя, версия токенизатора другая и границы сдвинутся.
use-chunks:
	$(PY) scripts/install_chunks.py $(FILE) --config $(CHUNKS)

kaggle:
	$(PY) scripts/make_kaggle_package.py

# Сборке разметки предшествует сверка нарезок: если под эталонным фрагментом
# лежит не тот текст, по которому писался вопрос, метрики будут бессмысленны,
# а заметить это по ним самим нельзя.
bench: check-alignment
	$(PY) scripts/build_benchmark.py

metrics:
	$(PY) scripts/metrics_report.py --config $(CHUNKS)

errors:
	$(PY) scripts/error_analysis.py --run data/runs/$(CHUNKS)__hybrid-rerank.jsonl \
		--chunks $(CHUNKS) --deep-run data/runs/$(CHUNKS)__hybrid.jsonl

latency:
	$(PY) scripts/measure_latency.py --chunks $(CHUNKS) --dense $(DENSE)

clean-raw:
	rm -rf data/raw/*
