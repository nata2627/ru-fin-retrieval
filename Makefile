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
FROM ?= base
DENSE ?= bge-m3

.PHONY: help test lint probe check-split check-bm25 check-alignment remap length-effect chunking-effect corpus chunks use-chunks explan explan-apply gold gold-apply queries pool kaggle bench metrics latency errors clean-raw

help:
	@echo "Проверки (ни данных, ни видеокарты не требуют):"
	@echo "  test         тесты: pytest"
	@echo "  lint         проверка стиля: ruff"
	@echo ""
	@echo "Локально (памяти не требует):"
	@echo "  probe        разведка источников: доступность, объём, качество текстового слоя"
	@echo "  check-split  проверка нарезки выпусков «Вестника» на отдельные акты"
	@echo "  check-bm25   сверка своей реализации BM25 с rank_bm25"
	@echo "  check-alignment  сверка: тот ли текст под эталонным фрагментом"
	@echo "  remap        перенести эталоны на другую нарезку (CHUNKS=size-256)"
	@echo "  length-effect    растёт ли отставание с длиной эталона (DENSE=...)"
	@echo "  chunking-effect  меняется ли разрыв при смене нарезки (CHUNKS=...)"
	@echo "  corpus       сбор корпуса: выпуски «Вестника» -> акты"
	@echo "  chunks       нарезка актов на фрагменты (CONFIGS=base ...)"
	@echo "  use-chunks   поставить нарезку, выгруженную с видеокарты (FILE=...)"
	@echo "  explan       собрать живые вопросы из «Разъяснений» Банка России"
	@echo "  explan-apply перенести проверенный выбор в разметку"
	@echo "  gold         подобрать эталоны к вопросам, написанным руками"
	@echo "  gold-apply   перенести выбор по ним в разметку"
	@echo "  queries      собрать тексты запросов для прогона (эталоны не нужны)"
	@echo "  pool         лист разметки по объединённым выдачам всех конфигураций"
	@echo "  bench        сборка набора запросов и выгрузка в формате MTEB"
	@echo "  metrics      метрики по выдачам, посчитанным на Kaggle (CHUNKS=$(CHUNKS))"
	@echo "  errors       разбор провальных запросов"
	@echo "  latency      замеры задержки на этой машине (DENSE=$(DENSE))"
	@echo ""
	@echo "На видеокарте Kaggle (см. kaggle/README.md):"
	@echo "  kaggle       собрать пакет для загрузки (dist/kaggle, ~20 МБ)"
	@echo "               этап A: синтетические запросы и эмбеддинги"
	@echo "               этап B: выдачи всех поисковых конфигураций"

# Тесты на собранных данных помечены `data` и пропускаются, если данных нет.
test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests scripts kaggle

probe:
	$(PY) scripts/probe_sources.py --sample $(SAMPLE) --pause $(PAUSE) $(if $(REFRESH),--refresh,)

check-split:
	$(PY) scripts/validate_split.py --issues $(ISSUES) --pause $(PAUSE)

check-bm25:
	$(PY) scripts/check_bm25.py

check-alignment:
	$(PY) scripts/check_alignment.py --chunks $(CHUNKS)

# Эталон записан идентификатором фрагмента, а нумерация у каждой нарезки своя.
# Без переноса метрики по другой нарезке выйдут нулевыми — и ноль будет
# означать «эталона тут нет», а не «нарезка плохая».
remap:
	$(PY) scripts/remap_qrels.py --from $(FROM) --to $(CHUNKS)

# Обе проверки читают уже посчитанные выдачи и выносят вердикт с интервалом:
# без него столбик цифр легко прочитать глазами как закономерность.
length-effect:
	$(PY) scripts/length_effect.py --dense dense-$(DENSE)

chunking-effect:
	$(PY) scripts/chunking_effect.py --from $(FROM) --to $(CHUNKS) --dense dense-$(DENSE)

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
# Живые вопросы: собираются с сайта Банка России, к ним подбираются кандидаты
# в эталонный фрагмент, человек выбирает подходящий.
explan:
	$(PY) scripts/collect_explanations.py
	$(PY) scripts/prepare_explan_queries.py --target 50

explan-apply:
	$(PY) scripts/apply_explan_choices.py

# Тот же лист и та же процедура, но для вопросов, написанных руками.
gold-apply:
	$(PY) scripts/apply_explan_choices.py \
		--tsv data/queries/manual_candidates.tsv \
		--out data/queries/manual.jsonl --origin ручной

gold:
	$(PY) scripts/find_gold.py --file data/queries/manual_questions.txt

queries:
	$(PY) scripts/build_query_texts.py

# Кандидаты для разметки берутся из выдач всех конфигураций сразу: подбор
# эталона выдачей одного метода дал бы ему незаслуженное преимущество.
pool:
	$(PY) scripts/pool_candidates.py --config $(CHUNKS)

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
