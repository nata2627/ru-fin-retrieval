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
REPO ?= nata2627/ru-fin-retrieval

.PHONY: help test lint probe check-split check-bm25 check-alignment remap length-effect chunking-effect corpus chunks use-chunks explan queries pool kaggle bench metrics latency errors clean-raw split live manual-task manual-apply judge sample-for-human kappa check-bench publish check-train dev-metrics recipe matryoshka

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
	@echo "  explan       скачать живые вопросы из «Разъяснений» Банка России"
	@echo "  split        сплит корпуса по актам (обязателен до генерации)"
	@echo "  live         живые вопросы: привязка к акту и разметка по пунктам"
	@echo "  manual-task  лист актов, по которым писать ручные запросы"
	@echo "  manual-apply перенести заполненный лист в набор"
	@echo "  judge        перенести разметку судьи (judge.tsv с Kaggle) в эталоны"
	@echo "  sample-for-human  выборка 150 пар для ручной проверки судьи"
	@echo "  kappa        согласие судьи с человеком числом"
	@echo "  queries      собрать тексты запросов для прогона (эталоны не нужны)"
	@echo "  pool         лист разметки по объединённым выдачам всех конфигураций"
	@echo "  bench        сборка набора запросов и выгрузка в формате MTEB"
	@echo "  check-bench  сверка выгрузки самой с собой перед публикацией"
	@echo "  publish      публикация набора на HuggingFace (REPO=$(REPO))"
	@echo "  metrics      метрики по выдачам, посчитанным на Kaggle (CHUNKS=$(CHUNKS))"
	@echo "  errors       разбор провальных запросов"
	@echo "  latency      замеры задержки на этой машине (DENSE=$(DENSE))"
	@echo ""
	@echo "Дообучение (счёт на Kaggle, проверки и отчёт локально):"
	@echo "  check-train  проверка обучающих данных: утечка, эталоны, негативы"
	@echo "  dev-metrics  метрики ТОЛЬКО по dev — на время подбора рецепта"
	@echo "  recipe       таблица «этап рецепта -> dev» из журнала обучения"
	@echo "  matryoshka   индексы урезанных размерностей из полной матрицы"
	@echo ""
	@echo "На видеокарте Kaggle (см. kaggle/README.md):"
	@echo "  kaggle       собрать пакет для загрузки (dist/kaggle, ~20 МБ)"
	@echo "               этап A: синтетические запросы и эмбеддинги"
	@echo "               этап B: выдачи всех поисковых конфигураций"
	@echo "               этап C: обучающая выборка, dev, тест, судья"
	@echo "               этап D: подготовка, рецепт обучения, финал"

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

# Живые вопросы качаются с сайта Банка России один раз; дальше с собранным
# файлом работает `make live`.
explan:
	$(PY) scripts/collect_explanations.py

# Сплит режется по актам: фрагменты одного акта нарезаны с перекрытием,
# и сплит по фрагментам — это утечка. Пересобирать после каждой новой
# разметки и обязательно ДО генерации обучающей выборки.
split:
	$(PY) scripts/make_split.py $(if $(SEED),--seed $(SEED),)

# Живые вопросы из «Разъяснений»: к какому акту относится вопрос, какие
# из них размечаются по названным пунктам, какие идут в пул, какие
# отбрасываются, потому что названного пункта в нашей редакции акта нет.
live:
	$(PY) scripts/label_by_clause.py --chunks $(CHUNKS)

# Ручные запросы — единственное место, где разнообразие актов задаётся
# нарочно. Лист назначает акт каждому запросу; заполняется руками.
manual-task:
	$(PY) scripts/manual_task.py --chunks $(CHUNKS)

manual-apply:
	$(PY) scripts/apply_manual_task.py

# Разметка пула считается судьёй на Kaggle; сюда приезжает judge.tsv.
judge:
	$(PY) scripts/apply_judge.py

# У судьи есть погрешность, и она измеряется согласием с человеком.
sample-for-human:
	$(PY) scripts/sample_for_human.py --chunks $(CHUNKS)

kappa:
	$(PY) scripts/kappa.py

# Сборке разметки предшествует сверка нарезок: если под эталонным фрагментом
# лежит не тот текст, по которому писался вопрос, метрики будут бессмысленны,
# а заметить это по ним самим нельзя. Она встроена в `make bench`.
queries:
	$(PY) scripts/build_query_texts.py

# Кандидаты для разметки берутся из выдач всех конфигураций сразу: подбор
# эталона выдачей одного метода дал бы ему незаслуженное преимущество.
pool:
	$(PY) scripts/pool_candidates.py --config $(CHUNKS)

bench: check-alignment
	$(PY) scripts/build_benchmark.py

# Публикация: сначала проверки, потом заливка. Опубликованный набор
# с эталоном на несуществующий фрагмент хуже, чем ненапубликованный.
check-bench:
	$(PY) scripts/publish_hf.py --dry-run

publish: check-bench
	$(PY) scripts/publish_hf.py --repo $(REPO)

metrics:
	$(PY) scripts/metrics_report.py --config $(CHUNKS)

# Обучающие данные проверяются до того, как включена видеокарта: утечка
# по актам обесценивает dev, а значит и весь подбор рецепта, а съехавший
# эталон учит модель находить не то. Обе беды по метрикам не видны.
check-train:
	$(PY) scripts/check_train_data.py --chunks $(CHUNKS)

# Тест трогается один раз, в самом конце. Пока подбирается рецепт, цель
# `metrics` опасна: она считает по всему размеченному, то есть и по тесту,
# и запустить её по привычке ничего не стоит. Эта считает только по dev.
dev-metrics:
	$(PY) scripts/metrics_report.py --config $(CHUNKS) --only dev

# Вердикты пересчитываются из поквериных значений тем же кодом, которым
# их считала видеокарта: расхождение означает, что журнал приехал не тем,
# чем уехал.
recipe:
	$(PY) scripts/recipe_table.py

# Матрёшка не требует второго прогона модели: 256 и 128 получаются из 384
# срезом с повторной нормировкой. Нужны, чтобы померить задержку и размер
# индекса по каждой размерности.
matryoshka:
	$(PY) scripts/matryoshka_index.py --chunks $(CHUNKS) --from-model $(DENSE)

errors:
	$(PY) scripts/error_analysis.py --run data/runs/$(CHUNKS)__hybrid-rerank.jsonl \
		--chunks $(CHUNKS) --deep-run data/runs/$(CHUNKS)__hybrid.jsonl

latency:
	$(PY) scripts/measure_latency.py --chunks $(CHUNKS) --dense $(DENSE)

clean-raw:
	rm -rf data/raw/*
