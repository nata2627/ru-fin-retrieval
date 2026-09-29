# Таск MTEB

Свой бенчмарк, который прогоняется только своим кодом, — это утверждение,
а не измерение. Здесь тонкая обёртка, позволяющая прогнать набор чужим
кодом и сравнить результат с чем угодно из MTEB.

```bash
pip install mteb
# до публикации на HuggingFace — прямо из папки проекта
export RU_FIN_RETRIEVAL_PATH=$(pwd)/data/benchmark
python -c "
import mteb, sys; sys.path.insert(0, 'mteb_task')
from ru_fin_retrieval import RuFinRetrieval
from sentence_transformers import SentenceTransformer
mteb.MTEB(tasks=[RuFinRetrieval()]).run(
    SentenceTransformer('intfloat/multilingual-e5-small'),
    output_folder='results/mteb')
"
```

Таск считает одно разделение — `test`. В него входит всё, что размечено:
живые вопросы, ручные запросы, невиданные акты и синтетика. Отдельным файлом
лежит `qrels/dev.tsv` — dev по отложенным целиком актам, он нужен обучению,
а не сравнению моделей, и в таск не включён.

Одной цифрой по `test` пользоваться нельзя: подвыборки — это разные речевые
режимы (медиана длины 370, 62 и 136 знаков), и среднее по ним говорит
о смеси, которой не существует. Разбивка считается своим кодом
(`make metrics` печатает её по `data/queries/podvyborki.json`), и там же
видно заголовочную цифру — живые, ручные и невиданные акты без синтетики.

Осторожно с префиксами: у e5 и bge-m3 они разные, и ошибка в префиксе
стоит несколько пунктов, не давая при этом ни ошибки, ни предупреждения.
Схемы для шести проверенных моделей — `src/rufin/retrieval/model_specs.py`.
