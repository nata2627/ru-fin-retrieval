"""Списочная дистилляция ранжирования: KL между распределениями по списку.

Учитель — кросс-энкодер `bge-reranker-v2-m3`, он читает вопрос и фрагмент
вместе и поэтому знает про пару то, чего би-энкодер знать не может.
Перенести от него можно не ответ, а **порядок**: для списка из нескольких
документов учитель задаёт, насколько каждый уместнее остальных.

Почему KL по списку, а не MSE по парам. MarginMSE учит воспроизводить
разность оценок двух документов и тянет ученика к абсолютной шкале учителя,
которая ученику недоступна: у кросс-энкодера оценка это логит
классификатора, у би-энкодера — косинус, ограниченный единицей. KL же
сравнивает распределения, то есть только относительную уместность внутри
списка, а это ровно то, что мерит ранжирующая метрика.

Почему своя функция потерь, а не готовая из библиотеки. Нужны две вещи,
которых у готовой нет: отдельная температура для ученика и для учителя
(этап E подбирает её по dev) и множитель косинуса. Косинус лежит
в [-1, 1], и softmax по нему почти ровный — ученик получал бы слабый
градиент независимо от того, насколько он ошибся. Поэтому сходства
умножаются на тот же множитель 20, что стоит в контрастивной функции
потерь, и уже потом делятся на температуру.

Версия библиотеки тоже довод: `DistillKLDivLoss` появился в
sentence-transformers не сразу, а какая версия встанет на Kaggle, заранее
не известно. Тридцать строк своего кода надёжнее проверки версии в рантайме.
"""
from __future__ import annotations


def make_listwise_kl(model, scale: float = 20.0, temperature: float = 1.0,
                     teacher_temperature: float = 1.0):
    """Собрать функцию потерь. torch импортируется внутри: модуль читается
    и на маке, где torch не стоит."""
    import torch
    import torch.nn.functional as F
    from torch import nn

    class ListwiseKLLoss(nn.Module):
        """KL(учитель ‖ ученик) по списку документов одного вопроса.

        Входные признаки: первый — вопрос, остальные — документы списка,
        по одному столбцу на документ. `labels` — оценки учителя тех же
        документов в том же порядке, матрица «вопросов × документов».
        """

        def __init__(self) -> None:
            super().__init__()
            self.model = model
            self.scale = scale
            self.temperature = temperature
            self.teacher_temperature = teacher_temperature

        def forward(self, sentence_features: list[dict], labels) -> "torch.Tensor":
            if len(sentence_features) < 3:
                raise ValueError(
                    "списочной дистилляции нужен вопрос и хотя бы два документа: "
                    f"пришло столбцов {len(sentence_features)}. На одном документе "
                    f"softmax по списку тождественно равен единице, и градиента нет")
            embeddings = [self.model(f)["sentence_embedding"] for f in sentence_features]
            anchor, docs = embeddings[0], embeddings[1:]
            # косинус: векторы нормируются здесь, а не в модели, потому что
            # матрёшка обрезает вектор до нужной длины уже после пулинга
            anchor = F.normalize(anchor, p=2, dim=1)
            sims = torch.stack([(anchor * F.normalize(d, p=2, dim=1)).sum(dim=1)
                                for d in docs], dim=1)
            ученик = F.log_softmax(sims * self.scale / self.temperature, dim=1)
            учитель = F.softmax(labels.float() / self.teacher_temperature, dim=1)
            return F.kl_div(ученик, учитель, reduction="batchmean")

        def get_config_dict(self) -> dict:
            """Уезжает в карточку модели вместе с весами."""
            return {"scale": self.scale, "temperature": self.temperature,
                    "teacher_temperature": self.teacher_temperature}

    return ListwiseKLLoss()
