"""Удаление ссылок на акты: что должно уйти и что обязано остаться.

Главный риск здесь — удалить лишнее. В живых вопросах есть «процентная
ставка № 1», «0,01% от величины капитала», «форма 0409115»: если правка
заденет их, разница в метрике будет частично про испорченный вопрос,
а не про отсутствие номера акта.
"""
from rufin.references import has_act_number, strip_numbers, strip_references, tidy


def test_nomer_akta_uhodit_slovo_ostaetsya():
    t = "не применять п. 3.12 Положения № 590-П"
    assert strip_numbers(t) == "не применять п. Положения"


def test_data_i_nomer_uhodyat_vmeste():
    t = "согласно пункту 1.16 Положения Банка России от 27.03.2020 № 714-П"
    assert strip_numbers(t) == "согласно пункту Положения Банка России"


def test_nomer_bez_reshetki():
    assert strip_numbers("норма Положения 590-П") == "норма Положения"
    assert strip_numbers("норма Положения N 590-П") == "норма Положения"


def test_perechislenie_punktov():
    t = "на цели, указанные в п.п. 3.13 и 3.14 Положения № 590-П"
    assert strip_numbers(t) == "на цели, указанные в п.п. Положения"


def test_diapazon_statey():
    t = "урегулирован ст. 123.17 — 123.20 ГК РФ"
    assert strip_numbers(t) == "урегулирован ст. ГК РФ"


def test_nomer_ne_akta_ostaetsya():
    """Ставка под номером — часть вопроса, а не адрес акта."""
    t = "об отмене ставок № 1 и № 2 и установлении ставки № 3 (10% годовых)"
    assert strip_numbers(t) == t


def test_procenty_i_summy_ostayutsya():
    t = "если величина таких ссуд не превышает 0,01% от величины капитала"
    assert strip_numbers(t) == t


def test_polozhenie_bez_nomera_ne_trogaem():
    t = "в соответствии с Положением о порядке формирования резервов"
    assert strip_numbers(t) == t
    assert strip_references(t) == t


def test_ssylka_uhodit_celikom():
    t = "не применять п. 3.12 Положения № 590-П, который предусматривает"
    assert strip_references(t) == "не применять, который предусматривает"


def test_priznak_nomera_akta():
    assert has_act_number("требования пп. 3.14.1 Положения № 590-П")
    assert has_act_number("нормы Положения 590-П")
    assert not has_act_number("об отмене ставок № 1 и № 2")
    assert not has_act_number("вправе ли КО сохранить оценку финансового положения")


def test_tidy_ne_ostavlyaet_sledov():
    assert tidy("слово  ,  второе") == "слово, второе"
    assert tidy("текст (  ) дальше") == "текст дальше"
    assert tidy("перечень ( ) и") == "перечень и"
