import pytest                                            # фреймворк тестов
import pandas as pd                                      # DataFrame
import pandera as pa                                     # pa.errors.SchemaError
from scipy.stats import ks_2samp                         # KS-тест
from data_validation import (                            # импортируем наши функции
    validate_data,
    validate_with_great_expectations,
    load_titanic,
)


# ---------------------------------------------------------------
# 1. ТЕСТЫ ВАЛИДАЦИИ СХЕМЫ (PANDERA)
# ---------------------------------------------------------------
def test_valid_data_pandera():
    """Корректный Titanic должен проходить схему."""
    df = load_titanic("data.csv")                        # читаем CSV
    validated = validate_data(df)                        # валидируем (не должно упасть)
    assert not validated.empty                           # результат не пустой
    assert len(validated) == 891                         # ровно 891 строка, как в CSV


def test_invalid_data_pandera_negative_fare():
    """Отрицательный Fare должен вызывать SchemaError."""
    df = load_titanic("data.csv")                        # берём реальные данные
    df.loc[0, "Fare"] = -10.0                            # портим первую строку
    with pytest.raises(pa.errors.SchemaError):           # ожидаем ошибку схемы
        validate_data(df)                                # валидация должна упасть


def test_invalid_data_pandera_bad_sex():
    """Недопустимое значение Sex должно вызывать SchemaError."""
    df = load_titanic("data.csv")
    df.loc[1, "Sex"] = "unknown"                         # ставим недопустимый пол
    with pytest.raises(pa.errors.SchemaError):           # Check.isin отловит
        validate_data(df)


# ---------------------------------------------------------------
# 2. ТЕСТЫ GREAT EXPECTATIONS
# ---------------------------------------------------------------
def test_great_expectations_checks():
    """GX-проверки должны проходить на реальных данных."""
    df = load_titanic("data.csv")                        # читаем Titanic
    assert validate_with_great_expectations(df) is True  # ожидаем True


def test_great_expectations_detects_bad_survived():
    """GX должен падать, если Survived содержит значение вне {0,1}."""
    df = load_titanic("data.csv")
    df.loc[0, "Survived"] = 2                            # ставим недопустимое значение
    assert validate_with_great_expectations(df) is False # GX вернёт False


# ---------------------------------------------------------------
# 3. ТЕСТЫ НА ОБНАРУЖЕНИЕ ДРЕЙФА (KS-ТЕСТ)
# ---------------------------------------------------------------
NUMERIC_COLS = ["Age", "Fare", "SibSp", "Parch"]         # числовые колонки для KS-теста


def test_data_drift_detection_no_drift():
    """Две половины одного датасета не должны отличаться (дрейфа нет)."""
    df = load_titanic("data.csv")                        # читаем Titanic

    reference = df.sample(frac=0.5, random_state=42)     # случайные 50% — эталон
    current = df.drop(reference.index)                   # остальные 50% — «текущие»

    for col in NUMERIC_COLS:                             # по каждой числовой колонке
        ref = reference[col].dropna()                    # убираем NaN (KS не любит NaN)
        cur = current[col].dropna()
        p_value = ks_2samp(ref, cur).pvalue              # p-value KS-теста
        assert p_value > 0.01                            # p > 0.01 → дрейфа нет


def test_simulated_drift_failure_fare():
    """Искусственный сдвиг Fare на +100 должен обнаруживаться KS-тестом."""
    df = load_titanic("data.csv")

    reference = df.sample(frac=0.5, random_state=42)     # эталон
    current = df.drop(reference.index).copy()            # копия, чтобы не портить df

    current["Fare"] = current["Fare"] * 5 + 100          # вносим сильный сдвиг в распределение

    ref_fare = reference["Fare"].dropna()                # убираем NaN
    cur_fare = current["Fare"].dropna()

    p_value = ks_2samp(ref_fare, cur_fare).pvalue        # p-value
    assert p_value < 0.05                                # p < 0.05 → дрейф найден
