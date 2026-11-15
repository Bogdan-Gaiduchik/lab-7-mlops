import pandas as pd                                    # pandas — DataFrame и read_csv
import pandera as pa                                   # pandera — валидация схемы
from pandera import Column, Check                      # Column — описание колонки, Check — правило
import great_expectations as gx                        # Great Expectations (GX 1.x)
from great_expectations.core import ExpectationSuite   # контейнер для expectations


# ---------------------------------------------------------------
# Декларативная схема Pandera под датасет Titanic
# ---------------------------------------------------------------
titanic_schema = pa.DataFrameSchema(
    columns={                                          # словарь колонок
        "PassengerId": Column(int, Check.ge(1)),       # id пассажира ≥ 1
        "Survived":    Column(int, Check.isin([0, 1])),# только 0/1
        "Pclass":      Column(int, Check.isin([1, 2, 3])),  # только 1/2/3
        "Name":        Column(str, Check.str_length(min_value=1)),  # непустая строка
        "Sex":         Column(str, Check.isin(["male", "female"])), # два пола
        # Age — есть NaN (у 177 пассажиров), поэтому nullable=True
        "Age":         Column(float, Check.ge(0.0), nullable=True),
        "SibSp":       Column(int, Check.ge(0)),       # число братьев/сестёр/супругов ≥ 0
        "Parch":       Column(int, Check.ge(0)),       # число родителей/детей ≥ 0
        "Ticket":      Column(str, Check.str_length(min_value=1)),  # непустой тикет
        "Fare":        Column(float, Check.ge(0.0)),   # стоимость билета ≥ 0
        # Cabin — очень много пропусков, поэтому nullable=True
        "Cabin":       Column(str, nullable=True),
        # Embarked — 2 пропуска, допустимые значения S/C/Q
        "Embarked":    Column(str, Check.isin(["S", "C", "Q"]), nullable=True),
    },
    strict=True,                                       # запрещаем лишние колонки
    coerce=True,                                       # приводить типы автоматически
)


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Валидирует DataFrame по схеме Pandera; при нарушении бросает SchemaError."""
    return titanic_schema.validate(df)                 # возвращает df или бросает pa.errors.SchemaError


def load_titanic(path: str = "data.csv") -> pd.DataFrame:
    """Загружает Titanic из CSV (для использования в тестах)."""
    return pd.read_csv(path)                           # читаем CSV в DataFrame


def validate_with_great_expectations(df: pd.DataFrame) -> bool:
    """GX 1.x: ephemeral-контекст + pandas-датасорс + батч + suite."""
    context = gx.get_context(mode="ephemeral")         # временный контекст (без записи на диск)

    data_source = context.data_sources.add_pandas("titanic_source")   # регистрируем источник
    data_asset = data_source.add_dataframe_asset(name="titanic_asset")# регистрируем ассет
    batch_definition = data_asset.add_batch_definition_whole_dataframe("titanic_batch")  # батч = весь df
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})  # получаем батч

    suite = context.suites.add(ExpectationSuite(name="titanic_suite")) # пустой набор проверок

    # 1) все обязательные колонки существуют
    for col in ["PassengerId", "Survived", "Pclass", "Sex", "Age", "Fare"]:
        suite.add_expectation(
            gx.expectations.ExpectColumnToExist(column=col)            # ожидание: колонка есть
        )

    # 2) Survived принимает только 0/1
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="Survived", value_set=[0, 1]                        # множество допустимых значений
        )
    )

    # 3) Pclass принимает только 1/2/3
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="Pclass", value_set=[1, 2, 3]
        )
    )

    # 4) Sex принимает только male/female
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="Sex", value_set=["male", "female"]
        )
    )

    # 5) Fare ≥ 0
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="Fare", min_value=0.0                               # только нижняя граница
        )
    )

    # 6) Age либо NaN, либо в физически разумных пределах [0, 120]
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="Age", min_value=0.0, max_value=120.0, mostly=0.95   # 95% значений — в диапазоне
        )
    )

    # 7) PassengerId уникален
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeUnique(column="PassengerId")
    )

    result = batch.validate(suite)                     # запускаем все проверки
    return bool(result.success)                        # numpy.bool_ → python bool
