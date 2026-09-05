import pandas as pd

df = pd.read_csv("data.csv")

EXPECTED_COLUMNS = [
    "sno", "age", "gender", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca",
    "thal", "target"
]

ALLOWED_NULL_COLUMNS = {"trestbps", "chol", "thalach"}

def test_expected_schema():
    assert list(df.columns) == EXPECTED_COLUMNS

def test_no_unexpected_missing_values():
    null_counts = df.isnull().sum()
    for col, count in null_counts.items():
        if count > 0:
            assert col in ALLOWED_NULL_COLUMNS, f"Unexpected nulls in {col}"

def test_categorical_values():
    assert set(df["gender"].dropna().unique()) <= {"male", "female"}
    assert set(df["target"].dropna().unique()) <= {"yes", "no"}

def test_value_ranges():
    assert df["age"].between(1, 120).all()
    assert df["trestbps"].dropna().between(60, 250).all()
    assert df["chol"].dropna().between(80, 700).all()
    assert df["thalach"].dropna().between(50, 250).all()

def test_dataset_shape():
    assert df.shape[0] > 0
    assert df.shape[1] == len(EXPECTED_COLUMNS)
