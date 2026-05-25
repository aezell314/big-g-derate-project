from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder


def cast_feature_types(df: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]) -> pd.DataFrame:
    out = df.copy()

    for feature in numeric_features:
        if feature in out.columns:
            out[feature] = pd.to_numeric(out[feature], errors="coerce").astype("float32")

    for feature in categorical_features:
        if feature in out.columns:
            out[feature] = out[feature].astype("object").where(out[feature].notna(), np.nan)

    return out


def forward_fill_running_counts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for feature in ["DistanceLtd", "EngineTimeLtd", "FuelLtd"]:
        if feature in out.columns:
            out[feature] = out.groupby("EquipmentID")[feature].ffill().bfill()
    return out


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", IterativeImputer(random_state=0, verbose=2)),
            ("scaler", MaxAbsScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def split_train_test_by_date(df: pd.DataFrame, split_date: str):
    split_ts = pd.to_datetime(split_date)
    train_data = df[df["EventTimeStamp"] < split_ts].copy().reset_index(drop=True)
    test_data = df[df["EventTimeStamp"] >= split_ts].copy().reset_index(drop=True)
    return train_data, test_data