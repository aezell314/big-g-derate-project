from __future__ import annotations
from .config import NUMERIC_FEATURES,CATEGORICAL_FEATURES

from dataclasses import dataclass

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

from .evaluation import calculate_cost_savings


@dataclass
class FoldResult:
    fold: int
    time_window: str
    upweight: int
    oversample: int
    f_beta: float
    roc_auc: float
    precision: float
    recall: float
    false_positives: int
    true_positives: int
    cost_savings: float

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if not c.startswith("derate") and c not in ("EquipmentID", "EventTimeStamp")
    ]
    
def make_xgb_model() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=1000,
        objective="multi:softprob",
        num_class=3,
        n_jobs=-1,
        eval_metric="mlogloss",
    )


def fit_model(X_train, y_train, strategy: str = "upweight", random_state: int = 321):
    model = make_xgb_model()

    if strategy == "oversample":
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        model.fit(X_train, y_train)
        return model, X_train, y_train

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    weight_dict = {cls: weight for cls, weight in zip(classes, weights)}
    sample_weight = np.vectorize(weight_dict.get)(y_train)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model, X_train, y_train


def build_sequential_parts(train_data: pd.DataFrame, target: str, n_parts: int = 5):
    positive_class = train_data[train_data[target] == 1][[target]]
    if positive_class.empty:
        raise ValueError(f"No positive examples found for target={target!r}")

    groups, cutoffs = pd.qcut(positive_class.index, q=n_parts, labels=False, retbins=True)
    if len(cutoffs) < 3:
        raise ValueError(f"Not enough data to build {n_parts} parts for target={target!r}")

    return np.array_split(train_data, cutoffs[1:-1].astype(int))


def evaluate_fold(model, X_val, y_val, val_set: pd.DataFrame):
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)

    evaluation_df = val_set[["EquipmentID", "EventTimeStamp"]].copy()
    evaluation_df["pred_derate"] = y_pred.astype(int)
    evaluation_df["actual_derate"] = y_val.astype(int)

    cm = confusion_matrix(y_val, y_pred)
    tp = int(cm[1, 1]) if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
    fp = int(cm[0, 1]) if cm.shape[0] > 1 and cm.shape[1] > 1 else 0

    return {
        "f_beta": fbeta_score(y_val, y_pred, beta=8, average="weighted", zero_division=0),
        "roc_auc": roc_auc_score(y_val, y_pred_proba, multi_class="ovo", labels=[0, 1, 2]),
        "precision": precision_score(y_val, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_val, y_pred, average="weighted", zero_division=0),
        "false_positives": fp,
        "true_positives": tp,
        "cost_savings": calculate_cost_savings(evaluation_df),
        "evaluation_df": evaluation_df,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
    }


def run_cross_validation(
    train_data: pd.DataFrame,
    preprocessor,
    targets: list[str],
    n_parts: int = 5,
    strategies: tuple[str, ...] = ("upweight", "oversample"),
):
    results = []

    print(train_data.dtypes)
    print(train_data.isna().sum())

    for target in targets:
        print(f"TARGET: {target}")
        parts = build_sequential_parts(train_data, target=target, n_parts=n_parts)
        
        for strategy in strategies:
            print(f"BALANCING STRATEGY: {strategy}")

            for i in range(4):
                train_parts = [parts[j] for j in range(4) if j <= i]
                train_set = pd.concat(train_parts).copy()
                val_set = parts[i + 1].copy()

                feature_cols = get_feature_columns(train_set)
                X_train = train_set[feature_cols]
                y_train = train_set[target]
                X_val = val_set[feature_cols]
                y_val = val_set[target]

                X_train = preprocessor.fit_transform(X_train)
                X_val = preprocessor.transform(X_val)

                model, _, _ = fit_model(X_train, y_train, strategy=strategy)
                metrics = evaluate_fold(model, X_val, y_val, val_set)

                results.append(
                    FoldResult(
                        fold=i + 1,
                        time_window=target,
                        upweight=1 if strategy == "upweight" else 0,
                        oversample=1 if strategy == "oversample" else 0,
                        f_beta=metrics["f_beta"],
                        roc_auc=metrics["roc_auc"],
                        precision=metrics["precision"],
                        recall=metrics["recall"],
                        false_positives=metrics["false_positives"],
                        true_positives=metrics["true_positives"],
                        cost_savings=metrics["cost_savings"],
                    )
                )

    return pd.DataFrame([r.__dict__ for r in results])


def train_final_model(train_set: pd.DataFrame, test_set: pd.DataFrame, target: str, preprocessor):
    feature_cols = get_feature_columns(train_set)
    X_train = train_set[feature_cols]
    y_train = train_set[target]
    X_test = test_set[feature_cols]
    y_test = test_set[target]

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    model, _, _ = fit_model(X_train, y_train, strategy="upweight")
    return model, X_test, y_test