from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, fbeta_score, precision_score, recall_score, roc_auc_score


def calculate_cost_savings(evaluation_df: pd.DataFrame) -> float:
    """
    Returns the amount of money saved by implementing the model.

    Expects:
        EquipmentID, EventTimeStamp, actual_derate, pred_derate
    """
    df = evaluation_df.copy()
    df["date"] = pd.to_datetime(df["EventTimeStamp"]).dt.date

    model_cost_total = (
        df.groupby(["EquipmentID", "date"])
        .agg({"actual_derate": "max", "pred_derate": "max"})
        .sort_values("actual_derate", ascending=False)
    )

    total_cost_list = []
    for _, eval_row in model_cost_total.iterrows():
        if (eval_row["actual_derate"] == 0) and (eval_row["pred_derate"] == 1):
            total_cost_list.append(-500)
        elif (eval_row["actual_derate"] == 1) and (eval_row["pred_derate"] == 1):
            total_cost_list.append(4000)
        else:
            total_cost_list.append(0)

    return float(np.sum(total_cost_list))


def evaluate_predictions(y_true, y_pred, y_pred_proba, evaluation_df: pd.DataFrame) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    tp = int(cm[1, 1]) if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
    fp = int(cm[0, 1]) if cm.shape[0] > 1 and cm.shape[1] > 1 else 0

    return {
        "f_beta": fbeta_score(y_true, y_pred, beta=8, average="weighted", zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_pred_proba, multi_class="ovo", labels=[0, 1, 2]),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "false_positives": fp,
        "true_positives": tp,
        "cost_savings": calculate_cost_savings(evaluation_df),
    }