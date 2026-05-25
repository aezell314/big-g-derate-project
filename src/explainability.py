from __future__ import annotations

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, make_scorer
import matplotlib.pyplot as plt

def permutation_importance_frame(model, pipe, X_test, y_test, sample_size: int = 2000) -> pd.DataFrame:
    f1_scorer = make_scorer(f1_score, zero_division=0, average="weighted")
    X_sample = pd.DataFrame(X_test).sample(n=min(sample_size, len(X_test)), random_state=0)
    y_sample = pd.Series(y_test).sample(n=min(sample_size, len(y_test)), random_state=0)

    scores = permutation_importance(
        model,
        X_sample,
        y_sample,
        random_state=0,
        scoring=f1_scorer,
    )

    df = pd.DataFrame(
        {
            "variable": pipe.get_feature_names_out(),
            "importance": scores["importances_mean"],
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    df.to_csv("assets/perm_feature_importance.csv", index=False)

    return df


def shap_explanation(model, pipe, X_test):
    import shap

    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_test)
    explanation.feature_names = pipe.get_feature_names_out()

    plt.figure(figsize=(10, 6))
    shap.plots.bar(explanation[:, :, 1], show=False)
    
    plt.savefig('assets/shap_bar_plot.png', bbox_inches='tight')
    plt.close()

