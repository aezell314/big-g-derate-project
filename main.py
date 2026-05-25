from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import CATEGORICAL_FEATURES, DATE_SPLIT, NUMERIC_FEATURES, SERVICE_STATIONS, TARGETS
from src.data_loading import (
    drop_unneeded_fault_columns,
    load_diagnostics,
    load_faults,
    merge_faults_and_diagnostics,
    pivot_diagnostics,
)
from src.evaluation import evaluate_predictions
from src.explainability import permutation_importance_frame, shap_explanation
from src.feature_engineering import (
    add_service_station_distances,
    add_time_bucket_features,
    add_time_features,
    clean_decimal_commas,
    create_derate_flag,
    create_time_window_targets,
    deduplicate_faults,
    drop_model_unneeded_columns,
    drop_rows_within_12h_of_derate,
    filter_near_service_stations,
    merge_targets_into_faults,
)
from src.modeling import run_cross_validation, train_final_model
from src.preprocessing import (
    build_preprocessor,
    cast_feature_types,
    forward_fill_running_counts,
    split_train_test_by_date,
)


def build_dataset(data_dir: str = "data") -> pd.DataFrame:
    data_dir = Path(data_dir)

    faults = load_faults(str(data_dir / "J1939Faults.csv"))
    diagnostics = load_diagnostics(str(data_dir / "VehicleDiagnosticOnboardData.csv"))

    faults = drop_unneeded_fault_columns(faults)
    faults = add_service_station_distances(faults, SERVICE_STATIONS)
    faults = filter_near_service_stations(faults, min_miles=1.0)

    diagnostics_pivoted = pivot_diagnostics(diagnostics)
    faults_full = merge_faults_and_diagnostics(faults, diagnostics_pivoted)

    faults_full = create_derate_flag(faults_full)
    faults_full["EventTimeStamp"] = pd.to_datetime(faults_full["EventTimeStamp"])

    faults_dedup = deduplicate_faults(faults_full)
    target_df = create_time_window_targets(faults_dedup)
    faults_full = merge_targets_into_faults(faults_full, target_df)

    faults_full = drop_model_unneeded_columns(faults_full)
    faults_full = clean_decimal_commas(faults_full)
    faults_full = add_time_features(faults_full)
    faults_full = add_time_bucket_features(faults_full)
    faults_full = drop_rows_within_12h_of_derate(faults_full)

    faults_full = cast_feature_types(faults_full, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    faults_full = forward_fill_running_counts(faults_full)
    
    return faults_full


def main():
    faults_full = build_dataset()

    train_data, test_data = split_train_test_by_date(faults_full, DATE_SPLIT)
    preprocessor = build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)

    #cv_results = run_cross_validation(train_data, preprocessor, TARGETS)
    #print("\nCross-validation results:")
    #print(cv_results.head())

    #summary_df = (
    #    cv_results.groupby(["time_window", "upweight", "oversample"])
   #     .mean(numeric_only=True)
   #     .reset_index()
   # )
   # print("\nSummary:")
   # print(summary_df.sort_values(["time_window", "f_beta"], ascending=[True, False]))

    target = "derate_6_hr"
    model, X_test, y_test = train_final_model(train_data, test_data, target, preprocessor)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    evaluation_df = test_data[["EquipmentID", "EventTimeStamp"]].copy()
    evaluation_df["pred_derate"] = y_pred.astype(int)
    evaluation_df["actual_derate"] = y_test.astype(int)

    metrics = evaluate_predictions(y_test, y_pred, y_pred_proba, evaluation_df)
    print("\nFinal test metrics:")
    print(metrics)

    importance_df = permutation_importance_frame(model, preprocessor, X_test, y_test)
    print("\nTop permutation importance features:")
    print(importance_df.head(15))

    shap_explanation(model, preprocessor, X_test)
    print("\nSHAP explanation figure ready (saved in assets folder)")

if __name__ == "__main__":
    main()