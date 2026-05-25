from __future__ import annotations

import pandas as pd


def load_faults(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_diagnostics(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def drop_unneeded_fault_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [
        "ecuSoftwareVersion",
        "ecuSerialNumber",
        "ecuModel",
        "ecuMake",
        "ecuSource",
        "MCTNumber",
        "ESS_Id",
        "actionDescription",
        "faultValue"
    ]
    existing = [c for c in cols_to_drop if c in df.columns]
    return df.drop(columns=existing)


def pivot_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    return diagnostics.pivot(index="FaultId", columns="Name", values="Value").reset_index()


def merge_faults_and_diagnostics(
    faults: pd.DataFrame,
    diagnostics_pivoted: pd.DataFrame,
) -> pd.DataFrame:
    return pd.merge(
        faults,
        diagnostics_pivoted,
        left_on="RecordID",
        right_on="FaultId",
        how="inner",
    )