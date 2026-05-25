from __future__ import annotations

import numpy as np
import pandas as pd


def haversine_vectorize(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    newlon = lon2 - lon1
    newlat = lat2 - lat1
    haver_formula = (
        np.sin(newlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(newlon / 2.0) ** 2
    )
    dist = 2 * np.arcsin(np.sqrt(haver_formula))
    return 3958 * dist


def add_service_station_distances(
    df: pd.DataFrame,
    stations: list[tuple[float, float]],
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
) -> pd.DataFrame:
    out = df.copy()
    for i, (lat, lon) in enumerate(stations, start=1):
        out[f"dist_from_stat_{i}"] = haversine_vectorize(lon, lat, out[lon_col], out[lat_col])
    return out


def filter_near_service_stations(df: pd.DataFrame, min_miles: float = 1.0) -> pd.DataFrame:
    required = ["dist_from_stat_1", "dist_from_stat_2", "dist_from_stat_3"]
    return df[
        (df[required[0]] > min_miles)
        & (df[required[1]] > min_miles)
        & (df[required[2]] > min_miles)
    ].copy()


def create_derate_flag(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    active_bool = out["active"].astype(str).str.lower().isin(["true", "1", "yes"])
    out["derate_flag"] = ((out["spn"] == 5246) & active_bool).astype(int)
    return out


def deduplicate_faults(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["EquipmentID", "EventTimeStamp"], as_index=False).max('derate_flag')


def _rolling_sum_by_equipment(
    df: pd.DataFrame,
    window: str,
    value_col: str,
    count_col: str,
) -> pd.Series:
    rolled = (
        df.groupby("EquipmentID")
        .rolling(window=window, on="EventTimeStamp")[value_col]
        .sum()
        .reset_index(level=0, drop=True)
    )
    return rolled.rename(count_col)


def _rolling_count_by_equipment(
    df: pd.DataFrame,
    window: str,
    value_col: str,
    count_col: str,
) -> pd.Series:
    rolled = (
        df.groupby("EquipmentID")
        .rolling(window=window, on="EventTimeStamp")[value_col]
        .count()
        .reset_index(level=0, drop=True)
    )
    return rolled.rename(count_col)


def create_time_window_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduces the notebook's rolling-window target creation.
    """
    faults_dedup = df.sort_values(by="EventTimeStamp", ascending=False).copy()

    target = faults_dedup[["EquipmentID", "EventTimeStamp"]].copy()
    target["total_derates_2_hr"] = _rolling_sum_by_equipment(
        faults_dedup, "2h", "derate_flag", "total_derates_2_hr"
    ).to_numpy()
    target["total_derates_6_hr"] = _rolling_sum_by_equipment(
        faults_dedup, "6h", "derate_flag", "total_derates_6_hr"
    ).to_numpy()
    target["total_derates_12_hr"] = _rolling_sum_by_equipment(
        faults_dedup, "12h", "derate_flag", "total_derates_12_hr"
    ).to_numpy()
    target["total_derates_24_hr"] = _rolling_sum_by_equipment(
        faults_dedup, "24h", "derate_flag", "total_derates_24_hr"
    ).to_numpy()
    target["faults_in_last_hr"] = _rolling_count_by_equipment(
        faults_dedup, "1h", "spn", "faults_in_last_hr"
    ).to_numpy()

    target["derate_6_hr"] = np.where(
        target["total_derates_2_hr"] > 0,
        2,
        np.where(target["total_derates_6_hr"] - target["total_derates_2_hr"] > 0, 1, 0),
    )
    target["derate_12_hr"] = np.where(
        target["total_derates_2_hr"] > 0,
        2,
        np.where(target["total_derates_12_hr"] - target["total_derates_2_hr"] > 0, 1, 0),
    )
    target["derate_24_hr"] = np.where(
        target["total_derates_2_hr"] > 0,
        2,
        np.where(target["total_derates_24_hr"] - target["total_derates_2_hr"] > 0, 1, 0),
    )

    target = target.drop(
        columns=[
            "total_derates_2_hr",
            "total_derates_6_hr",
            "total_derates_12_hr",
            "total_derates_24_hr",
        ]
    )
    return target


def merge_targets_into_faults(faults_full: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    out = pd.merge(target, faults_full, on=["EventTimeStamp", "EquipmentID"], how="inner")
    return out.sort_values(by="EventTimeStamp").copy()

def drop_model_unneeded_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [
        "Latitude",
        "Longitude",
        "RecordID",
        "LocationTimeStamp",
        "eventDescription",
        "ServiceDistance",
    ]
    existing = [c for c in cols_to_drop if c in df.columns]
    return df.drop(columns=existing)

def clean_decimal_commas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    comma_cols = [col for col in out.columns if out[col].astype(str).str.contains(",").any()]
    for col in comma_cols:
        out[col] = out[col].astype(str).str.replace(",", ".", regex=False)
    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["EventTimeStamp"] = pd.to_datetime(out["EventTimeStamp"])
    out["year"] = out["EventTimeStamp"].dt.year
    out["month"] = out["EventTimeStamp"].dt.month
    out["day"] = out["EventTimeStamp"].dt.weekday
    out["hour"] = out["EventTimeStamp"].dt.hour
    return out


def add_time_bucket_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["weekday"] = out["day"].apply(lambda x: 0 if x >= 5 else 1)
    out["summer"] = out["month"].apply(lambda x: 1 if x in (6, 7) else 0)
    out["daytime"] = out["hour"].apply(lambda x: 1 if x >= 7 or x <= 17 else 0)
    out["nighttime"] = out["hour"].apply(lambda x: 1 if x < 7 or x > 17 else 0)
    return out


def drop_rows_within_12h_of_derate(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    derate_events = (
        out[out["derate_flag"] == 1][["EquipmentID", "EventTimeStamp"]]
        .rename(columns={"EventTimeStamp": "last_derate_time"})
        .copy()
    )

    out = out.sort_values(["EventTimeStamp", "EquipmentID"]).reset_index(drop=True)
    derate_events = derate_events.sort_values(["last_derate_time", "EquipmentID"]).reset_index(drop=True)

    out = pd.merge_asof(
        out,
        derate_events,
        left_on="EventTimeStamp",
        right_on="last_derate_time",
        by="EquipmentID",
        direction="backward",
    )

    out = out[out["EventTimeStamp"] > out["last_derate_time"] + pd.Timedelta(hours=12)]
    out = out.drop(columns=["derate_flag", "last_derate_time"])
    return out.copy()