import json
import pandas as pd
import numpy as np
from time import time



def prepare_dataset(df):
    df.rename(columns={
        "DateTimestamp": "DateTimestamp",
        "selected_plc_line": "Plc_Line",
        "work_order": "WorkOrder",
        "batch_no": "BatchNumber",
        "serial_no": "ProductSerial"
    }, inplace=True)

    # Handle ProductSerial == "0" and mark anomaly types
    df["ProductSerial"] = df["ProductSerial"].astype(str)
    df["Anomaly"] = 0
    df.loc[df["ProductSerial"] == "0", "Anomaly"] = 1  # Missing Scan
    df.loc[df["ProductSerial"] == "0", "ProductSerial"] = '0000000000'
    df.loc[df["ProductSerial"].str.len() != 10, "Anomaly"] = 2  # Incorrect Format

    # Convert timestamps and sort
    df["DateTimestamp"] = pd.to_datetime(df["DateTimestamp"])
    df = df.sort_values("DateTimestamp")

    # Time-based features
    df["hour"] = df["DateTimestamp"].dt.hour
    df["day_of_week"] = df["DateTimestamp"].dt.dayofweek
    df["day_of_month"] = df["DateTimestamp"].dt.day
    df["month"] = df["DateTimestamp"].dt.month
    df["weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Cyclic encoding
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Rolling anomaly count
    df["rolling_anomaly_10"] = df["Anomaly"].rolling(window=10, min_periods=1).sum()

    # Time since last anomaly
    df["last_anomaly_time"] = df["DateTimestamp"].where(df["Anomaly"] > 0).ffill()
    df["time_since_last_anomaly"] = (df["DateTimestamp"] - df["last_anomaly_time"]).dt.total_seconds()
    df["time_since_last_anomaly"].fillna(0, inplace=True)

    # Lag features
    for lag in range(1, 6):
        df[f"Anomaly_lag_{lag}"] = df["Anomaly"].shift(lag)

    df.dropna(inplace=True)

    # Time until next anomaly
    df["next_anomaly_time"] = df["DateTimestamp"].where(df["Anomaly"] > 0).bfill()
    df["time_until_next_anomaly"] = (df["next_anomaly_time"] - df["DateTimestamp"]).dt.total_seconds()
    df.dropna(inplace=True)
    df["Plc_Line"] = df["Plc_Line"].astype(int)
    df["BatchNumber"] = df["BatchNumber"].astype(int)
    df["last_anomaly_seconds"] = (df["last_anomaly_time"] - df["DateTimestamp"].min()).dt.total_seconds()
    df["next_anomaly_seconds"] = (df["next_anomaly_time"] - df["DateTimestamp"].min()).dt.total_seconds()
 
    required_cols = [
        "DateTimestamp", "Plc_Line", "WorkOrder", "BatchNumber", "ProductSerial", "Anomaly",
        "hour", "day_of_week", "day_of_month", "month", "weekend",
        "sin_hour", "cos_hour", "rolling_anomaly_10", "last_anomaly_time",
        "time_since_last_anomaly", "Anomaly_lag_1", "Anomaly_lag_2", "Anomaly_lag_3",
        "Anomaly_lag_4", "Anomaly_lag_5", "last_anomaly_seconds", "time_until_next_anomaly"
    ]

    df = df[required_cols]

    return df



def prepare_and_save_dataset(json_path: str, output_csv: str = f"prepared_dataset/{time()}.csv") -> pd.DataFrame:
    """
    Transforms the API response data into a processed anomaly-aware DataFrame with time-based features.

    Parameters:
        json_path (str): Path to the JSON file containing transformed rows.
        output_csv (str): Path to save the output CSV.

    Returns:
        pd.DataFrame: The processed DataFrame.
    """
    with open(json_path, "r") as file:
        data = json.load(file)

    records = data["actions"][0]["obj"]

    # Load into DataFrame and rename for consistency
    df = pd.DataFrame(records)[[
        "DateTimestamp", "selected_plc_line", "work_order",
        "batch_no", "serial_no"
    ]]

    df = prepare_dataset(df)

    # Save and return
    df.to_csv(output_csv, index=False)
    return df

df = prepare_and_save_dataset("response.json")
print(df.head())