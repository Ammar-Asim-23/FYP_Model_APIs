import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model

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
        "Anomaly_lag_4", "Anomaly_lag_5", "time_until_next_anomaly"
    ]

    df = df[required_cols]

    return df

def prepare_X_sequences_predict(data, seq_length):
    X = []
    for i in range(len(data) - seq_length + 1):
        X.append(data[i:i+seq_length])
    return np.array(X)

def predict_next_anomaly(json_data, artifact_dir="model_artifacts"):
    """
    End-to-end function that takes a JSON response, processes it, and predicts when the next anomaly will occur.
    Only uses the minimum required samples for prediction.
    
    Parameters:
        json_data (dict or str): JSON data or path to JSON file
        artifact_dir (str): Directory containing model artifacts
        
    Returns:
        dict: Prediction results containing:
            - current_timestamp: Last timestamp in the data
            - predicted_seconds: Predicted seconds until next anomaly
            - predicted_anomaly_time: Datetime of predicted next anomaly
            - confidence_score: Confidence score (0-1) of the prediction
    """
    # Load the JSON data
    if isinstance(json_data, str):
        with open(json_data, 'r') as f:
            data = json.load(f)
    else:
        data = json_data
    
    # Load model artifacts
    artifacts = load_latest_artifacts(artifact_dir)
    model = artifacts['model']
    X_scaler = artifacts['X_scaler']
    y_scaler = artifacts['y_scaler']
    
    # Process the data
    df = process_input_data(data)
    
    # Get the sequence length from the model's input shape
    seq_length = model.input_shape[1]
    
    # Only take the minimum required samples (most recent)
    if len(df) > seq_length:
        df = df.iloc[-seq_length:]
    
    # Prepare sequences for the model
    X_pred = prepare_prediction_sequence(df, X_scaler, seq_length)
    
    # Make prediction
    y_pred_scaled = model.predict(X_pred)
    
    # Inverse transform the prediction to get seconds
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    predicted_seconds = 5 * round(float(y_pred[0][0])/5)    
    # Get the current timestamp (last timestamp in the dataset)
    current_timestamp = len(df["DateTimestamp"])
    
    # Calculate the predicted anomaly time
    predicted_anomaly_time = current_timestamp + timedelta(seconds=predicted_seconds)
    
    # Calculate a simple confidence score based on model metrics
    # This is a placeholder - you might want to implement a more sophisticated confidence metric
    mae = artifacts['history']['val_mae'].values[-1] if 'val_mae' in artifacts['history'] else None
    
    # Lower confidence if we don't have enough data
    data_sufficiency_factor = min(1.0, len(df) / seq_length)
    confidence_score = calculate_confidence_score(predicted_seconds, mae) * data_sufficiency_factor
    
    # Format the result
    result = {
        "current_timestamp": current_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "predicted_seconds": predicted_seconds,
        "predicted_anomaly_time": predicted_anomaly_time.strftime("%Y-%m-%d %H:%M:%S"),
        "confidence_score": round(confidence_score, 3),
        "samples_used": len(df),
        "minimum_samples_required": seq_length
    }
    
    return result

def load_latest_artifacts(base_dir="model_artifacts"):
    """
    Load artifacts from the latest timestamped directory
    Returns: Dictionary of loaded artifacts
    """
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    # Get all timestamp directories
    try:
        dirs = [d for d in os.listdir(base_dir) 
               if os.path.isdir(os.path.join(base_dir, d)) and d.replace('.', '').isdigit()]
    except FileNotFoundError:
        dirs = []
        
    if not dirs:
        raise FileNotFoundError(f"No artifact directories found in {base_dir}")

    # Find latest directory
    latest_dir = max(dirs, key=lambda x: float(x))
    artifact_dir = os.path.join(base_dir, latest_dir)

    # Load artifacts
    artifacts = {
        'model': load_model(os.path.join(artifact_dir, "lstm_model.keras")),
        'history': pd.read_csv(os.path.join(artifact_dir, "training_history.csv")),
        'y_scaler': joblib.load(os.path.join(artifact_dir, "y_scaler.joblib")),
        'X_scaler': joblib.load(os.path.join(artifact_dir, "X_scaler.joblib")),
        'artifact_dir': artifact_dir
    }

    return artifacts

def process_input_data(data):
    """Process the input JSON data similar to prepare_anomaly_dataset function"""
    try:
        records = data["actions"][0]["obj"]
    except (KeyError, IndexError):
        raise ValueError("Invalid JSON format: expected 'actions[0].obj' structure")

    # Load into DataFrame and rename for consistency
    df = pd.DataFrame(records)[[
        "DateTimestamp", "selected_plc_line", "work_order",
        "batch_no", "serial_no"
    ]]

    df = prepare_dataset(df)

    return df

def prepare_prediction_sequence(df, X_scaler, seq_length):
    """
    Prepare the sequence from the dataframe for prediction
    Only uses the latest data and pads if necessary
    """
    # Drop columns that won't be used in prediction
    drop_cols = ["DateTimestamp", "ProductSerial", "last_anomaly_time","time_until_next_anomaly"]
    pred_df = df.drop(columns=drop_cols)

    # Normalize the data
    values = pred_df.values
    scaled_values = X_scaler.transform(values)

    # Pad with zeros if we don't have enough data
    actual_length = len(scaled_values)
    if actual_length < seq_length:
        pad_length = seq_length - actual_length
        pad_array = np.zeros((pad_length, scaled_values.shape[1]))
        scaled_values = np.vstack([pad_array, scaled_values])

    # Use the helper function to get the last valid sequence
    X_sequences = prepare_X_sequences_predict(scaled_values, seq_length)
    # Only the last sequence is needed for prediction
    return X_sequences


def calculate_confidence_score(predicted_seconds, mae=None):
    """
    Calculate a confidence score for the prediction
    
    This is a simple placeholder implementation. You might want to implement
    a more sophisticated confidence metric based on model uncertainty.
    """
    if mae is None:
        # If we don't have MAE, use a simple heuristic based on the prediction value
        # Very short or very long predictions might be less reliable
        if predicted_seconds < 60:  # Less than a minute
            return 0.5
        elif predicted_seconds > 86400:  # More than a day
            return 0.6
        else:
            return 0.8
    else:
        # With MAE, we can calculate something more meaningful
        # Higher MAE = lower confidence
        confidence = max(0, min(1, 1.0 - (mae / 3600)))  # Normalize MAE as fraction of an hour
        return confidence

# Simple usage example
def predict_from_file(json_path="response.json"):
    """Return a single prediction result"""
    result = predict_next_anomaly(json_path)
    return result

def predict_from_api_response(api_response):
    """Return a single prediction result from API response"""
    result = predict_next_anomaly(api_response)
    return result

# Example usage
if __name__ == "__main__":
    try:
        # File-based prediction
        prediction = predict_from_file("response.json")
        print(prediction)
    except Exception as e:
        print(f"Error: {str(e)}")