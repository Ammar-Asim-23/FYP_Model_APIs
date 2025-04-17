import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input
import tensorflow as tf
import time

def prepare_X_sequences(data, seq_length):
    X = []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:i+seq_length].values)
    return np.array(X)
def prepare_y_sequences(data, target_col, seq_length):
    y = []
    for i in range(len(data) - seq_length):
        y.append(data.iloc[i + seq_length][target_col])
    return np.array(y)

def build_lstm_model(input_shape, lstm_units=32, dropout_rate=0.22, learning_rate=0.0007):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(lstm_units, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(lstm_units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(1, activation='relu')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

def train_lstm_model(csv_path="prepared_dataset.csv"):
    # Internal default parameters
    target_column = "time_until_next_anomaly"
    seq_length = 55
    lstm_units = 32
    dropout = 0.2239
    learning_rate = 0.0007046
    batch_size = 32
    epochs = 150
    validation_split = 0.2
    shuffle = True
    
    # Load data
    df = pd.read_csv(csv_path, parse_dates=["DateTimestamp", "last_anomaly_time", "next_anomaly_time"])
    df.drop(columns=["ProductSerial"], inplace=True)
    df.drop(columns=["DateTimestamp", "last_anomaly_time", "next_anomaly_time"], inplace=True)


    # Normalize features
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    numerical_cols = df.columns.difference([target_column])
    df[numerical_cols] = X_scaler.fit_transform(df[numerical_cols])
    df[[target_column]] = y_scaler.fit_transform(df[[target_column]])
    # Prepare sequences
    X = prepare_X_sequences(df[numerical_cols], seq_length)
    y = prepare_y_sequences(df, target_column, seq_length)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_split, shuffle=shuffle, random_state=42)
    # Build and train model
    model = build_lstm_model((seq_length, X_train.shape[2]), lstm_units, dropout, learning_rate)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=batch_size,
        epochs=epochs,
        verbose=1
    )

    return model, history, X_test, y_test, y_scaler, X_scaler

def save_artifacts(model, history, X_test, y_test, y_scaler, X_scaler, 
                  base_dir="model_artifacts"):
    """
    Save all training artifacts in timestamped directory
    Returns: Path to the created artifact directory
    """
    # Create timestamped directory
    timestamp = str(time.time())
    artifact_dir = os.path.join(base_dir, timestamp)
    os.makedirs(artifact_dir, exist_ok=True)
    
    # Save Keras model
    model_path = os.path.join(artifact_dir, "lstm_model.keras")
    save_model(model, model_path)

    # Save training history
    history_path = os.path.join(artifact_dir, "training_history.csv")
    pd.DataFrame(history.history).to_csv(history_path, index=False)

    # Save test data
    np.save(os.path.join(artifact_dir, "X_test.npy"), X_test)
    np.save(os.path.join(artifact_dir, "y_test.npy"), y_test)

    # Save scalers
    joblib.dump(y_scaler, os.path.join(artifact_dir, "y_scaler.joblib"))
    joblib.dump(X_scaler, os.path.join(artifact_dir, "X_scaler.joblib"))

    print(f"Artifacts saved to: {artifact_dir}")
    return artifact_dir

def load_artifacts(base_dir="model_artifacts"):
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
        'model': tf.keras.models.load_model(os.path.join(artifact_dir, "lstm_model.keras")),
        'history': pd.read_csv(os.path.join(artifact_dir, "training_history.csv")),
        'X_test': np.load(os.path.join(artifact_dir, "X_test.npy")),
        'y_test': np.load(os.path.join(artifact_dir, "y_test.npy")),
        'y_scaler': joblib.load(os.path.join(artifact_dir, "y_scaler.joblib")),
        'X_scaler': joblib.load(os.path.join(artifact_dir, "X_scaler.joblib")),
        'artifact_dir': artifact_dir
    }

    print(f"Loaded artifacts from: {artifact_dir}")
    return artifacts

# After training
model, history, X_test, y_test, y_scaler, X_scaler = train_lstm_model("prepared_dataset.csv")

# Save artifacts
file_paths = save_artifacts(model, history, X_test, y_test, y_scaler, X_scaler, "model_artifacts")

# # Later, to load everything back
artifacts = load_artifacts("model_artifacts")

# Access loaded objects
loaded_model = artifacts['model']
loaded_history = artifacts['history']
loaded_X_test = artifacts['X_test']
loaded_y_scaler = artifacts['y_scaler']