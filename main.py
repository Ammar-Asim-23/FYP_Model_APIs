# To run server, try fastapi dev main.py
from fastapi import FastAPI, File, UploadFile, HTTPException,  Request
from fastapi.responses import JSONResponse
import uvicorn
import shutil
import os
import json
from model_utils import (
    prepare_and_save_dataset, train_lstm_model, save_artifacts,
    predict_next_anomaly
)

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "LSTM Anomaly Prediction API is running."}

@app.post("/preprocess/")
async def preprocess_data(request: Request):
    """
    Accept nested JSON, extract records, preprocess, and return info.
    """
    try:
        # Read full JSON body
        payload = await request.json()


        # Preprocess (replace with your actual preprocessing logic)
        processed_df = prepare_and_save_dataset(payload)

        return {"message": "Preprocessing completed", "rows": len(processed_df)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train/")
def train_model():
    """
    Train the LSTM model from latest prepared dataset and save artifacts.
    """
    try:
        latest_files = sorted(os.listdir("prepared_dataset"), reverse=True)
        if not latest_files:
            raise FileNotFoundError("No prepared dataset found.")

        csv_path = os.path.join("prepared_dataset", latest_files[0])
        model, history, X_test, y_test, y_scaler, X_scaler = train_lstm_model(csv_path)
        artifact_dir = save_artifacts(model, history, X_test, y_test, y_scaler, X_scaler)

        return {"message": "Training completed", "artifact_dir": artifact_dir}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/")
async def predict_anomaly(request: Request):
    """
    Predict the next anomaly time from a JSON file.
    """
    try:
        # Save uploaded file
        json_data = await request.json()

        prediction = predict_next_anomaly(json_data)
        return JSONResponse(content=prediction)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
