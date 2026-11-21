from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import requests
import pandas as pd
import io
import uvicorn
import math
from pathlib import Path

import joblib
import pandas as pd

from features_v2 import TARGET_COLUMNS, engineer_features

# Resolve relative to this file so it works from any working directory (Docker, uvicorn, VM)
MODEL_ARTIFACT_PATH = Path(__file__).parent / "models" / "xgb_vulnerability_models.joblib"

# Load once at startup, not on every request
_artifact = joblib.load(MODEL_ARTIFACT_PATH)
_models = _artifact["models"]
_feature_columns = _artifact["feature_columns"]

app = FastAPI(title="ML Prediction API")


class EvaluationBody(BaseModel):
    testdata_download_endpoint: str
    prediction_upload_endpoint: str


def run_model(df: pd.DataFrame) -> pd.DataFrame:
    # ==== EXAMPLE: ADD YOUR PREDICTIONS OF TRAINED MODEL HERE ====
    # df contains the input time-series metrics for a package (multiple rows).
    # Return a single-row DataFrame containing the target columns required by the competition:
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    # Same logic as before: use the first record
    record_df = df.iloc[[0]].copy()

    X = engineer_features(record_df)
    X = X.reindex(columns=_feature_columns)

    output_record = {}
    if "purl" in record_df.columns:
        output_record["purl"] = record_df["purl"].iloc[0]

    for target in TARGET_COLUMNS:
        value = float(_models[target].predict(X)[0])
        if not math.isfinite(value):   # guards against NaN/inf
            value = 0.0
        output_record[target] = round(max(value, 0.0), 6)

    prediction_df = pd.DataFrame([output_record])
    assert len(prediction_df) == 1
    return prediction_df

def start_prediction(download_url: str, upload_url: str):
    try:
        result = requests.get(download_url, timeout=30)
        result.raise_for_status()

        # Read input time-series CSV (comma-separated, US float format)
        input_data = pd.read_csv(io.StringIO(result.text), sep=",", decimal=".")

        # ADD YOUR PREDICTION MODEL HERE TO PREDICT A SINGLE ROW WITH THE TARGET HEADERS
        # ======== Example =========
        prediction_data = run_model(input_data)
        # ======== END Example =====

        assert len(prediction_data) == 1, "Prediction DataFrame must contain exactly 1 row"
        output_buffer = io.StringIO()
        prediction_data.to_csv(output_buffer, sep=",", decimal=".", header=True, index=False)
        output_buffer.seek(0)

        file = {"predictions": ("prediction.csv", output_buffer.getvalue(), "text/csv")}

        upload_res = requests.post(upload_url, files=file, timeout=30)
        upload_res.raise_for_status()
        print("Evaluation Result:", upload_res.json())

    except Exception as e:
        print("Error:", e)


@app.post("/predict", status_code=200)
async def prediction(payload: EvaluationBody, background_tasks: BackgroundTasks):
    # Start prediction in background
    background_tasks.add_task(
        start_prediction,
        payload.testdata_download_endpoint,
        payload.prediction_upload_endpoint,
    )
    return {"status": "prediction started"}


@app.get("/health", status_code=200)
async def health():
    return {"health": "up"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
