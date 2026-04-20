from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import pandas as pd

app = FastAPI(title="MLOps Production API", version="1.0.0")

# Define the input data schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Target names mapping
TARGET_NAMES = ["setosa", "versicolor", "virginica"]

# Load the model during startup
MODEL_PATH = "models/model.joblib"
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
    else:
        print(f"Model not found at {MODEL_PATH}. Please run training script first.")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Prediction API"}

@app.get("/health")
def health_check():
    if model:
        return {"status": "healthy", "model_loaded": True}
    return {"status": "unhealthy", "model_loaded": False}

@app.post("/predict")
def predict(data: IrisInput):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded. Try again later.")
    
    # Prepare data for prediction
    input_df = pd.DataFrame([data.dict().values()], columns=[
        "sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"
    ])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df).tolist()[0]
    
    return {
        "prediction": int(prediction),
        "class_name": TARGET_NAMES[prediction],
        "probabilities": probability
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
