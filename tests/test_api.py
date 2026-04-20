from fastapi.testclient import TestClient
import sys
import os

# Add the project root to sys.path to allow importing from api.main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Iris Prediction API"}

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    # Note: model_loaded depends on if train.py was run
    assert "status" in response.json()

def test_predict_validation():
    # Test invalid data
    response = client.post("/predict", json={"invalid": "data"})
    assert response.status_code == 422
