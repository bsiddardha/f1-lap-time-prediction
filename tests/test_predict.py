import pytest
from app import app

client = app.test_client()

# Correct full payload matching all 17 training features
VALID_PAYLOAD = {
    "Driver": "VER",
    "Team": "Red Bull Racing",
    "Circuit": "Italian Grand Prix",
    "Compound": "SOFT",
    "TrackStatus": 1,
    "LapNumber": 10,
    "TyreLife": 5,
    "Stint": 2,
    "Position": 1,
    "InPit": 0,
    "PrevLapTime": 91.234,
    "AvgLast3Laps": 91.5,
    "AvgLast5Laps": 91.8,
    "AirTemp": 24.0,
    "TrackTemp": 38.0,
    "Humidity": 60.0,
    "Rainfall": 0
}

def test_predict_endpoint():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200

def test_predict_returns_prediction():
    response = client.post("/predict", json=VALID_PAYLOAD)
    data = response.get_json()
    assert "predicted_next_lap_time" in data
    assert isinstance(data["predicted_next_lap_time"], float)