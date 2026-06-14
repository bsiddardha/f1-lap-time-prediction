from app import app

client = app.test_client()


def test_predict_endpoint():

    payload = {
        "Driver": "Max Verstappen",
        "Team": "Red Bull",
        "Circuit": "Monza",
        "Compound": "Soft",
        "TrackStatus": "Green",
        "LapNumber": 10,
        "TyreLife": 5,
        "Position": 1
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 200


def test_predict_returns_prediction():

    payload = {
        "Driver": "Max Verstappen",
        "Team": "Red Bull",
        "Circuit": "Monza",
        "Compound": "Soft",
        "TrackStatus": "Green",
        "LapNumber": 10,
        "TyreLife": 5,
        "Position": 1
    }

    response = client.post(
        "/predict",
        json=payload
    )

    data = response.get_json()

    assert "predicted_next_lap_time" in data