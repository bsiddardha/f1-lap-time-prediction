from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("xgboost_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

CAT_COLS = ["Driver", "Team", "Circuit", "Compound"]
NUM_COLS = ["TrackStatus", "InPit", "LapNumber", "Position", "Stint",
            "TyreLife", "PrevLapTime", "AvgLast3Laps", "AvgLast5Laps",
            "AirTemp", "TrackTemp", "Humidity", "Rainfall"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])

        # Cast categoricals
        for col in CAT_COLS:
            if col in df.columns:
                df[col] = df[col].astype("category")

        # Cast numerics — this fixes the TrackStatus: object error
        for col in NUM_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Reorder columns to match training
        df = df[feature_columns]

        prediction = model.predict(df)[0]
        return jsonify({"predicted_next_lap_time": round(float(prediction), 3)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)