from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("f1_xgboost_model.pkl")

# Load feature columns
feature_columns = joblib.load("f1_feature_columns.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Receive JSON from fetch()
        data = request.get_json()

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Same encoding used during training
        df = pd.get_dummies(
            df,
            columns=[
                "Driver",
                "Team",
                "Circuit",
                "Compound",
                "TrackStatus"
            ]
        )

        # Match training columns
        df = df.reindex(
            columns=feature_columns,
            fill_value=0
        )

        # Predict
        prediction = model.predict(df)[0]

        return jsonify({
            "predicted_next_lap_time":
                round(float(prediction), 3)
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)