import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("f1_laptime_dataset_2021_2025.csv")

cat_cols = ["Driver", "Team", "Circuit", "Compound"]

for col in cat_cols:
    df[col] = df[col].astype("category")

# Ensure numeric columns are proper types
num_cols = ["TrackStatus", "InPit", "LapNumber", "Position", "Stint",
            "TyreLife", "PrevLapTime", "AvgLast3Laps", "AvgLast5Laps",
            "AirTemp", "TrackTemp", "Humidity", "Rainfall"]
for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

X = df.drop("NextLapTime", axis=1)
y = df["NextLapTime"]

# Save feature order for inference
joblib.dump(list(X.columns), "feature_columns.pkl")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    enable_categorical=True,
    random_state=42
)

model.fit(X_train, y_train)
preds = model.predict(X_test)
print("MAE:", round(mean_absolute_error(y_test, preds), 3))

joblib.dump(model, "xgboost_model.pkl")
print("Model saved!")