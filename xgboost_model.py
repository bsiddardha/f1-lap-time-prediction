import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("f1_laptime_dataset_2021_2025.csv")

print("Original Shape:", df.shape)

# =========================
# BASIC CLEANING
# =========================

df = df.drop_duplicates()
df = df.dropna()

# Remove unrealistic lap times
df = df[
    (df["PrevLapTime"] > 50) &
    (df["PrevLapTime"] < 250)
]

df = df[
    (df["NextLapTime"] > 50) &
    (df["NextLapTime"] < 250)
]

print("Cleaned Shape:", df.shape)

# =========================
# FEATURES / TARGET
# =========================

target = "NextLapTime"

features = [
    "Driver",
    "Team",
    "Circuit",

    "PrevLapTime",
    "AvgLast3Laps",
    "AvgLast5Laps",

    "TyreLife",
    "Compound",
    "Stint",

    "LapNumber",
    "Position",

    "TrackStatus",
    "InPit",

    "AirTemp",
    "TrackTemp",
    "Humidity",
    "Rainfall"
]

X = df[features]
y = df[target]

# =========================
# ENCODE CATEGORICALS
# =========================

categorical_cols = [
    "Driver",
    "Team",
    "Circuit",
    "Compound",
    "TrackStatus"
]

X = pd.get_dummies(
    X,
    columns=categorical_cols,
    drop_first=True
)

print("Feature Matrix Shape:", X.shape)

# =========================
# TRAIN TEST SPLIT
# =========================

split_idx = int(len(X) * 0.8)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]

y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

print("Train Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

# =========================
# XGBOOST MODEL
# =========================

model = XGBRegressor(
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

# =========================
# TRAIN
# =========================

print("\nTraining Model...")

model.fit(
    X_train,
    y_train
)

# =========================
# PREDICT
# =========================

preds = model.predict(X_test)

# =========================
# METRICS
# =========================

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print("\n========== RESULTS ==========")
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R2   : {r2:.4f}")

# =========================
# FEATURE IMPORTANCE
# =========================

importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 20 Features:")
print(importance.head(20))

# =========================
# SAVE MODEL
# =========================

joblib.dump(
    model,
    "f1_xgboost_model.pkl"
)

print("\nModel Saved:")
print("f1_xgboost_model.pkl")

# =========================
# SAVE FEATURE IMPORTANCE
# =========================

importance.to_csv(
    "feature_importance.csv",
    index=False
)

print("feature_importance.csv saved")