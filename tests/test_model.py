import joblib


def test_model_load():
    model = joblib.load("xgboost_model.pkl")

    assert model is not None


def test_feature_columns_load():
    feature_columns = joblib.load(
        "feature_columns.pkl"
    )

    assert feature_columns is not None
    assert len(feature_columns) > 0