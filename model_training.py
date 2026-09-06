"""Train and evaluate the house-price regression pipeline."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "house_price_prediction_dataset.csv"
MODEL_PATH = PROJECT_DIR / "trained_house_price_model.joblib"
METRICS_PATH = PROJECT_DIR / "model_metrics.csv"
PREDICTIONS_PATH = PROJECT_DIR / "test_predictions.csv"


def build_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    transformers = []

    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore", sparse_output=False
                            ),
                        ),
                    ]
                ),
                categorical_features,
            )
        )

    return Pipeline(
        [
            ("preprocessor", ColumnTransformer(transformers=transformers)),
            ("regressor", LinearRegression()),
        ]
    )


def main() -> None:
    df = pd.read_csv(DATASET_PATH)
    target = "Price"
    features = [column for column in df.columns if column not in {target, "House_ID"}]
    X = df[features]
    y = df[target]

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_pipeline(numeric_features, categorical_features)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = pd.DataFrame(
        [
            {
                "model": "LinearRegression",
                "test_size": 0.2,
                "random_state": 42,
                "features": ", ".join(features),
                "mae": mean_absolute_error(y_test, predictions),
                "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
                "r2": r2_score(y_test, predictions),
            }
        ]
    )
    metrics.to_csv(METRICS_PATH, index=False)

    pd.DataFrame(
        {"actual_price": y_test.to_numpy(), "predicted_price": predictions}
    ).to_csv(PREDICTIONS_PATH, index=False)
    joblib.dump(model, MODEL_PATH)

    print("Training complete")
    print(metrics.to_string(index=False))
    print(f"Saved model: {MODEL_PATH.name}")
    print(f"Saved metrics: {METRICS_PATH.name}")
    print(f"Saved predictions: {PREDICTIONS_PATH.name}")


if __name__ == "__main__":
    main()
