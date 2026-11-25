from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from mlflow_config import configure_mlflow, MODEL_NAME


DATA_PATH = Path("data/processed/telco.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES = [
    "Contract",
    "InternetService",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Churn"


def build_preprocessor():
    return ColumnTransformer([
        (
            "numeric",
            StandardScaler(),
            NUMERIC_FEATURES,
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            CATEGORICAL_FEATURES,
        ),
    ])


def build_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),

        "SVM": CalibratedClassifierCV(
            estimator=SVC(
                class_weight="balanced",
                random_state=42,
            ),
            method="sigmoid",
            cv=5,
            ensemble=False,
        ),
    }


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    configure_mlflow()

    print("\nLoading processed dataset...")
    df = pd.read_csv(DATA_PATH)

    if df.isnull().any().any():
        raise ValueError("Dataset contains missing values.")

    X = df[FEATURES]
    y = df[TARGET]

    if set(y.unique()) != {"Yes", "No"}:
        raise ValueError("Unexpected target values in Churn.")

    y = y.map({"No": 0, "Yes": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"Dataset records: {len(df)}")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")

    results = []
    trained_models = {}

    for name, classifier in build_models().items():
        print(f"\n========== {name} ==========")

        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ])

        with mlflow.start_run(run_name=name) as run:
            pipeline.fit(X_train, y_train)

            predictions = pipeline.predict(X_test)
            probabilities = pipeline.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(
                    y_test, predictions, zero_division=0
                ),
                "recall": recall_score(
                    y_test, predictions, zero_division=0
                ),
                "f1": f1_score(
                    y_test, predictions, zero_division=0
                ),
                "roc_auc": roc_auc_score(y_test, probabilities),
            }

            mlflow.log_param("classifier", name)
            mlflow.log_param("test_size", 0.20)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("features", ",".join(FEATURES))

            for param, value in classifier.get_params().items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    mlflow.log_param(f"model_{param}", value)

            mlflow.log_metrics(metrics)

            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                input_example=X_train.head(3),
                skops_trusted_types=[
                    "sklearn.tree._tree.Tree",
                    "sklearn.calibration._CalibratedClassifier",
                    "sklearn.calibration._SigmoidCalibration",
                ],
            )

            model_path = MODEL_DIR / (
                name.lower().replace(" ", "_") + ".joblib"
            )
            joblib.dump(pipeline, model_path)

            trained_models[name] = {
                "run_id": run.info.run_id,
                "model": pipeline,
                "metrics": metrics,
            }

            results.append({
                "model": name,
                **metrics,
                "run_id": run.info.run_id,
            })

            print(f"Run ID: {run.info.run_id}")

            for metric, value in metrics.items():
                print(f"{metric:12}: {value:.4f}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="f1",
        ascending=False,
    )

    results_df.to_csv(
        REPORT_DIR / "model_comparison.csv",
        index=False,
    )

    print("\n========== MODEL COMPARISON ==========")
    print(results_df.to_string(index=False))

    champion_name = results_df.iloc[0]["model"]
    champion_run_id = trained_models[champion_name]["run_id"]

    print(f"\nRegistering candidate: {champion_name}")

    model_uri = f"runs:/{champion_run_id}/model"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )

    print(f"Registered model: {registered_model.name}")
    print(f"Model version: {registered_model.version}")

    client = mlflow.MlflowClient()

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="Champion",
        version=registered_model.version,
    )

    print("Champion alias assigned successfully.")
    print("\nTraining and MLflow registration completed.")


if __name__ == "__main__":
    main()