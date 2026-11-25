import mlflow
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
MLFLOW_ARTIFACTS_DIR = PROJECT_ROOT / "mlartifacts"

MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB.as_posix()}"

EXPERIMENT_NAME = "Telco Customer Churn Airflow"
MODEL_NAME = "TelcoChurnClassifier"


def configure_mlflow():
    MLFLOW_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        mlflow.create_experiment(
            name=EXPERIMENT_NAME,
            artifact_location=MLFLOW_ARTIFACTS_DIR.resolve().as_uri(),
        )

    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"MLflow artifacts directory: {MLFLOW_ARTIFACTS_DIR}")
    print(f"MLflow experiment: {EXPERIMENT_NAME}")
