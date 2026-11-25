from datetime import datetime, timezone

from airflow import DAG
from airflow.operators.bash import BashOperator


with DAG(
    dag_id="telco_churn_pipeline",
    description="Prepare Telco Churn data, train models, and monitor data drift",
    start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
    schedule=None,
    catchup=False,
    tags=["telco", "mlops", "machine-learning"],
) as dag:

    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command=(
            "set -euo pipefail && "
            "/opt/ml-venv/bin/python src/prepare_data.py"
        ),
        cwd="/opt/airflow/project",
    )

    train_models = BashOperator(
        task_id="train_models",
        bash_command=(
            "set -euo pipefail && "
            "/opt/ml-venv/bin/python src/train.py"
        ),
        cwd="/opt/airflow/project",
    )

    monitor_data = BashOperator(
        task_id="monitor_data",
        bash_command=(
            "set -euo pipefail && "
            "/opt/ml-venv/bin/python -m src.monitor"
        ),
        cwd="/opt/airflow/project",
    )

    prepare_data >> train_models >> monitor_data
