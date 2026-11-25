from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "telco.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "data_drift.html"

FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "InternetService",
]


def generate_drift_report():
    df = pd.read_csv(DATA_PATH)

    reference = df[FEATURES].sample(frac=0.7, random_state=42)
    current = df[FEATURES].drop(reference.index)

    report = Report([
        DataDriftPreset(),
    ])

    result = report.run(
        reference_data=reference,
        current_data=current,
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.save_html(str(REPORT_PATH))

    print(f"Data drift report generated: {REPORT_PATH}")


if __name__ == "__main__":
    generate_drift_report()
