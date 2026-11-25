import pandas as pd
import pytest

from src import prepare_data


def test_missing_total_charges_with_zero_tenure(monkeypatch, tmp_path):
    raw_file = tmp_path / "raw.csv"
    processed_file = tmp_path / "processed" / "telco.csv"

    df = pd.DataFrame({
        "customerID": ["A", "B"],
        "tenure": [0, 12],
        "TotalCharges": [" ", "500.50"],
        "Churn": ["No", "Yes"],
    })

    df.to_csv(raw_file, index=False)

    monkeypatch.setattr(prepare_data, "RAW_DATA", raw_file)
    monkeypatch.setattr(prepare_data, "PROCESSED_DATA", processed_file)

    prepare_data.main()

    result = pd.read_csv(processed_file)

    assert result.loc[0, "TotalCharges"] == 0.0
    assert result.loc[1, "TotalCharges"] == 500.50
    assert result["TotalCharges"].isna().sum() == 0


def test_missing_total_charges_with_nonzero_tenure(monkeypatch, tmp_path):
    raw_file = tmp_path / "raw.csv"

    df = pd.DataFrame({
        "customerID": ["A"],
        "tenure": [12],
        "TotalCharges": [" "],
        "Churn": ["Yes"],
    })

    df.to_csv(raw_file, index=False)

    monkeypatch.setattr(prepare_data, "RAW_DATA", raw_file)

    with pytest.raises(ValueError, match="tenure is not zero"):
        prepare_data.main()
