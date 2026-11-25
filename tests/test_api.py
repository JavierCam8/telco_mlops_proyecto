import pandas as pd
import pytest

from pydantic import ValidationError
from src import api


class FakeModel:
    def predict(self, features):
        assert isinstance(features, pd.DataFrame)
        return [1]


def test_prediction_returns_churn_yes(monkeypatch):
    monkeypatch.setattr(api.app.state, "model", FakeModel(), raising=False)

    customer = api.CustomerData(
        tenure=12,
        MonthlyCharges=75.5,
        TotalCharges=906.0,
        Contract="Month-to-month",
        InternetService="Fiber optic",
    )

    result = api.predict(customer)

    assert result["prediction"] == 1
    assert result["churn"] == "Yes"
    assert result["model"] == "TelcoChurnClassifier@Champion"


def test_customer_data_rejects_negative_tenure():
    with pytest.raises(ValidationError):
        api.CustomerData(
            tenure=-1,
            MonthlyCharges=75.5,
            TotalCharges=906.0,
            Contract="Month-to-month",
            InternetService="Fiber optic",
        )
