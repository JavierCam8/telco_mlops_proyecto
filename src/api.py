import mlflow
import pandas as pd

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.mlflow_config import MLFLOW_TRACKING_URI, MODEL_NAME


class CustomerData(BaseModel):
    tenure: int = Field(ge=0)
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)
    Contract: Literal["Month-to-month", "One year", "Two year"]
    InternetService: Literal["DSL", "Fiber optic", "No"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_uri = f"models:/{MODEL_NAME}@Champion"
    app.state.model = mlflow.pyfunc.load_model(model_uri)

    print(f"Loaded model: {model_uri}")

    yield


app = FastAPI(
    title="Telco Customer Churn API",
    description="API for predicting customer churn using the registered MLflow Champion model.",
    version="1.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": f"{MODEL_NAME}@Champion",
    }


@app.post("/predict")
def predict(customer: CustomerData):
    features = pd.DataFrame([customer.model_dump()])

    prediction = int(app.state.model.predict(features)[0])

    return {
        "prediction": prediction,
        "churn": "Yes" if prediction == 1 else "No",
        "model": f"{MODEL_NAME}@Champion",
    }
