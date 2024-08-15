import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from mlflow import pyfunc
from mlflow.pyfunc import PyFuncModel
from pydantic import BaseModel

load_dotenv()

app = FastAPI()


class WaterQuality(BaseModel):
    aluminium: float
    ammonia: float
    arsenic: float
    barium: float
    cadmium: float
    chloramine: float
    chromium: float
    copper: float
    flouride: float
    bacteria: float
    viruses: float
    lead: float
    nitrates: float
    nitrites: float
    mercury: float
    perchlorate: float
    radium: float
    selenium: float
    silver: float
    uranium: float


def load_registered_model(
    model_name: str, alias: str = "champion"
) -> PyFuncModel:
    """
    Load the registered model from MLFlow.

    Args:
        model_name (str): Name of the registered model.
        alias (str, optional): Alias of the registered model.
                               Defaults to "champion".

    Returns:
        PyFuncModel: The loaded registered model.
    """
    model_uri = f"models:/{model_name}@{alias}"  # noqa: E231
    loaded_model = pyfunc.load_model(model_uri)
    return loaded_model


@app.get("/")
def read_root():
    return {"message": "Welcome to the water quality prediction API"}


@app.post("/predict/")
def predict(data: WaterQuality) -> dict:
    model_name = os.getenv(
        "REGISTER_MODEL_NAME", default="water_quality_prediction"
    )
    model = load_registered_model(model_name)
    input_payload = data.model_dump()
    prediction = model.predict(input_payload)
    return {"prediction": prediction[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
