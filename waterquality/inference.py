import os

import uvicorn
from fastapi import FastAPI
from mlflow import pyfunc
from mlflow.pyfunc import PyFuncModel
from pydantic import BaseModel

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
def read_root() -> dict[str, str]:
    return {"message": "Welcome to the water quality prediction API"}


@app.post("/predict/")
def predict(data: WaterQuality) -> dict[str, float]:
    model_name = os.getenv(
        "REGISTER_MODEL_NAME", default="water_quality_prediction"
    )
    model = load_registered_model(model_name)
    input_payload = data.model_dump()
    prediction = model.predict(input_payload)
    return {"prediction": prediction[0]}


if __name__ == "__main__":
    host = os.getenv("HOST", default="127.0.0.1")
    port = int(os.getenv("PORT", default=8000))
    uvicorn.run(app, host=host, port=port)
