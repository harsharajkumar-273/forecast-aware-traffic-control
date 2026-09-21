"""Optional FastAPI inference service for the controller boundary."""
try:
    from fastapi import FastAPI
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    FastAPI = None

from .config import Config
from .forecast import GraphForecaster
from .mappo import MAPPO

if FastAPI:
    app = FastAPI(title="Forecast-Aware Traffic Control", version="0.1.0")
    forecaster = GraphForecaster(Config().forecast_horizon)
    controllers = {"mappo": MAPPO(False), "forecast": MAPPO(True), "forecast_uncertainty": MAPPO(True, True)}

    class InferenceRequest(BaseModel):
        history: list[list[float]] = Field(min_length=2)
        controller: str = "forecast_uncertainty"

    @app.post("/v1/infer")
    def infer(req: InferenceRequest):
        pred, unc = forecaster.predict(req.history)
        controller = controllers[req.controller]
        action = controller.action(req.history[-1], pred, unc)
        return {"actions": action, "forecast": pred.tolist(), "uncertainty": unc.tolist()}
else:
    app = None

