# Forecast-Aware MARL Traffic Control

Production-oriented V1 for testing whether short-horizon traffic forecasts and calibrated uncertainty improve multi-intersection signal control. The first milestone is a deterministic synthetic 2x2 (four-intersection) network; SUMO, PyTorch/PyG, MLflow, and FastAPI are optional integrations.

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
pytest -q
python -m traffic_control.cli smoke --seed 7
python -m traffic_control.cli evaluate --episodes 3 --steps 120 --output runs/v1.json
```

The evaluator compares fixed-time, actuated, MAPPO without forecasts, forecast-aware MAPPO, and forecast+uncertainty-aware MAPPO. It reports waiting/travel time, queue length, throughput, forecast MAE/RMSE, and controller latency. Results are JSON and include the full config for reproducibility.

## Layout

`src/traffic_control/` contains the simulator adapter, graph forecaster, multi-agent environment, MAPPO policy/trainer, baselines, metrics, experiment runner, and optional inference API. `configs/` holds versioned experiment settings. `sumo/` contains the route/network generation boundary for a later SUMO-backed run. `docs/` documents the scaling plan and experiment protocol.

## Optional integrations

Install `.[ml,serve,tracking]` for PyTorch/PyG forecasting, FastAPI serving, and MLflow tracking. The reference implementation uses a dependency-light graph diffusion forecaster so CI and local smoke tests work without GPU or SUMO. When SUMO is installed, the adapter boundary in `simulator.py` is the integration point; V1 intentionally does not require a system-wide SUMO install.

