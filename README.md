# Forecast-Aware Traffic Signal Control

An early-stage experiment: do short-horizon traffic forecasts, and an estimate of their uncertainty, help learned controllers run a small network of traffic lights better than fixed-time or actuated signals?

**Status: prototype, no results yet.** Everything runs on a deterministic synthetic 2×2 grid (four intersections). The learned controller is a deliberately small stand-in, described below.

## What's here

| Piece | What it actually is |
|---|---|
| Simulator | A synthetic 2×2 intersection grid ([`simulator.py`](src/traffic_control/simulator.py)), with disruption scenarios. A SUMO adapter is planned, not built. |
| Forecaster | A dependency-light graph-diffusion forecaster over the intersection graph ([`forecast.py`](src/traffic_control/forecast.py)), with MAE/RMSE reporting |
| Controllers | Fixed-time and actuated baselines, plus three learned variants: no forecast, forecast, and forecast + uncertainty |
| Learned policy | A shared linear Bernoulli policy per intersection, trained with a clipped, baseline-subtracted policy-gradient (REINFORCE-style) update in NumPy ([`mappo.py`](src/traffic_control/mappo.py)). The module is named `mappo` because a MAPPO actor-critic is the intended replacement; it is **not** MAPPO yet (no critic, no PPO objective). |
| Evaluation | Waiting and travel time, queue length, throughput, forecast error, and controller latency, written to JSON with the full config |
| Serving | Optional FastAPI inference endpoint ([`api.py`](src/traffic_control/api.py)) |
| Data | A loader that turns a traffic-count CSV (for example from AIKosh) into a demand profile ([`aikosh.py`](src/traffic_control/aikosh.py)) |

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
pytest -q                                   # 7 tests

python -m traffic_control.cli smoke --seed 7
python -m traffic_control.cli evaluate --episodes 3 --steps 120 --train-episodes 15 --output runs/v1.json
```

Each learned variant trains for `--train-episodes` unrecorded episodes before the recorded evaluation episodes. Optional extras: `pip install -e '.[ml,serve,tracking]'` for PyTorch/PyG, FastAPI, and MLflow.

## Next steps

1. Replace the linear policy with a real MAPPO actor-critic (PyTorch).
2. Run on SUMO with a real city network instead of the synthetic grid.
3. Report results across seeds with confidence intervals, comparing all five controllers.

Design notes: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · experiment protocol: [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).
