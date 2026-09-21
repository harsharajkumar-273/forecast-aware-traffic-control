# V1 experiment protocol

Every controller is evaluated with the same seed set and episode length. Scenarios are `none`, `demand_spike`, `closure`, and sensor corruption variants. Report mean and standard deviation over at least 10 seeds for waiting time, travel time, queue length, throughput, forecast MAE/RMSE, and p50/p95 inference latency. Compare the five controllers in the README and retain raw JSON plus the config used.

The scientific ablation is forecast value: MAPPO vs forecast-aware MAPPO vs forecast+uncertainty-aware MAPPO. Do not claim improvement until the confidence intervals and disruption results are recorded.

