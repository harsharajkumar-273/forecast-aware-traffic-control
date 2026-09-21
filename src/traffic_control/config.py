from dataclasses import dataclass, asdict
import json

@dataclass
class Config:
    seed: int = 7
    episode_steps: int = 240
    forecast_horizon: int = 5
    history_length: int = 8
    control_interval: int = 5
    min_green: int = 5
    max_green: int = 45
    sensor_noise: float = 0.0
    missing_sensor_rate: float = 0.0
    disruption: str = "none"
    agents: int = 4

    def to_dict(self): return asdict(self)
    def to_json(self): return json.dumps(self.to_dict(), sort_keys=True)

