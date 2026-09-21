import json,time
from .config import Config
from .simulator import SyntheticTraffic
from .forecast import GraphForecaster
from .baselines import FixedTime,Actuated
from .mappo import MAPPO
from .metrics import Metrics

def run_controller(name,cfg,episodes=1):
    factory={"fixed_time":FixedTime,"actuated":Actuated,"mappo":lambda:MAPPO(False),"forecast_mappo":lambda:MAPPO(True),"forecast_uncertainty_mappo":lambda:MAPPO(True,True)}[name]; results=Metrics(); f=GraphForecaster(cfg.forecast_horizon,cfg.seed)
    for ep in range(episodes):
        env=SyntheticTraffic(cfg.seed+ep,cfg.disruption,cfg.sensor_noise,cfg.missing_sensor_rate); obs=env.reset(); hist=[obs]; ctrl=factory()
        for _ in range(cfg.episode_steps):
            pred,unc=f.predict(hist[-cfg.history_length:]) if len(hist)>1 else (None,None); start=time.perf_counter(); act=ctrl.action(obs,pred,unc); latency=(time.perf_counter()-start)*1000; obs,row,done=env.step(act); hist.append(obs); results.add(row,latency)
    return results.summary()

def evaluate(cfg=None,episodes=1):
    cfg=cfg or Config(); return {n:run_controller(n,cfg,episodes) for n in ["fixed_time","actuated","mappo","forecast_mappo","forecast_uncertainty_mappo"]}

