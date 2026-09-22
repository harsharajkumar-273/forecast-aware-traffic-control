import time

from .baselines import Actuated, FixedTime
from .config import Config
from .forecast import GraphForecaster
from .mappo import MAPPO
from .metrics import Metrics
from .simulator import Network, SyntheticTraffic


def run_controller(name,cfg,episodes=1):
    factory={"fixed_time":FixedTime,"actuated":Actuated,"mappo":lambda:MAPPO(False),"forecast_mappo":lambda:MAPPO(True),"forecast_uncertainty_mappo":lambda:MAPPO(True,True)}[name]
    ctrl=factory(); f=GraphForecaster(cfg.forecast_horizon,cfg.seed,network=Network())
    if hasattr(ctrl,"train"):
        ctrl.train(True)
        for ep in range(cfg.train_episodes):
            env=SyntheticTraffic(cfg.seed+1000+ep,cfg.disruption,cfg.sensor_noise,cfg.missing_sensor_rate,episode_steps=cfg.episode_steps)
            obs=env.reset(); hist=[obs]
            for _ in range(cfg.episode_steps):
                pred,unc=f.predict(hist[-cfg.history_length:]) if len(hist)>1 else (None,None)
                act=ctrl.action(obs,pred,unc); obs,row,done=env.step(act); hist.append(obs); ctrl.record_reward(-row["queue_length"])
                if done: break
            ctrl.update()
        ctrl.train(False)
    results=Metrics()
    for ep in range(episodes):
        env=SyntheticTraffic(cfg.seed+ep,cfg.disruption,cfg.sensor_noise,cfg.missing_sensor_rate,episode_steps=cfg.episode_steps); obs=env.reset(); hist=[obs]
        for _ in range(cfg.episode_steps):
            pred,unc=f.predict(hist[-cfg.history_length:]) if len(hist)>1 else (None,None); start=time.perf_counter(); act=ctrl.action(obs,pred,unc); latency=(time.perf_counter()-start)*1000; obs,row,done=env.step(act); hist.append(obs); results.add(row,latency)
            if done: break
    return results.summary()

def evaluate(cfg=None,episodes=1):
    cfg=cfg or Config(); return {n:run_controller(n,cfg,episodes) for n in ["fixed_time","actuated","mappo","forecast_mappo","forecast_uncertainty_mappo"]}

