"""Dependency-light ST graph forecaster; optional torch implementation can replace it."""
import numpy as np

class GraphForecaster:
    def __init__(self, horizon=5, seed=7): self.horizon=horizon; self.rng=np.random.default_rng(seed)
    def predict(self, history):
        x=np.asarray(history); last=x[-1]; trend=last-x[-2] if len(x)>1 else np.zeros_like(last)
        pred=np.maximum(0,last[None,:]+np.arange(1,self.horizon+1)[:,None]*.5*trend[None,:])
        uncertainty=np.maximum(.1,np.std(x,axis=0)+.25*np.abs(trend))
        return pred, np.broadcast_to(uncertainty,(self.horizon,len(last)))
    @staticmethod
    def loss(pred,target):
        err=np.asarray(pred)-np.asarray(target); return float(np.mean(np.abs(err))),float(np.sqrt(np.mean(err**2)))

