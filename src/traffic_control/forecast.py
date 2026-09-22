"""Dependency-light ST graph forecaster; optional torch implementation can replace it."""
import numpy as np

from .simulator import Network


class GraphForecaster:
    def __init__(self, horizon=5, seed=7, network=None, diffusion=0.35):
        self.horizon=horizon; self.rng=np.random.default_rng(seed)
        self.network=network or Network(); self.diffusion=diffusion; self.adjacency=self.network.adjacency()
    def predict(self, history):
        x=np.asarray(history); last=x[-1]; trend=last-x[-2] if len(x)>1 else np.zeros_like(last)
        neighbor_trend=self.adjacency@trend; neighbor_level=self.adjacency@last
        own_trend=(1-self.diffusion)*trend+self.diffusion*neighbor_trend
        level_pull=self.diffusion*0.1*(neighbor_level-last)
        pred=np.maximum(0,last[None,:]+level_pull[None,:]+np.arange(1,self.horizon+1)[:,None]*.5*own_trend[None,:])
        uncertainty=np.maximum(.1,np.std(x,axis=0)+.25*np.abs(own_trend))
        return pred, np.broadcast_to(uncertainty,(self.horizon,len(last)))
    @staticmethod
    def loss(pred,target):
        err=np.asarray(pred)-np.asarray(target); return float(np.mean(np.abs(err))),float(np.sqrt(np.mean(err**2)))

