"""Deterministic synthetic network plus a narrow future SUMO adapter boundary."""
from dataclasses import dataclass
import numpy as np

@dataclass
class Network:
    n: int = 4
    edges: tuple = ((0,1),(1,0),(2,3),(3,2),(0,2),(2,0),(1,3),(3,1))

class SyntheticTraffic:
    def __init__(self, seed=7, disruption="none", noise=0.0, missing=0.0):
        self.rng=np.random.default_rng(seed); self.network=Network(); self.noise=noise; self.missing=missing; self.disruption=disruption
        self.reset()
    def reset(self):
        self.t=0; self.queue=np.zeros(8,dtype=float); self.passed=0; self.travel=0.; self.wait=0.; return self.observe()
    def observe(self):
        q=self.queue.copy()
        if self.disruption == "demand_spike" and 50 <= self.t < 100: q += 3.0
        if self.disruption == "closure": q[4:6] += 5.0
        if self.noise: q += self.rng.normal(0,self.noise,q.shape)
        if self.missing: q[self.rng.random(q.shape)<self.missing]=0
        return np.maximum(q,0)
    def step(self, actions):
        actions=np.asarray(actions); phase=actions % 2
        arrivals=self.rng.poisson(1.1,8).astype(float)
        if self.disruption == "demand_spike" and 50 <= self.t < 100: arrivals *= 2.5
        if self.disruption == "closure": arrivals[4:6] *= .1
        self.queue += arrivals
        served=np.array([min(self.queue[e], 2.0 if phase[e//2] == e%2 else .4) for e in range(8)])
        self.queue=np.maximum(0,self.queue-served); self.passed += served.sum(); self.wait += self.queue.sum(); self.travel += self.queue.sum()+arrivals.sum()
        self.t += 1
        return self.observe(), {"waiting_time":float(self.wait),"travel_time":float(self.travel),"queue_length":float(self.queue.sum()),"throughput":float(self.passed)}, self.t>=240

