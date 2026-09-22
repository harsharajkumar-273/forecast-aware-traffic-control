import numpy as np

from traffic_control.mappo import MAPPO
from traffic_control.simulator import SyntheticTraffic


def test_mappo_update_changes_weights():
 ctrl=MAPPO(forecast=False,seed=1); ctrl.train(True)
 env=SyntheticTraffic(seed=1,episode_steps=20); obs=env.reset()
 for _ in range(20):
  act=ctrl.action(obs); obs,row,done=env.step(act); ctrl.record_reward(-row["queue_length"])
  if done: break
 before=ctrl.weights.copy(); info=ctrl.update()
 assert info["episodes"]==1
 assert not np.allclose(before,ctrl.weights)
def test_mappo_greedy_when_not_training():
 ctrl=MAPPO(forecast=False,seed=1)
 act=ctrl.action([1.0]*8)
 assert len(act)==4 and all(a in (0,1) for a in act)
