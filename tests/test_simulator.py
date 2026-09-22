from traffic_control.simulator import SyntheticTraffic


def test_episode_length_respected():
 env=SyntheticTraffic(seed=1,episode_steps=5); env.reset()
 for i in range(5):
  _obs,_row,done=env.step([0,0,0,0]); assert done==(i==4)
