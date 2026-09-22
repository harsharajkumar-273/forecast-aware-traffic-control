from traffic_control.config import Config
from traffic_control.evaluate import evaluate
from traffic_control.forecast import GraphForecaster
from traffic_control.simulator import Network


def test_forecaster_shapes():
 p,u=GraphForecaster(3).predict([[1]*8,[2]*8]); assert p.shape==(3,8) and u.shape==(3,8)
def test_forecast_diffuses_across_neighbors():
 net=Network(); history=[[0.0]*8,[0.0]*8]; history[-1][0]=5.0  # spike on edge 0; edge 1 is its mutual neighbor
 pred_diffuse,_=GraphForecaster(3,network=net,diffusion=0.8).predict(history)
 pred_isolated,_=GraphForecaster(3,network=net,diffusion=0.0).predict(history)
 assert pred_diffuse[0,1]>pred_isolated[0,1]
def test_all_controllers_run():
 r=evaluate(Config(episode_steps=8),episodes=1); assert len(r)==5; assert all("throughput" in x for x in r.values())

