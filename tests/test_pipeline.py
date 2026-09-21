from traffic_control.config import Config
from traffic_control.forecast import GraphForecaster
from traffic_control.evaluate import evaluate
def test_forecaster_shapes():
 p,u=GraphForecaster(3).predict([[1]*8,[2]*8]); assert p.shape==(3,8) and u.shape==(3,8)
def test_all_controllers_run():
 r=evaluate(Config(episode_steps=8),episodes=1); assert len(r)==5; assert all("throughput" in x for x in r.values())

