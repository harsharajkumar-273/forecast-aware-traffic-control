class Controller:
    def action(self, obs, forecast=None, uncertainty=None): raise NotImplementedError
class FixedTime(Controller):
    def action(self, obs, *args, **kwargs): return [0]*4
class Actuated(Controller):
    def action(self, obs, *args, **kwargs): return [int(obs[i*2+1] > obs[i*2]) for i in range(4)]
