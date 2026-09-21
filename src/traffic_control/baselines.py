class Controller:
    def action(self, obs, forecast=None, uncertainty=None): raise NotImplementedError
class FixedTime(Controller):
    def action(self, obs, *args, **kwargs): return [0]*4
class Actuated(Controller):
    def action(self, obs, *args, **kwargs): return [int(obs[i*2+1] > obs[i*2]) for i in range(4)]
class ForecastAware(Actuated):
    def __init__(self, uncertainty=False): self.use_uncertainty=uncertainty
    def action(self, obs, forecast=None, uncertainty=None):
        if forecast is not None: obs=obs+forecast[-1]*(.35 if not self.use_uncertainty else 1/(1+uncertainty[-1]))
        return super().action(obs)
