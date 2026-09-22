"""Small MAPPO-shaped controller: a shared linear Bernoulli policy per agent
trained with a numpy-only myopic policy gradient (per-step reward advantage
against a running baseline, clipped update). Replace with a torch
actor/critic for long runs."""
import numpy as np


def _sigmoid(x): return 1/(1+np.exp(-np.clip(x,-20,20)))

class MAPPO:
    # current queue diff (b-a), forecast queue diff (b-a; uncertainty-damped when enabled).
    # Deliberately no bias term: the domain is symmetric (no reason to globally prefer
    # clearing "a" over "b"), and a bias shares the same noisy update/clip as the other
    # weights with nothing keeping it smaller -- once it dominates the (tanh-bounded) diff
    # weight, sign(logit) stops depending on the observation at all, and the greedy policy
    # collapses to a constant action regardless of state. Dropping it removes that failure
    # mode structurally instead of just making it less likely.
    FEATURES=2
    AGENTS=4

    def __init__(self, forecast=False, uncertainty=False, seed=7, lr=0.05):
        self.forecast=forecast; self.uncertainty=uncertainty; self.rng=np.random.default_rng(seed)
        self.lr=lr; self.weights=np.zeros(self.FEATURES)
        # Running baseline/scale for the per-step reward, used to normalize the advantage.
        # This is a *continuing* congestion-minimization task with a dense per-step reward,
        # not a sparse-return game, so a discounted return-to-go just encodes a deterministic
        # "steps remaining in the episode" trend into the advantage regardless of the action
        # taken -- that was the actual cause of the policy collapsing to a constant action
        # under longer episodes. A myopic reward-vs-baseline advantage avoids that confound.
        self.reward_mean=0.0; self.reward_var=1.0
        self.training=False; self._reset_buffer()

    def _reset_buffer(self): self.buffer=[]

    def train(self, flag=True): self.training=flag

    def _features(self, obs, forecast, uncertainty):
        # The action that actually matters here is binary -- which of an agent's two queues
        # (a vs b) gets the green light -- so the discriminative signal is the *difference*
        # between them, exactly what the Actuated baseline already keys off
        # (`obs[i*2+1] > obs[i*2]`). Feeding the two raw (or even tanh-squashed) queue levels
        # separately doesn't work: once both queues are large they saturate to ~the same
        # value and the model loses the ability to tell which one is bigger, so gradient
        # noise pushes both weights the same way and the greedy policy collapses to a
        # constant action. Using the difference directly avoids that failure mode and is
        # scale-bounded (tanh) without losing the sign that decides the action.
        obs=np.asarray(obs,dtype=float)
        diff_now=obs[1::2]-obs[0::2]  # per-agent (queue_b - queue_a), length 4
        if self.forecast and forecast is not None:
            diff_fc=forecast[-1][1::2]-forecast[-1][0::2]
            if self.uncertainty and uncertainty is not None:
                damp=1.0/(1.0+uncertainty[-1][1::2]+uncertainty[-1][0::2])
                diff_fc=diff_fc*damp
        else:
            diff_fc=np.zeros(self.AGENTS)
        feats=np.stack([np.tanh(diff_now/10.0),np.tanh(diff_fc/10.0)],axis=1)
        return feats

    def action(self, obs, forecast=None, uncertainty=None):
        feats=self._features(obs,forecast,uncertainty)
        probs=_sigmoid(feats@self.weights)
        if self.training:
            actions=(self.rng.random(self.AGENTS)<probs).astype(int)
            self.buffer.append([feats,actions,probs,None])
        else:
            actions=(probs>=0.5).astype(int)
        return actions.tolist()

    def record_reward(self, reward):
        if self.training and self.buffer: self.buffer[-1][3]=reward

    def update(self):
        if not self.training or not self.buffer:
            self._reset_buffer(); return {"loss":0.0,"episodes":0}
        rewards=np.array([step[3] if step[3] is not None else 0.0 for step in self.buffer])
        advantages=(rewards-self.reward_mean)/np.sqrt(self.reward_var)
        grad=np.zeros(self.FEATURES)
        for (feats,actions,probs,_),adv in zip(self.buffer,advantages):
            grad+=adv*np.mean((actions-probs)[:,None]*feats,axis=0)
        grad=np.clip(grad/len(self.buffer),-2.0,2.0)
        # Hard-bound the weights too: with squashed (-1,1) features this keeps the maximum
        # possible logit magnitude (~FEATURES*bound) far from where sigmoid fully saturates,
        # so the greedy policy can't collapse into always picking the same action regardless
        # of observation.
        self.weights=np.clip(self.weights+self.lr*grad,-4.0,4.0)
        self.reward_mean=0.95*self.reward_mean+0.05*float(rewards.mean())
        self.reward_var=max(1e-3,0.95*self.reward_var+0.05*float(rewards.var()))
        loss=float(-rewards.mean())
        self._reset_buffer()
        return {"loss":loss,"episodes":1}
