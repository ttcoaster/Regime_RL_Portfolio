
"""PPO action and observation adapter for FinRL portfolio environments."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PPOPortfolioAdapter(gym.Wrapper):
    """Convert PPO actions to portfolio weights and flatten observations."""

    def __init__(self, env, logit_scale=5.0):
        super().__init__(env)

        if not (np.isfinite(logit_scale) and logit_scale > 0):
            raise ValueError("logit_scale must be finite and positive")

        self.logit_scale = float(logit_scale)

        n = int(np.prod(env.observation_space.shape))

        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(n,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            -1.0,
            1.0,
            shape=env.action_space.shape,
            dtype=np.float32,
        )

    def action_to_weights(self, action):
        a = np.asarray(action, dtype=np.float64)

        if a.shape != self.action_space.shape or not np.all(np.isfinite(a)):
            raise ValueError("Action shape or finite check failed")

        if np.any(a < -1.000001) or np.any(a > 1.000001):
            raise ValueError("Action must be within [-1, 1]")

        logits = self.logit_scale * np.clip(a, -1.0, 1.0)
        logits -= np.max(logits)

        exp_logits = np.exp(logits)
        weights = (exp_logits / exp_logits.sum()).astype(np.float32)

        return weights

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)

        return np.asarray(obs, dtype=np.float32).reshape(-1), info

    def step(self, action):
        weights = self.action_to_weights(action)

        obs, reward, terminated, truncated, info = self.env.step(weights)

        info = dict(info)
        info["target_weights"] = weights.copy()

        return (
            np.asarray(obs, dtype=np.float32).reshape(-1),
            reward,
            terminated,
            truncated,
            info,
        )

