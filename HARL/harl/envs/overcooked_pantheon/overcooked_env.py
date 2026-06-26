import copy
import sys
from pathlib import Path

import gym
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from overcookedgym.overcooked import OvercookedMultiEnv
except ImportError as exc:
    raise ImportError(
        "Failed to import PantheonRL Overcooked environment. "
        "Make sure you run inside the PantheonRL workspace and install the "
        "Overcooked dependencies described in README.md."
    ) from exc


class OvercookedPantheonEnv:
    def __init__(self, args):
        self.args = copy.deepcopy(args)
        self.layout_name = self.args["layout_name"]
        self.ego_agent_idx = self.args.get("ego_agent_idx", 0)
        self.horizon = self.args.get("horizon", 400)
        self.baselines = self.args.get("baselines", False)
        self.n_agents = 2
        self.cur_step = 0
        self._seed = 0

        self.env = OvercookedMultiEnv(
            layout_name=self.layout_name,
            ego_agent_idx=self.ego_agent_idx,
            baselines=self.baselines,
        )

        self.observation_space = [self.env.observation_space for _ in range(self.n_agents)]
        self.action_space = [self.env.action_space for _ in range(self.n_agents)]

        local_obs = self._format_obs(self.env.multi_reset())
        share_obs = self._build_share_obs(local_obs)[0]
        self.share_observation_space = [
            gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=share_obs.shape,
                dtype=np.float32,
            )
            for _ in range(self.n_agents)
        ]

    def step(self, actions):
        """Return local_obs, share_obs, rewards, dones, infos, available_actions."""
        ego_action = self._extract_action(actions[0])
        alt_action = self._extract_action(actions[1])
        obs, rewards, done, _ = self.env.multi_step(ego_action, alt_action)

        self.cur_step += 1
        local_obs = self._format_obs(obs)
        share_obs = self._build_share_obs(local_obs)

        rewards = [[float(reward)] for reward in rewards]
        dones = [bool(done)] * self.n_agents
        info = {
            "layout_name": self.layout_name,
            "step": self.cur_step,
            "episode_reward": float(np.mean(rewards)),
        }
        if done and self.cur_step >= self.horizon:
            info["bad_transition"] = True
        infos = [copy.deepcopy(info) for _ in range(self.n_agents)]
        return local_obs, share_obs, rewards, dones, infos, self.get_avail_actions()

    def reset(self):
        """Return local observations, repeated share observations and action masks."""
        self.cur_step = 0
        local_obs = self._format_obs(self.env.multi_reset())
        share_obs = self._build_share_obs(local_obs)
        return local_obs, share_obs, self.get_avail_actions()

    def get_avail_actions(self):
        return [[1] * self.action_space[agent_id].n for agent_id in range(self.n_agents)]

    def render(self):
        self.env.render()

    def close(self):
        base_env = getattr(self.env, "base_env", None)
        if base_env is not None and hasattr(base_env, "close"):
            base_env.close()

    def seed(self, seed):
        self._seed = seed
        np.random.seed(seed)

    def _format_obs(self, obs):
        return [
            np.asarray(agent_obs, dtype=np.float32).reshape(-1) for agent_obs in obs
        ]

    def _build_share_obs(self, local_obs):
        share_obs = np.concatenate(local_obs, axis=0).astype(np.float32)
        return [share_obs.copy() for _ in range(self.n_agents)]

    @staticmethod
    def _extract_action(action):
        return int(np.asarray(action).reshape(-1)[0])
