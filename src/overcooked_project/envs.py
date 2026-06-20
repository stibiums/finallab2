from __future__ import annotations

import gym
import numpy as np
from gym.envs.registration import register, registry
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.planning.planners import MediumLevelPlanner, NO_COUNTERS_PARAMS
from pantheonrl.common.multiagentenv import SimultaneousEnv

from .config import DEFAULT_REWARD_SHAPING


ENV_ID = "OvercookedShapedMultiEnv-v0"


class ConfigurableOvercookedMultiEnv(SimultaneousEnv):
    """PantheonRL-compatible Overcooked env with configurable reward shaping."""

    def __init__(
        self,
        layout_name: str,
        ego_agent_idx: int = 0,
        baselines: bool = False,
        horizon: int = 400,
        reward_shaping: dict[str, float] | None = None,
    ):
        super().__init__()
        self.layout_name = layout_name
        self.horizon = horizon
        self.reward_shaping = dict(DEFAULT_REWARD_SHAPING)
        if reward_shaping is not None:
            self.reward_shaping.update(reward_shaping)

        self.mdp = OvercookedGridworld.from_layout_name(
            layout_name=layout_name,
            rew_shaping_params=self.reward_shaping,
        )
        mlp = MediumLevelPlanner.from_pickle_or_compute(
            self.mdp,
            NO_COUNTERS_PARAMS,
            force_compute=False,
        )

        self.base_env = OvercookedEnv(self.mdp, horizon=horizon)
        self.featurize_fn = lambda state: self.mdp.featurize_state(state, mlp)

        if baselines:
            np.random.seed(0)

        self.observation_space = self._setup_observation_space()
        self.lA = len(Action.ALL_ACTIONS)
        self.action_space = gym.spaces.Discrete(self.lA)
        self.ego_agent_idx = ego_agent_idx
        self.multi_reset()

    def _setup_observation_space(self):
        dummy_state = self.mdp.get_standard_start_state()
        obs_shape = self.featurize_fn(dummy_state)[0].shape
        high = np.ones(obs_shape, dtype=np.float32) * np.inf
        return gym.spaces.Box(-high, high, dtype=np.float64)

    def multi_step(self, ego_action, alt_action):
        ego_action_obj = Action.INDEX_TO_ACTION[int(ego_action)]
        alt_action_obj = Action.INDEX_TO_ACTION[int(alt_action)]
        if self.ego_agent_idx == 0:
            joint_action = (ego_action_obj, alt_action_obj)
        else:
            joint_action = (alt_action_obj, ego_action_obj)

        next_state, sparse_reward, done, info = self.base_env.step(joint_action)
        shaped_reward = info["shaped_r"]
        reward = sparse_reward + shaped_reward

        ob_p0, ob_p1 = self.featurize_fn(next_state)
        if self.ego_agent_idx == 0:
            ego_obs, alt_obs = ob_p0, ob_p1
        else:
            ego_obs, alt_obs = ob_p1, ob_p0

        step_info = dict(info)
        step_info["sparse_reward"] = float(sparse_reward)
        step_info["shaped_reward"] = float(shaped_reward)
        step_info["joint_action"] = [int(ego_action), int(alt_action)]
        return (ego_obs, alt_obs), (reward, reward), done, step_info

    def multi_reset(self):
        self.base_env.reset()
        ob_p0, ob_p1 = self.featurize_fn(self.base_env.state)
        if self.ego_agent_idx == 0:
            ego_obs, alt_obs = ob_p0, ob_p1
        else:
            ego_obs, alt_obs = ob_p1, ob_p0
        return ego_obs, alt_obs

    def render(self, mode="human", close=False):
        state_string = self.base_env.mdp.state_string(self.base_env.state)
        if mode == "ansi":
            return state_string
        print(state_string)
        return None


def register_envs() -> None:
    if ENV_ID not in registry.env_specs:
        register(
            id=ENV_ID,
            entry_point="overcooked_project.envs:ConfigurableOvercookedMultiEnv",
        )
