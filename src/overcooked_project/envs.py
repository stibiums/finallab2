from __future__ import annotations

from dataclasses import dataclass

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


@dataclass
class LayoutRuntime:
    layout_name: str
    mdp: OvercookedGridworld
    base_env: OvercookedEnv
    featurize_fn: object


class ConfigurableOvercookedMultiEnv(SimultaneousEnv):
    """PantheonRL-compatible Overcooked env with configurable reward shaping."""

    def __init__(
        self,
        layout_name: str,
        layout_names: list[str] | None = None,
        layout_sampling_weights: list[float] | None = None,
        ego_agent_idx: int = 0,
        baselines: bool = False,
        horizon: int = 400,
        reward_shaping: dict[str, float] | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.layout_names = list(layout_names) if layout_names else [layout_name]
        if not self.layout_names:
            raise ValueError("At least one layout must be configured.")
        self.layout_sampling_weights = self._normalize_layout_weights(layout_sampling_weights)
        self.layout_name = self.layout_names[0]
        self.horizon = horizon
        self.reward_shaping = dict(DEFAULT_REWARD_SHAPING)
        if reward_shaping is not None:
            self.reward_shaping.update(reward_shaping)

        self.rng = np.random.default_rng(seed)
        self.layouts = {name: self._build_layout_runtime(name) for name in self.layout_names}
        self.active_layout = self.layouts[self.layout_name]
        self.mdp = self.active_layout.mdp
        self.base_env = self.active_layout.base_env
        self.featurize_fn = self.active_layout.featurize_fn

        if baselines:
            np.random.seed(0)

        self.observation_space = self._setup_observation_space()
        self._validate_observation_shapes()
        self.lA = len(Action.ALL_ACTIONS)
        self.action_space = gym.spaces.Discrete(self.lA)
        self.ego_agent_idx = ego_agent_idx
        self.multi_reset()

    def _normalize_layout_weights(self, weights: list[float] | None):
        if weights is None:
            return None
        if len(weights) != len(self.layout_names):
            raise ValueError("layout_sampling_weights must match layout_names length.")
        weights_array = np.asarray(weights, dtype=np.float64)
        if np.any(weights_array < 0):
            raise ValueError("layout_sampling_weights cannot contain negative values.")
        total = weights_array.sum()
        if total <= 0:
            raise ValueError("layout_sampling_weights must have a positive sum.")
        return weights_array / total

    def _build_layout_runtime(self, layout_name: str) -> LayoutRuntime:
        mdp = OvercookedGridworld.from_layout_name(
            layout_name=layout_name,
            rew_shaping_params=self.reward_shaping,
        )
        mlp = MediumLevelPlanner.from_pickle_or_compute(
            mdp,
            NO_COUNTERS_PARAMS,
            force_compute=False,
        )
        base_env = OvercookedEnv(mdp, horizon=self.horizon)
        featurize_fn = lambda state, mdp=mdp, mlp=mlp: mdp.featurize_state(state, mlp)
        return LayoutRuntime(
            layout_name=layout_name,
            mdp=mdp,
            base_env=base_env,
            featurize_fn=featurize_fn,
        )

    def _setup_observation_space(self):
        dummy_state = self.mdp.get_standard_start_state()
        obs_shape = self.featurize_fn(dummy_state)[0].shape
        high = np.ones(obs_shape, dtype=np.float32) * np.inf
        return gym.spaces.Box(-high, high, dtype=np.float64)

    def _validate_observation_shapes(self) -> None:
        expected_shape = self.observation_space.shape
        for layout in self.layouts.values():
            obs_shape = layout.featurize_fn(layout.mdp.get_standard_start_state())[0].shape
            if obs_shape != expected_shape:
                raise ValueError(
                    f"Layout {layout.layout_name} observation shape {obs_shape} "
                    f"does not match expected shape {expected_shape}."
                )

    def _activate_layout(self, layout_name: str) -> None:
        self.active_layout = self.layouts[layout_name]
        self.layout_name = layout_name
        self.mdp = self.active_layout.mdp
        self.base_env = self.active_layout.base_env
        self.featurize_fn = self.active_layout.featurize_fn

    def _sample_layout_name(self) -> str:
        if len(self.layout_names) == 1:
            return self.layout_names[0]
        return str(self.rng.choice(self.layout_names, p=self.layout_sampling_weights))

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
        step_info["layout_name"] = self.layout_name
        return (ego_obs, alt_obs), (reward, reward), done, step_info

    def multi_reset(self):
        self._activate_layout(self._sample_layout_name())
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
