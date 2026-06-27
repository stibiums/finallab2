from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import gym
import numpy as np
from gym.envs.registration import register, registry
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import (
    ObjectState,
    OvercookedGridworld,
    OvercookedState,
    PlayerState,
)
from overcooked_ai_py.planning.planners import MediumLevelPlanner, NO_COUNTERS_PARAMS
from pantheonrl.common.multiagentenv import SimultaneousEnv

from .config import DEFAULT_REWARD_SHAPING


ENV_ID = "OvercookedShapedMultiEnv-v0"
MOVE_DIRECTIONS = ((0, -1), (0, 1), (1, 0), (-1, 0))


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
        event_reward_shaping: dict[str, object] | None = None,
        start_state_mode: str | None = None,
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
        self.event_reward_shaping = dict(event_reward_shaping or {})
        self.start_state_mode = start_state_mode
        self._distance_maps: dict[tuple[str, tuple[str, ...]], dict[tuple[int, int], int]] = {}
        self._interaction_goals: dict[
            tuple[str, tuple[str, ...]], dict[tuple[int, int], set[tuple[int, int]]]
        ] = {}
        self.cumulative_event_shaped_reward = 0.0

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
        base_env = OvercookedEnv(
            mdp,
            start_state_fn=self._start_state_fn_for_layout(layout_name),
            horizon=self.horizon,
        )
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

        prev_state = self.base_env.state.deepcopy()
        next_state, sparse_reward, done, info = self.base_env.step(joint_action)
        base_shaped_reward = float(info["shaped_r"])
        event_shaped_reward = self._compute_event_shaped_reward(prev_state, next_state)
        self.cumulative_event_shaped_reward += event_shaped_reward
        shaped_reward = base_shaped_reward + event_shaped_reward
        reward = sparse_reward + shaped_reward

        ob_p0, ob_p1 = self.featurize_fn(next_state)
        if self.ego_agent_idx == 0:
            ego_obs, alt_obs = ob_p0, ob_p1
        else:
            ego_obs, alt_obs = ob_p1, ob_p0

        step_info = dict(info)
        step_info["base_shaped_reward"] = base_shaped_reward
        step_info["event_shaped_reward"] = float(event_shaped_reward)
        step_info["shaped_r"] = float(shaped_reward)
        step_info["sparse_reward"] = float(sparse_reward)
        step_info["shaped_reward"] = float(shaped_reward)
        step_info["joint_action"] = [int(ego_action), int(alt_action)]
        step_info["layout_name"] = self.layout_name
        if "episode" in step_info:
            step_info["episode"] = dict(step_info["episode"])
            step_info["episode"]["ep_base_shaped_r"] = float(
                step_info["episode"].get("ep_shaped_r", 0.0)
            )
            step_info["episode"]["ep_event_shaped_r"] = float(
                self.cumulative_event_shaped_reward
            )
            step_info["episode"]["ep_shaped_r"] = (
                step_info["episode"]["ep_base_shaped_r"]
                + step_info["episode"]["ep_event_shaped_r"]
            )
        return (ego_obs, alt_obs), (reward, reward), done, step_info

    def multi_reset(self):
        self._activate_layout(self._sample_layout_name())
        self.base_env.reset()
        self.cumulative_event_shaped_reward = 0.0
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

    def _start_state_fn_for_layout(self, layout_name: str):
        if self.start_state_mode in (None, "", "standard"):
            return None
        if self.start_state_mode == "small_corridor_delivery":
            if layout_name != "small_corridor":
                raise ValueError(
                    "start_state_mode='small_corridor_delivery' only supports "
                    "layout_name='small_corridor'."
                )
            return self._small_corridor_delivery_start_state
        raise ValueError(f"Unsupported start_state_mode: {self.start_state_mode}")

    def _small_corridor_delivery_start_state(self) -> OvercookedState:
        holder_idx = int(self.rng.integers(0, 2))
        holder_positions = [(x, 3) for x in range(1, 12)]
        holder_pos = holder_positions[int(self.rng.integers(0, len(holder_positions)))]
        other_pos = (10, 1) if holder_idx == 0 else (5, 1)

        players = []
        for player_idx in range(2):
            if player_idx == holder_idx:
                soup = ObjectState("soup", holder_pos, ("onion", 3, 20))
                players.append(PlayerState(holder_pos, Direction.WEST, soup))
            else:
                players.append(PlayerState(other_pos, Direction.WEST))
        return OvercookedState(players, {}, order_list=None)

    def _compute_event_shaped_reward(self, prev_state, next_state) -> float:
        if not self.event_reward_shaping:
            return 0.0

        reward = self._object_transition_reward(prev_state, next_state)

        progress_scale = float(
            self.event_reward_shaping.get("DISTANCE_PROGRESS_REWARD", 0.0)
        )
        skip_progress = bool(
            self.event_reward_shaping.get("IGNORE_PROGRESS_ON_HELD_OBJECT_CHANGE", False)
        ) and self._held_objects_changed(prev_state, next_state)
        if progress_scale != 0.0 and not skip_progress:
            prev_potential = self._state_subgoal_potential(prev_state)
            next_potential = self._state_subgoal_potential(next_state)
            if prev_potential is not None and next_potential is not None:
                progress = prev_potential - next_potential
                if bool(
                    self.event_reward_shaping.get(
                        "DISTANCE_PROGRESS_POSITIVE_ONLY", False
                    )
                ):
                    progress = max(progress, 0.0)
                reward += progress_scale * progress

        return float(reward)

    def _object_transition_reward(self, prev_state, next_state) -> float:
        reward = 0.0
        for prev_player, next_player in zip(prev_state.players, next_state.players):
            prev_held = self._held_object_name(prev_player)
            next_held = self._held_object_name(next_player)
            if prev_held is not None or next_held is None:
                continue
            if bool(self.event_reward_shaping.get("PICKUP_REWARD_SOURCE_ONLY", False)):
                if self._unowned_object_count(next_state, next_held) < self._unowned_object_count(
                    prev_state, next_held
                ):
                    continue
            if next_held == "onion":
                reward += float(self.event_reward_shaping.get("ONION_PICKUP_REWARD", 0.0))
            elif next_held == "tomato":
                reward += float(self.event_reward_shaping.get("TOMATO_PICKUP_REWARD", 0.0))
            elif next_held == "dish":
                reward += float(self.event_reward_shaping.get("DISH_PICKUP_REWARD", 0.0))

        pot_progress = self._pot_item_count(next_state) - self._pot_item_count(prev_state)
        if pot_progress > 0:
            reward += pot_progress * float(
                self.event_reward_shaping.get("POT_PLACEMENT_REWARD", 0.0)
            )
        return reward

    @staticmethod
    def _held_object_name(player) -> str | None:
        obj = getattr(player, "held_object", None)
        if obj is None:
            return None
        return str(getattr(obj, "name", ""))

    def _held_objects_changed(self, prev_state, next_state) -> bool:
        return any(
            self._held_object_name(prev_player) != self._held_object_name(next_player)
            for prev_player, next_player in zip(prev_state.players, next_state.players)
        )

    @staticmethod
    def _unowned_object_count(state, object_name: str) -> int:
        return sum(
            1
            for obj in state.objects.values()
            if str(getattr(obj, "name", "")) == object_name
        )

    def _pot_item_count(self, state) -> int:
        count = 0
        for position, obj in state.objects.items():
            if self.mdp.get_terrain_type_at_pos(position) != "P":
                continue
            if getattr(obj, "name", None) != "soup":
                continue
            _, num_items, _ = obj.state
            count += int(num_items)
        return count

    def _state_subgoal_potential(self, state) -> float | None:
        distances = []
        for player in state.players:
            distance = self._player_subgoal_distance(player, state)
            if distance is not None:
                distances.append(distance)
        if not distances:
            return None
        return float(sum(distances))

    def _player_subgoal_distance(self, player, state) -> int | None:
        target_terrains = self._target_terrains_for_player(player, state)
        if not target_terrains:
            return None
        if bool(self.event_reward_shaping.get("INCLUDE_ORIENTATION_IN_DISTANCE", False)):
            goals = self._interaction_goals_for_targets(target_terrains)
            orientations = goals.get(tuple(player.position))
            if orientations is not None:
                return 0 if tuple(player.orientation) in orientations else 1
        distances = self._distance_map_for_targets(target_terrains)
        return distances.get(tuple(player.position))

    def _target_terrains_for_player(self, player, state) -> tuple[str, ...]:
        held = self._held_object_name(player)
        if held is None:
            dish_target_min_items = int(
                self.event_reward_shaping.get("DISH_TARGET_MIN_POT_ITEMS", 0)
            )
            if (
                dish_target_min_items > 0
                and self._max_pot_item_count(state) >= dish_target_min_items
            ):
                return self._terrain_tuple("READY_EMPTY_TARGET_TERRAINS", ("D",))
            return self._terrain_tuple("EMPTY_TARGET_TERRAINS", ("O", "D"))
        if held in {"onion", "tomato"}:
            return self._terrain_tuple("HELD_ITEM_TARGET_TERRAINS", ("P",))
        if held == "dish":
            return self._terrain_tuple("HELD_DISH_TARGET_TERRAINS", ("P",))
        if held == "soup":
            return self._terrain_tuple("HELD_SOUP_TARGET_TERRAINS", ("S",))
        return ()

    def _terrain_tuple(self, key: str, default: tuple[str, ...]) -> tuple[str, ...]:
        value = self.event_reward_shaping.get(key, default)
        if isinstance(value, str):
            return (value,)
        return tuple(str(item) for item in value)

    def _max_pot_item_count(self, state) -> int:
        max_count = 0
        for position, obj in state.objects.items():
            if self.mdp.get_terrain_type_at_pos(position) != "P":
                continue
            if getattr(obj, "name", None) != "soup":
                continue
            _, num_items, _ = obj.state
            max_count = max(max_count, int(num_items))
        return max_count

    def _interaction_goals_for_targets(
        self, target_terrains: tuple[str, ...]
    ) -> dict[tuple[int, int], set[tuple[int, int]]]:
        cache_key = (self.layout_name, target_terrains)
        if cache_key in self._interaction_goals:
            return self._interaction_goals[cache_key]

        goals: dict[tuple[int, int], set[tuple[int, int]]] = {}
        open_positions = set(self.mdp.terrain_pos_dict.get(" ", []))
        for terrain in target_terrains:
            for target in self.mdp.terrain_pos_dict.get(terrain, []):
                for dx, dy in MOVE_DIRECTIONS:
                    candidate = (target[0] - dx, target[1] - dy)
                    if candidate in open_positions:
                        goals.setdefault(candidate, set()).add((dx, dy))

        self._interaction_goals[cache_key] = goals
        return goals

    def _distance_map_for_targets(self, target_terrains: tuple[str, ...]):
        cache_key = (self.layout_name, target_terrains)
        if cache_key in self._distance_maps:
            return self._distance_maps[cache_key]

        goal_positions = set(self._interaction_goals_for_targets(target_terrains))
        open_positions = set(self.mdp.terrain_pos_dict.get(" ", []))

        distances: dict[tuple[int, int], int] = {}
        queue = deque()
        for goal in goal_positions:
            distances[goal] = 0
            queue.append(goal)

        while queue:
            position = queue.popleft()
            for dx, dy in MOVE_DIRECTIONS:
                next_position = (position[0] + dx, position[1] + dy)
                if next_position not in open_positions or next_position in distances:
                    continue
                distances[next_position] = distances[position] + 1
                queue.append(next_position)

        self._distance_maps[cache_key] = distances
        return distances


def register_envs() -> None:
    if ENV_ID not in registry.env_specs:
        register(
            id=ENV_ID,
            entry_point="overcooked_project.envs:ConfigurableOvercookedMultiEnv",
        )
