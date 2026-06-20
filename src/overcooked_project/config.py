from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


DEFAULT_REWARD_SHAPING = {
    "PLACEMENT_IN_POT_REW": 3,
    "DISH_PICKUP_REWARD": 3,
    "SOUP_PICKUP_REWARD": 5,
    "DISH_DISP_DISTANCE_REW": 0,
    "POT_DISTANCE_REW": 0,
    "SOUP_DISTANCE_REW": 0,
}


DEFAULT_CONFIG: dict[str, Any] = {
    "run_name": "overcooked_run",
    "env_id": "OvercookedShapedMultiEnv-v0",
    "layout_name": "simple",
    "horizon": 400,
    "seed": 0,
    "device": "auto",
    "total_timesteps": 500000,
    "eval_episodes": 20,
    "output_root": "outputs/runs",
    "reward_shaping": DEFAULT_REWARD_SHAPING,
    "ego_config": {
        "n_steps": 1024,
        "batch_size": 256,
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
        "verbose": 1,
    },
    "alt_config": {
        "n_steps": 1024,
        "batch_size": 256,
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
        "verbose": 0,
    },
}


def deep_update(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def load_config(path: str | Path) -> dict[str, Any]:
    return deep_update(DEFAULT_CONFIG, load_json(path))


def run_dir_from_config(config: Mapping[str, Any]) -> Path:
    return Path(str(config.get("output_root", "outputs/runs"))) / str(config["run_name"])


def env_config_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "layout_name": config["layout_name"],
        "horizon": int(config["horizon"]),
        "reward_shaping": dict(config["reward_shaping"]),
    }
