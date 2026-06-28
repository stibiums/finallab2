from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import gym
from stable_baselines3 import PPO
from pantheonrl.common.util import action_from_policy, clip_actions

from . import register_envs
from .config import env_config_from_config, load_json, save_json
from .trace_episode import action_text, state_summary


def model_path(run_dir: Path, name: str) -> Path:
    path = run_dir / "models" / f"{name}.zip"
    if not path.exists():
        raise FileNotFoundError(f"Missing model: {path}")
    return path


def held_object_name(player) -> str | None:
    obj = getattr(player, "held_object", None)
    if obj is None:
        return None
    return str(getattr(obj, "name", ""))


def has_held_soup(state) -> bool:
    return any(held_object_name(player) == "soup" for player in state.players)


def has_ready_soup_object(state) -> bool:
    for obj in state.objects.values():
        if str(getattr(obj, "name", "")) != "soup":
            continue
        obj_state = getattr(obj, "state", None)
        if not isinstance(obj_state, tuple) or len(obj_state) < 3:
            return True
        _, num_items, cook_time = obj_state
        if int(num_items) >= 3 and int(cook_time) <= 0:
            return True
    return False


def select_route(state, switch_mode: str) -> str:
    if switch_mode == "held_soup":
        return "delivery" if has_held_soup(state) else "base"
    if switch_mode == "held_or_ready_soup":
        return "delivery" if has_held_soup(state) or has_ready_soup_object(state) else "base"
    raise ValueError(f"Unsupported switch_mode: {switch_mode}")


def current_observations(env) -> tuple[Any, Any]:
    state = env.unwrapped.base_env.state
    ob_p0, ob_p1 = env.unwrapped.featurize_fn(state)
    if env.unwrapped.ego_agent_idx == 0:
        return ob_p0, ob_p1
    return ob_p1, ob_p0


def partner_policy_action(model: PPO, obs) -> int:
    actions, _, _ = action_from_policy(obs, model.policy)
    return int(clip_actions(actions, model.policy)[0])


def run_subtask_router_episode(
    env,
    base_ego: PPO,
    base_alt: PPO,
    delivery_ego: PPO,
    delivery_alt: PPO,
    deterministic: bool,
    switch_mode: str,
    max_steps: int,
    record_trace: bool = False,
) -> dict[str, Any]:
    env.unwrapped.multi_reset()
    initial_state = state_summary(env)
    route_counts: Counter[str] = Counter()
    joint_action_counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    total_reward = 0.0
    sparse_reward = 0.0
    shaped_reward = 0.0
    done = False
    steps = 0

    while not done and steps < max_steps:
        before = state_summary(env) if record_trace else None
        state = env.unwrapped.base_env.state
        route = select_route(state, switch_mode)
        ego_obs, alt_obs = current_observations(env)
        if route == "delivery":
            ego_model, alt_model = delivery_ego, delivery_alt
        else:
            ego_model, alt_model = base_ego, base_alt
        ego_action, _ = ego_model.predict(ego_obs, deterministic=deterministic)
        alt_action = partner_policy_action(alt_model, alt_obs)

        _, rewards, done, info = env.unwrapped.multi_step(int(ego_action), alt_action)
        route_counts[route] += 1

        joint_action = [int(ego_action), int(alt_action)]
        joint_action_text = [action_text(action) for action in joint_action]
        joint_action_counts[" | ".join(joint_action_text)] += 1

        row_sparse_reward = float(info.get("sparse_reward", 0.0))
        row_shaped_reward = float(info.get("shaped_reward", 0.0))
        row_reward = float(rewards[0])
        total_reward += row_reward
        sparse_reward += row_sparse_reward
        shaped_reward += row_shaped_reward

        if record_trace:
            rows.append(
                {
                    "step": steps,
                    "route": route,
                    "joint_action": joint_action,
                    "joint_action_text": joint_action_text,
                    "reward": row_reward,
                    "sparse_reward": row_sparse_reward,
                    "shaped_reward": row_shaped_reward,
                    "done": bool(done),
                    "before": before,
                    "after": state_summary(env),
                }
            )
        steps += 1

    result = {
        "episode_reward": total_reward,
        "sparse_reward": sparse_reward,
        "shaped_reward": shaped_reward,
        "soups_delivered": sparse_reward / 20.0,
        "steps": steps,
        "base_steps": int(route_counts.get("base", 0)),
        "delivery_steps": int(route_counts.get("delivery", 0)),
    }
    if record_trace:
        result.update(
            {
                "initial_state": initial_state,
                "final_state": rows[-1]["after"] if rows else initial_state,
                "route_counts": dict(route_counts),
                "joint_action_counts": dict(joint_action_counts),
                "steps_detail": rows,
            }
        )
    return result


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    rewards = [float(row["episode_reward"]) for row in rows]
    sparse_rewards = [float(row["sparse_reward"]) for row in rows]
    soups = [float(row["soups_delivered"]) for row in rows]
    base_steps = [float(row["base_steps"]) for row in rows]
    delivery_steps = [float(row["delivery_steps"]) for row in rows]
    return {
        "mean_episode_reward": mean(rewards),
        "std_episode_reward": pstdev(rewards) if len(rewards) > 1 else 0.0,
        "mean_sparse_reward": mean(sparse_rewards),
        "mean_soups_delivered": mean(soups),
        "mean_base_steps": mean(base_steps),
        "mean_delivery_steps": mean(delivery_steps),
    }


def evaluate_subtask_router(config: dict[str, Any]) -> dict[str, Any]:
    register_envs()
    base_run_dir = Path(str(config["base_run_dir"]))
    delivery_run_dir = Path(str(config["delivery_run_dir"]))
    output_dir = Path(str(config.get("output_dir", Path("outputs/runs") / config["run_name"])))
    metrics_dir = output_dir / "metrics"
    trace_dir = output_dir / "traces"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)

    env_config = load_json(base_run_dir / "config.resolved.json")
    env_config = dict(env_config)
    env_config["layout_name"] = str(config.get("layout_name", env_config["layout_name"]))
    env_config["layout_names"] = [env_config["layout_name"]]
    env_config.pop("layout_sampling_weights", None)
    if bool(config.get("standard_start", True)):
        env_config.pop("start_state_mode", None)
    elif config.get("start_state_mode"):
        env_config["start_state_mode"] = str(config["start_state_mode"])
    if config.get("horizon") is not None:
        env_config["horizon"] = int(config["horizon"])

    base_ego = PPO.load(str(model_path(base_run_dir, "ego")))
    base_alt = PPO.load(str(model_path(base_run_dir, "alt")))
    delivery_ego = PPO.load(str(model_path(delivery_run_dir, "ego")))
    delivery_alt = PPO.load(str(model_path(delivery_run_dir, "alt")))
    env = gym.make(env_config["env_id"], **env_config_from_config(env_config))

    episodes = int(config.get("episodes", 20))
    deterministic = bool(config.get("deterministic_eval", True))
    switch_mode = str(config.get("switch_mode", "held_soup"))
    max_steps = int(config.get("max_steps", env_config.get("horizon", 400)))
    output_name = str(config.get("output_name", "subtask_router_eval"))

    episode_rows = []
    first_trace = None
    for episode_idx in range(episodes):
        row = run_subtask_router_episode(
            env=env,
            base_ego=base_ego,
            base_alt=base_alt,
            delivery_ego=delivery_ego,
            delivery_alt=delivery_alt,
            deterministic=deterministic,
            switch_mode=switch_mode,
            max_steps=max_steps,
            record_trace=episode_idx == 0,
        )
        if episode_idx == 0:
            first_trace = dict(row)
            first_trace["episode"] = episode_idx
            trace_path = trace_dir / f"{output_name}_episode0.json"
            save_json(trace_path, first_trace)
        episode_rows.append({"episode": episode_idx, **{k: v for k, v in row.items() if k != "steps_detail"}})

    env.close()

    summary = summarize(episode_rows)
    summary.update(
        {
            "run_name": str(config["run_name"]),
            "base_run_dir": str(base_run_dir),
            "delivery_run_dir": str(delivery_run_dir),
            "layout_name": env_config["layout_name"],
            "start_state_mode": env_config.get("start_state_mode", "standard"),
            "horizon": int(env_config["horizon"]),
            "episodes": episodes,
            "deterministic": deterministic,
            "switch_mode": switch_mode,
            "output_dir": str(output_dir),
        }
    )

    summary_path = metrics_dir / f"{output_name}.json"
    episode_path = metrics_dir / f"{output_name}_episodes.csv"
    with episode_path.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "episode",
            "episode_reward",
            "sparse_reward",
            "shaped_reward",
            "soups_delivered",
            "steps",
            "base_steps",
            "delivery_steps",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in episode_rows)
    summary["episode_csv"] = str(episode_path)
    if first_trace is not None:
        summary["episode0_trace"] = str(trace_dir / f"{output_name}_episode0.json")
    save_json(summary_path, summary)
    save_json(output_dir / "subtask_router_config.resolved.json", {**config, "env_config": env_config})

    print(f"Subtask router summary: {summary}", flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a state-based small_corridor subtask router.")
    parser.add_argument("--config", required=True, help="JSON config for the subtask-router evaluation.")
    parser.add_argument("--base-run-dir", help="Override base/full-chain run directory.")
    parser.add_argument("--delivery-run-dir", help="Override delivery-specialist run directory.")
    parser.add_argument("--output-dir", help="Override output directory.")
    parser.add_argument("--run-name", help="Override run name.")
    parser.add_argument("--switch-mode", choices=["held_soup", "held_or_ready_soup"])
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--stochastic", action="store_true")
    return parser


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = dict(config)
    for key in ["base_run_dir", "delivery_run_dir", "output_dir", "run_name", "switch_mode"]:
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.episodes is not None:
        config["episodes"] = int(args.episodes)
    if args.horizon is not None:
        config["horizon"] = int(args.horizon)
        config["max_steps"] = int(args.horizon)
    if args.stochastic:
        config["deterministic_eval"] = False
    return config


def main() -> None:
    args = build_parser().parse_args()
    config = apply_overrides(load_json(args.config), args)
    evaluate_subtask_router(config)


if __name__ == "__main__":
    main()
