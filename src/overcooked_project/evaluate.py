from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, pstdev

import gym
from pantheonrl.common.agents import StaticPolicyAgent
from stable_baselines3 import PPO

from . import register_envs
from .config import env_config_from_config, load_json, save_json


def run_episode(env, ego_model: PPO, deterministic: bool) -> dict:
    obs = env.reset()
    done = False
    total_reward = 0.0
    sparse_reward = 0.0
    shaped_reward = 0.0
    steps = 0

    while not done:
        action, _ = ego_model.predict(obs, deterministic=deterministic)
        obs, reward, done, info = env.step(action)
        total_reward += float(reward)
        sparse_reward += float(info.get("sparse_reward", 0.0))
        shaped_reward += float(info.get("shaped_reward", 0.0))
        steps += 1

    return {
        "episode_reward": total_reward,
        "sparse_reward": sparse_reward,
        "shaped_reward": shaped_reward,
        "soups_delivered": sparse_reward / 20.0,
        "steps": steps,
    }


def evaluate_models(
    run_dir: str | Path,
    episodes: int = 20,
    deterministic: bool = True,
    output_name: str = "eval_metrics",
    layout: str | None = None,
    start_state_mode: str | None = None,
    standard_start: bool = False,
) -> dict:
    register_envs()
    run_dir = Path(run_dir)
    config = load_json(run_dir / "config.resolved.json")
    if layout is not None:
        config = dict(config)
        config["layout_name"] = layout
        config["layout_names"] = [layout]
        config.pop("layout_sampling_weights", None)
    if standard_start:
        config = dict(config)
        config.pop("start_state_mode", None)
    elif start_state_mode is not None:
        config = dict(config)
        config["start_state_mode"] = start_state_mode
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(config["env_id"], **env_config_from_config(config))
    ego_model = PPO.load(str(run_dir / "models" / "ego"))
    alt_model = PPO.load(str(run_dir / "models" / "alt"))
    env.add_partner_agent(StaticPolicyAgent(alt_model.policy))

    rows = [run_episode(env, ego_model, deterministic) for _ in range(episodes)]
    env.close()

    csv_path = metrics_dir / f"{output_name}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    rewards = [row["episode_reward"] for row in rows]
    sparse_rewards = [row["sparse_reward"] for row in rows]
    soups = [row["soups_delivered"] for row in rows]
    summary = {
        "episodes": episodes,
        "deterministic": deterministic,
        "layout_name": config["layout_name"],
        "start_state_mode": config.get("start_state_mode", "standard"),
        "mean_episode_reward": mean(rewards),
        "std_episode_reward": pstdev(rewards) if len(rewards) > 1 else 0.0,
        "mean_sparse_reward": mean(sparse_rewards),
        "mean_soups_delivered": mean(soups),
        "csv_path": str(csv_path),
    }
    save_json(metrics_dir / f"{output_name}.json", summary)
    print(f"Evaluation summary: {summary}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained Overcooked run.")
    parser.add_argument("--run-dir", required=True, help="Run directory under outputs/runs.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy actions.")
    parser.add_argument("--output-name", default="eval_metrics")
    parser.add_argument("--layout", help="Force a single layout for evaluation.")
    parser.add_argument("--start-state-mode", help="Override config start_state_mode for evaluation.")
    parser.add_argument(
        "--standard-start",
        action="store_true",
        help="Remove any configured start_state_mode for standard-start evaluation.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    evaluate_models(
        args.run_dir,
        episodes=args.episodes,
        deterministic=not args.stochastic,
        output_name=args.output_name,
        layout=args.layout,
        start_state_mode=args.start_state_mode,
        standard_start=args.standard_start,
    )


if __name__ == "__main__":
    main()
