from __future__ import annotations

import argparse
import copy
from pathlib import Path

import gym
from pantheonrl.common.agents import StaticPolicyAgent
from stable_baselines3 import PPO

from . import register_envs
from .config import env_config_from_config, load_json, save_json
from .rendering import render_state


def record_demo(
    run_dir: str | Path,
    max_steps: int = 400,
    deterministic: bool = True,
    output_name: str = "demo",
    layout: str | None = None,
) -> dict:
    register_envs()
    run_dir = Path(run_dir)
    config = load_json(run_dir / "config.resolved.json")
    if layout is not None:
        config = dict(config)
        config["layout_name"] = layout
        config["layout_names"] = [layout]
        config.pop("layout_sampling_weights", None)
    demo_dir = run_dir / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(config["env_id"], **env_config_from_config(config))
    ego_model = PPO.load(str(run_dir / "models" / "ego"))
    alt_model = PPO.load(str(run_dir / "models" / "alt"))
    env.add_partner_agent(StaticPolicyAgent(alt_model.policy))

    frames = []
    rewards = []
    sparse_rewards = []
    obs = env.reset()
    frames.append(render_state(env.unwrapped.mdp, copy.deepcopy(env.unwrapped.base_env.state)))

    done = False
    steps = 0
    while not done and steps < max_steps:
        action, _ = ego_model.predict(obs, deterministic=deterministic)
        obs, reward, done, info = env.step(action)
        rewards.append(float(reward))
        sparse_rewards.append(float(info.get("sparse_reward", 0.0)))
        frames.append(render_state(env.unwrapped.mdp, copy.deepcopy(env.unwrapped.base_env.state)))
        steps += 1

    gif_path = demo_dir / f"{output_name}.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=140,
        loop=0,
    )
    summary = {
        "gif_path": str(gif_path),
        "layout_name": env.unwrapped.layout_name,
        "steps": steps,
        "total_reward": sum(rewards),
        "sparse_reward": sum(sparse_rewards),
        "soups_delivered": sum(sparse_rewards) / 20.0,
        "deterministic": deterministic,
    }
    save_json(demo_dir / f"{output_name}.json", summary)
    print(f"Saved demo GIF to {gif_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record a GIF demo from trained models.")
    parser.add_argument("--run-dir", required=True, help="Run directory under outputs/runs.")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy actions.")
    parser.add_argument("--output-name", default="demo")
    parser.add_argument("--layout", help="Force a single layout for demo recording.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    record_demo(
        args.run_dir,
        max_steps=args.max_steps,
        deterministic=not args.stochastic,
        output_name=args.output_name,
        layout=args.layout,
    )


if __name__ == "__main__":
    main()
