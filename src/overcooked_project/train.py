from __future__ import annotations

import argparse
import time
from pathlib import Path

import gym
from pantheonrl.common.agents import OnPolicyAgent
from stable_baselines3 import PPO

from . import register_envs
from .config import env_config_from_config, load_config, run_dir_from_config, save_json
from .evaluate import evaluate_models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PPO agents on Overcooked.")
    parser.add_argument("--config", required=True, help="Path to a JSON config file.")
    parser.add_argument("--run-name", help="Override config run_name.")
    parser.add_argument("--timesteps", type=int, help="Override total_timesteps.")
    parser.add_argument("--layout", help="Override layout_name.")
    parser.add_argument("--seed", type=int, help="Override random seed.")
    parser.add_argument("--device", help="Override torch device: auto, cpu, cuda, mps.")
    parser.add_argument("--no-eval", action="store_true", help="Skip post-training evaluation.")
    return parser


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    if args.run_name:
        config["run_name"] = args.run_name
    if args.timesteps is not None:
        config["total_timesteps"] = args.timesteps
    if args.layout:
        config["layout_name"] = args.layout
    if args.seed is not None:
        config["seed"] = args.seed
    if args.device:
        config["device"] = args.device
    return config


def make_ppo_kwargs(config: dict, agent_key: str, env, tensorboard_log: Path) -> dict:
    kwargs = dict(config[agent_key])
    kwargs.setdefault("verbose", 1 if agent_key == "ego_config" else 0)
    kwargs["env"] = env
    kwargs["device"] = config["device"]
    kwargs["seed"] = int(config["seed"])
    kwargs["tensorboard_log"] = str(tensorboard_log)
    return kwargs


def train(config: dict) -> Path:
    register_envs()
    run_dir = run_dir_from_config(config)
    model_dir = run_dir / "models"
    tensorboard_dir = run_dir / "tensorboard"
    metrics_dir = run_dir / "metrics"
    model_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    save_json(run_dir / "config.resolved.json", config)

    env = gym.make(config["env_id"], **env_config_from_config(config))
    alt_env = env.getDummyEnv(1)

    ego = PPO(policy="MlpPolicy", **make_ppo_kwargs(config, "ego_config", env, tensorboard_dir))
    alt_model = PPO(
        policy="MlpPolicy",
        **make_ppo_kwargs(config, "alt_config", alt_env, tensorboard_dir),
    )
    partner = OnPolicyAgent(
        alt_model,
        tensorboard_log=str(tensorboard_dir),
        tb_log_name=f"{config['run_name']}_alt",
    )
    env.add_partner_agent(partner)

    started = time.time()
    ego.learn(
        total_timesteps=int(config["total_timesteps"]),
        tb_log_name=f"{config['run_name']}_ego",
    )
    elapsed = time.time() - started

    ego_path = model_dir / "ego"
    alt_path = model_dir / "alt"
    ego.save(str(ego_path))
    alt_model.save(str(alt_path))

    summary = {
        "run_name": config["run_name"],
        "layout_name": config["layout_name"],
        "layout_names": list(config.get("layout_names", [config["layout_name"]])),
        "total_timesteps": int(config["total_timesteps"]),
        "seed": int(config["seed"]),
        "elapsed_seconds": elapsed,
        "ego_model": str(ego_path) + ".zip",
        "alt_model": str(alt_path) + ".zip",
    }
    save_json(metrics_dir / "train_summary.json", summary)
    print(f"Saved models to {model_dir}")
    print(f"Training elapsed seconds: {elapsed:.2f}")
    return run_dir


def main() -> None:
    args = build_parser().parse_args()
    config = apply_overrides(load_config(args.config), args)
    run_dir = train(config)
    if not args.no_eval and int(config.get("eval_episodes", 0)) > 0:
        evaluate_models(run_dir, episodes=int(config["eval_episodes"]))


if __name__ == "__main__":
    main()
