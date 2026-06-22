from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import gym
from pantheonrl.common.agents import StaticPolicyAgent
from stable_baselines3 import PPO

from . import register_envs
from .config import env_config_from_config, load_config, run_dir_from_config, save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an ego PPO policy against multiple fixed partners.")
    parser.add_argument("--config", required=True, help="Path to a JSON config file.")
    parser.add_argument("--partner-run-dir", action="append", help="Run directory containing models/alt.zip.")
    parser.add_argument("--run-name", help="Override config run_name.")
    parser.add_argument("--timesteps", type=int, help="Override total_timesteps.")
    parser.add_argument("--seed", type=int, help="Override random seed.")
    parser.add_argument("--device", help="Override torch device.")
    return parser


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    if args.partner_run_dir:
        config["partner_run_dirs"] = args.partner_run_dir
    if args.run_name:
        config["run_name"] = args.run_name
    if args.timesteps is not None:
        config["total_timesteps"] = args.timesteps
    if args.seed is not None:
        config["seed"] = args.seed
    if args.device:
        config["device"] = args.device
    return config


def partner_model_path(run_dir: str | Path) -> Path:
    path = Path(run_dir) / "models" / "alt.zip"
    if not path.exists():
        raise FileNotFoundError(f"Missing partner model: {path}")
    return path


def train_diverse(config: dict) -> Path:
    register_envs()
    partner_run_dirs = [Path(p) for p in config.get("partner_run_dirs", [])]
    if not partner_run_dirs:
        raise ValueError("partner_run_dirs must contain at least one fixed partner run.")

    run_dir = run_dir_from_config(config)
    model_dir = run_dir / "models"
    tensorboard_dir = run_dir / "tensorboard"
    metrics_dir = run_dir / "metrics"
    model_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "config.resolved.json", config)

    env = gym.make(config["env_id"], **env_config_from_config(config))
    partner_paths = []
    for partner_run_dir in partner_run_dirs:
        model_path = partner_model_path(partner_run_dir)
        partner_paths.append(str(model_path))
        partner_model = PPO.load(str(model_path))
        env.add_partner_agent(StaticPolicyAgent(partner_model.policy))

    ego_kwargs = dict(config["ego_config"])
    ego_kwargs.setdefault("verbose", 1)
    ego = PPO(
        policy="MlpPolicy",
        env=env,
        device=config["device"],
        seed=int(config["seed"]),
        tensorboard_log=str(tensorboard_dir),
        **ego_kwargs,
    )

    started = time.time()
    ego.learn(
        total_timesteps=int(config["total_timesteps"]),
        tb_log_name=f"{config['run_name']}_ego",
    )
    elapsed = time.time() - started

    ego_path = model_dir / "ego"
    ego.save(str(ego_path))
    # Keep one default partner for tools that expect models/alt.zip.
    shutil.copyfile(partner_model_path(partner_run_dirs[0]), model_dir / "alt.zip")

    summary = {
        "run_name": config["run_name"],
        "layout_name": config["layout_name"],
        "layout_names": list(config.get("layout_names", [config["layout_name"]])),
        "total_timesteps": int(config["total_timesteps"]),
        "seed": int(config["seed"]),
        "elapsed_seconds": elapsed,
        "ego_model": str(ego_path) + ".zip",
        "default_alt_model": str(model_dir / "alt.zip"),
        "partner_model_paths": partner_paths,
    }
    save_json(metrics_dir / "train_summary.json", summary)
    print(f"Saved diverse ego model to {ego_path}.zip")
    print(f"Training elapsed seconds: {elapsed:.2f}")
    return run_dir


def main() -> None:
    args = build_parser().parse_args()
    config = apply_overrides(load_config(args.config), args)
    train_diverse(config)


if __name__ == "__main__":
    main()
