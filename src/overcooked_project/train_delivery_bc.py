from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import gym
import numpy as np
import torch
from stable_baselines3 import PPO
from torch.utils.data import DataLoader, TensorDataset

from . import register_envs
from .config import env_config_from_config, load_config, load_json, save_json


def run_dir_from_name(run_name: str) -> Path:
    return Path("outputs") / "runs" / run_name


def load_bc_dataset(demo_path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(demo_path).read_text(encoding="utf-8"))
    ego_obs, alt_obs, ego_actions, alt_actions = [], [], [], []
    action_counts = Counter()
    for demo in data["demos"]:
        if not demo.get("success", False):
            continue
        for row in demo["steps_detail"]:
            observations = row.get("player_observations")
            if observations is None:
                raise ValueError(
                    "Demo file does not contain player_observations. "
                    "Regenerate it with scripts/collect_delivery_demos.sh."
                )
            ego_obs.append(observations[0])
            alt_obs.append(observations[1])
            ego_actions.append(int(row["joint_action"][0]))
            alt_actions.append(int(row["joint_action"][1]))
            action_counts[str(row["joint_action_text"][0])] += 1
            action_counts[str(row["joint_action_text"][1])] += 1

    if not ego_obs:
        raise ValueError("No successful demonstration steps found.")

    return {
        "ego_obs": np.asarray(ego_obs, dtype=np.float32),
        "alt_obs": np.asarray(alt_obs, dtype=np.float32),
        "ego_actions": np.asarray(ego_actions, dtype=np.int64),
        "alt_actions": np.asarray(alt_actions, dtype=np.int64),
        "action_counts": dict(action_counts),
        "num_steps": len(ego_actions),
        "num_successful_episodes": int(data.get("successes", 0)),
        "source_summary": {
            key: data.get(key)
            for key in [
                "config",
                "mode",
                "episodes",
                "max_steps",
                "successes",
                "success_rate",
                "mean_success_steps",
            ]
        },
    }


def split_dataset(
    obs: np.ndarray,
    actions: np.ndarray,
    seed: int,
    val_fraction: float,
) -> tuple[TensorDataset, TensorDataset]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(actions))
    rng.shuffle(indices)
    val_size = max(1, int(round(len(indices) * val_fraction)))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    if len(train_indices) == 0:
        train_indices = val_indices

    train = TensorDataset(
        torch.as_tensor(obs[train_indices], dtype=torch.float32),
        torch.as_tensor(actions[train_indices], dtype=torch.long),
    )
    val = TensorDataset(
        torch.as_tensor(obs[val_indices], dtype=torch.float32),
        torch.as_tensor(actions[val_indices], dtype=torch.long),
    )
    return train, val


def evaluate_dataset(model: PPO, dataset: TensorDataset, batch_size: int) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.policy.set_training_mode(False)
    losses = []
    correct = 0
    total = 0
    with torch.no_grad():
        for obs, actions in loader:
            obs = obs.to(model.device)
            actions = actions.to(model.device)
            distribution = model.policy.get_distribution(obs)
            loss = -distribution.log_prob(actions).mean()
            predictions = distribution.distribution.probs.argmax(dim=1)
            correct += int((predictions == actions).sum().item())
            total += int(actions.numel())
            losses.append(float(loss.item()) * int(actions.numel()))
    return {
        "loss": sum(losses) / total,
        "accuracy": correct / total if total else 0.0,
    }


def train_model_bc(
    model: PPO,
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> dict[str, Any]:
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=learning_rate)
    history = []
    for epoch in range(1, epochs + 1):
        model.policy.set_training_mode(True)
        total_loss = 0.0
        total = 0
        correct = 0
        for obs, actions in loader:
            obs = obs.to(model.device)
            actions = actions.to(model.device)
            distribution = model.policy.get_distribution(obs)
            loss = -distribution.log_prob(actions).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                predictions = distribution.distribution.probs.argmax(dim=1)
                correct += int((predictions == actions).sum().item())
            batch_size_actual = int(actions.numel())
            total += batch_size_actual
            total_loss += float(loss.item()) * batch_size_actual

        train_metrics = {
            "loss": total_loss / total,
            "accuracy": correct / total if total else 0.0,
        }
        val_metrics = evaluate_dataset(model, val_dataset, batch_size=batch_size)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(row)
        print(f"BC epoch {epoch}: {row}", flush=True)
    return {
        "history": history,
        "final_train": {
            "loss": history[-1]["train_loss"],
            "accuracy": history[-1]["train_accuracy"],
        },
        "final_val": {
            "loss": history[-1]["val_loss"],
            "accuracy": history[-1]["val_accuracy"],
        },
    }


def train_delivery_bc(config: dict, args: argparse.Namespace) -> Path:
    register_envs()
    run_dir = Path(args.output_dir) if args.output_dir else run_dir_from_name(config["run_name"])
    model_dir = run_dir / "models"
    metrics_dir = run_dir / "metrics"
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_bc_dataset(args.demo_path)
    env = gym.make(config["env_id"], **env_config_from_config(config))
    alt_env = env.getDummyEnv(1)
    init_run_dir = Path(config["init_run_dir"])
    ego = PPO.load(str(init_run_dir / "models" / "ego"), env=env, device=config["device"])
    alt = PPO.load(str(init_run_dir / "models" / "alt"), env=alt_env, device=config["device"])

    ego_train, ego_val = split_dataset(
        dataset["ego_obs"],
        dataset["ego_actions"],
        seed=int(config["seed"]),
        val_fraction=args.val_fraction,
    )
    alt_train, alt_val = split_dataset(
        dataset["alt_obs"],
        dataset["alt_actions"],
        seed=int(config["seed"]) + 1,
        val_fraction=args.val_fraction,
    )

    ego_summary = train_model_bc(
        ego,
        ego_train,
        ego_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    alt_summary = train_model_bc(
        alt,
        alt_train,
        alt_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    ego.save(str(model_dir / "ego"))
    alt.save(str(model_dir / "alt"))
    env.close()

    resolved_config = dict(config)
    resolved_config["bc_demo_path"] = str(args.demo_path)
    resolved_config["bc_epochs"] = int(args.epochs)
    resolved_config["bc_batch_size"] = int(args.batch_size)
    resolved_config["bc_learning_rate"] = float(args.learning_rate)
    resolved_config["bc_val_fraction"] = float(args.val_fraction)
    save_json(run_dir / "config.resolved.json", resolved_config)
    dataset_summary = {
        "action_counts": dataset["action_counts"],
        "num_steps": dataset["num_steps"],
        "num_successful_episodes": dataset["num_successful_episodes"],
        "source_summary": dataset["source_summary"],
        "ego_observation_shape": list(dataset["ego_obs"].shape),
        "alt_observation_shape": list(dataset["alt_obs"].shape),
    }

    summary = {
        "run_name": config["run_name"],
        "init_run_dir": str(init_run_dir),
        "demo_path": str(args.demo_path),
        "dataset": dataset_summary,
        "ego": ego_summary,
        "alt": alt_summary,
        "ego_model": str(model_dir / "ego.zip"),
        "alt_model": str(model_dir / "alt.zip"),
    }
    save_json(metrics_dir / "bc_summary.json", summary)
    print(
        {
            "run_dir": str(run_dir),
            "dataset_steps": dataset["num_steps"],
            "ego_val_accuracy": ego_summary["final_val"]["accuracy"],
            "alt_val_accuracy": alt_summary["final_val"]["accuracy"],
        },
        flush=True,
    )
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Behavior-clone scripted small_corridor demonstrations.")
    parser.add_argument("--config", default="configs/small_corridor_delivery_warmstart_from_v3.json")
    parser.add_argument("--demo-path", default="outputs/demos/small_corridor_delivery_scripted.json")
    parser.add_argument("--run-name", default="small_corridor_delivery_bc_from_v3")
    parser.add_argument("--output-dir")
    parser.add_argument("--init-run-dir", help="Override config init_run_dir.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--device", help="Override config device.")
    parser.add_argument("--seed", type=int, help="Override config seed.")
    return parser


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    config = dict(config)
    config["run_name"] = args.run_name
    if args.init_run_dir:
        config["init_run_dir"] = args.init_run_dir
    if args.device:
        config["device"] = args.device
    if args.seed is not None:
        config["seed"] = args.seed
    return config


def main() -> None:
    args = build_parser().parse_args()
    config = apply_overrides(load_config(args.config), args)
    train_delivery_bc(config, args)


if __name__ == "__main__":
    main()
