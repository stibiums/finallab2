from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from statistics import mean, pstdev

import gym
from pantheonrl.common.agents import OnPolicyAgent, StaticPolicyAgent
from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_schedule_fn

from . import register_envs
from .config import env_config_from_config, load_config, run_dir_from_config, save_json
from .evaluate import run_episode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune PPO agents with staged Overcooked curricula.")
    parser.add_argument("--config", required=True, help="Path to a JSON config file.")
    parser.add_argument("--run-name", help="Override config run_name.")
    parser.add_argument("--init-run-dir", help="Run directory to load initial models from.")
    parser.add_argument("--timesteps", type=int, help="Override total_timesteps.")
    parser.add_argument("--eval-interval", type=int, help="Override curriculum_eval_interval.")
    parser.add_argument("--eval-episodes", type=int, help="Override eval_episodes.")
    parser.add_argument("--seed", type=int, help="Override random seed.")
    parser.add_argument("--device", help="Override torch device: auto, cpu, cuda, mps.")
    return parser


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    if args.run_name:
        config["run_name"] = args.run_name
    if args.init_run_dir:
        config["init_run_dir"] = args.init_run_dir
    if args.timesteps is not None:
        config["total_timesteps"] = args.timesteps
    if args.eval_interval is not None:
        config["curriculum_eval_interval"] = args.eval_interval
    if args.eval_episodes is not None:
        config["eval_episodes"] = args.eval_episodes
    if args.seed is not None:
        config["seed"] = args.seed
    if args.device:
        config["device"] = args.device
    return config


def apply_learning_rate(model: PPO, learning_rate: float) -> None:
    model.learning_rate = learning_rate
    model.lr_schedule = get_schedule_fn(learning_rate)


def fixed_layout_config(config: dict, layout: str) -> dict:
    layout_config = dict(config)
    layout_config["layout_name"] = layout
    layout_config["layout_names"] = [layout]
    layout_config.pop("layout_sampling_weights", None)
    return layout_config


def summarize_rows(rows: list[dict]) -> dict:
    rewards = [row["episode_reward"] for row in rows]
    sparse_rewards = [row["sparse_reward"] for row in rows]
    soups = [row["soups_delivered"] for row in rows]
    return {
        "mean_episode_reward": mean(rewards),
        "std_episode_reward": pstdev(rewards) if len(rewards) > 1 else 0.0,
        "mean_sparse_reward": mean(sparse_rewards),
        "mean_soups_delivered": mean(soups),
    }


def evaluate_pair(
    config: dict,
    ego_model: PPO,
    alt_model: PPO,
    layouts: list[str],
    episodes: int,
    deterministic: bool,
    trained_timesteps: int,
) -> tuple[list[dict], list[dict]]:
    summary_rows = []
    episode_rows = []
    for layout in layouts:
        env = gym.make(config["env_id"], **env_config_from_config(fixed_layout_config(config, layout)))
        env.add_partner_agent(StaticPolicyAgent(alt_model.policy))
        rows = [run_episode(env, ego_model, deterministic) for _ in range(episodes)]
        env.close()

        stats = summarize_rows(rows)
        summary_row = {
            "trained_timesteps": trained_timesteps,
            "layout": layout,
            "episodes": episodes,
            "deterministic": deterministic,
            **stats,
        }
        summary_rows.append(summary_row)
        for episode_idx, row in enumerate(rows):
            episode_rows.append(
                {
                    "trained_timesteps": trained_timesteps,
                    "layout": layout,
                    "episode": episode_idx,
                    **row,
                }
            )
        print(f"Curriculum eval: {summary_row}")
    return summary_rows, episode_rows


def curriculum_score(summary_rows: list[dict]) -> dict:
    soups = [row["mean_soups_delivered"] for row in summary_rows]
    mean_soups = mean(soups)
    min_soups = min(soups)
    return {
        "score": min_soups + 0.05 * mean_soups,
        "mean_soups_across_layouts": mean_soups,
        "min_soups_across_layouts": min_soups,
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train_curriculum(config: dict) -> Path:
    register_envs()
    if "init_run_dir" not in config:
        raise ValueError("Curriculum fine-tuning requires init_run_dir.")

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

    init_run_dir = Path(config["init_run_dir"])
    ego = PPO.load(
        str(init_run_dir / "models" / "ego"),
        env=env,
        device=config["device"],
        tensorboard_log=str(tensorboard_dir),
    )
    alt_model = PPO.load(
        str(init_run_dir / "models" / "alt"),
        env=alt_env,
        device=config["device"],
        tensorboard_log=str(tensorboard_dir),
    )
    apply_learning_rate(ego, float(config["ego_config"]["learning_rate"]))
    apply_learning_rate(alt_model, float(config["alt_config"]["learning_rate"]))

    partner = OnPolicyAgent(
        alt_model,
        tensorboard_log=str(tensorboard_dir),
        tb_log_name=f"{config['run_name']}_alt",
    )
    env.add_partner_agent(partner)

    total_timesteps = int(config["total_timesteps"])
    eval_interval = int(config.get("curriculum_eval_interval", total_timesteps))
    eval_episodes = int(config.get("eval_episodes", 20))
    eval_layouts = list(config.get("eval_layouts", config.get("layout_names", [config["layout_name"]])))
    deterministic = bool(config.get("deterministic_eval", True))

    summary_rows: list[dict] = []
    episode_rows: list[dict] = []
    best = {"score": float("-inf")}
    best_eval = None

    def evaluate_and_checkpoint(trained_timesteps: int) -> None:
        nonlocal best, best_eval
        rows, episodes = evaluate_pair(
            config,
            ego,
            alt_model,
            eval_layouts,
            eval_episodes,
            deterministic,
            trained_timesteps,
        )
        score_row = {"trained_timesteps": trained_timesteps, **curriculum_score(rows)}
        for row in rows:
            row.update(score_row)
        summary_rows.extend(rows)
        episode_rows.extend(episodes)
        print(f"Curriculum score: {score_row}")
        if score_row["score"] > best["score"]:
            best = score_row
            best_eval = rows
            ego.save(str(model_dir / "ego"))
            alt_model.save(str(model_dir / "alt"))
            ego.save(str(model_dir / "best_ego"))
            alt_model.save(str(model_dir / "best_alt"))
            print(f"Saved new best checkpoint at {trained_timesteps} trained timesteps.")

    started = time.time()
    evaluate_and_checkpoint(0)
    trained = 0
    while trained < total_timesteps:
        chunk = min(eval_interval, total_timesteps - trained)
        print(f"Training curriculum chunk: {trained} -> {trained + chunk} / {total_timesteps}")
        ego.learn(
            total_timesteps=chunk,
            reset_num_timesteps=False,
            tb_log_name=f"{config['run_name']}_ego",
        )
        trained += chunk
        evaluate_and_checkpoint(trained)
    elapsed = time.time() - started

    ego.save(str(model_dir / "final_ego"))
    alt_model.save(str(model_dir / "final_alt"))

    summary_fields = [
        "trained_timesteps",
        "layout",
        "episodes",
        "deterministic",
        "mean_episode_reward",
        "std_episode_reward",
        "mean_sparse_reward",
        "mean_soups_delivered",
        "score",
        "mean_soups_across_layouts",
        "min_soups_across_layouts",
    ]
    episode_fields = [
        "trained_timesteps",
        "layout",
        "episode",
        "episode_reward",
        "sparse_reward",
        "shaped_reward",
        "soups_delivered",
        "steps",
    ]
    write_csv(metrics_dir / "curriculum_eval.csv", summary_rows, summary_fields)
    write_csv(metrics_dir / "curriculum_eval_episodes.csv", episode_rows, episode_fields)

    train_summary = {
        "run_name": config["run_name"],
        "init_run_dir": str(init_run_dir),
        "layout_name": config["layout_name"],
        "layout_names": list(config.get("layout_names", [config["layout_name"]])),
        "layout_sampling_weights": list(config.get("layout_sampling_weights", [])),
        "eval_layouts": eval_layouts,
        "total_timesteps": total_timesteps,
        "curriculum_eval_interval": eval_interval,
        "eval_episodes": eval_episodes,
        "seed": int(config["seed"]),
        "elapsed_seconds": elapsed,
        "best_checkpoint": {
            **best,
            "ego_model": str(model_dir / "ego.zip"),
            "alt_model": str(model_dir / "alt.zip"),
            "best_ego_model": str(model_dir / "best_ego.zip"),
            "best_alt_model": str(model_dir / "best_alt.zip"),
            "eval_rows": best_eval or [],
        },
        "final_checkpoint": {
            "ego_model": str(model_dir / "final_ego.zip"),
            "alt_model": str(model_dir / "final_alt.zip"),
        },
    }
    save_json(metrics_dir / "train_summary.json", train_summary)
    save_json(
        metrics_dir / "curriculum_eval_summary.json",
        {
            "best_checkpoint": train_summary["best_checkpoint"],
            "summary_csv": str(metrics_dir / "curriculum_eval.csv"),
            "episode_csv": str(metrics_dir / "curriculum_eval_episodes.csv"),
        },
    )
    print(f"Saved best models to {model_dir / 'ego.zip'} and {model_dir / 'alt.zip'}")
    print(f"Saved final models to {model_dir / 'final_ego.zip'} and {model_dir / 'final_alt.zip'}")
    print(f"Curriculum training elapsed seconds: {elapsed:.2f}")
    return run_dir


def main() -> None:
    args = build_parser().parse_args()
    config = apply_overrides(load_config(args.config), args)
    train_curriculum(config)


if __name__ == "__main__":
    main()
