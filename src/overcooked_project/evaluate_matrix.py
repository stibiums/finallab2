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
from .evaluate import run_episode


DEFAULT_LAYOUTS = ["simple", "simple_tomato", "small_corridor", "random0", "random1", "unident_s"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate one ego model across layouts and fixed partners.")
    parser.add_argument("--ego-run-dir", required=True, help="Run directory containing models/ego.zip.")
    parser.add_argument("--partner-run-dir", action="append", help="Run directory containing models/alt.zip.")
    parser.add_argument("--layout", action="append", help="Layout to evaluate. Repeatable.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--output-name", default="matrix_eval")
    return parser


def summarize(rows: list[dict]) -> dict:
    rewards = [row["episode_reward"] for row in rows]
    sparse_rewards = [row["sparse_reward"] for row in rows]
    soups = [row["soups_delivered"] for row in rows]
    return {
        "mean_episode_reward": mean(rewards),
        "std_episode_reward": pstdev(rewards) if len(rewards) > 1 else 0.0,
        "mean_sparse_reward": mean(sparse_rewards),
        "mean_soups_delivered": mean(soups),
    }


def evaluate_matrix(
    ego_run_dir: str | Path,
    partner_run_dirs: list[str | Path],
    layouts: list[str],
    episodes: int,
    deterministic: bool,
    output_name: str,
) -> dict:
    register_envs()
    ego_run_dir = Path(ego_run_dir)
    config = load_json(ego_run_dir / "config.resolved.json")
    ego_model = PPO.load(str(ego_run_dir / "models" / "ego.zip"))
    metrics_dir = ego_run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    partner_run_dirs = [Path(p) for p in partner_run_dirs]
    if not partner_run_dirs:
        partner_run_dirs = [ego_run_dir]

    summary_rows = []
    detail_rows = []
    for layout in layouts:
        layout_config = dict(config)
        layout_config["layout_name"] = layout
        layout_config["layout_names"] = [layout]
        layout_config.pop("layout_sampling_weights", None)
        for partner_run_dir in partner_run_dirs:
            base_row = {
                "ego_run": ego_run_dir.name,
                "partner_run": partner_run_dir.name,
                "layout": layout,
                "episodes": episodes,
                "deterministic": deterministic,
                "status": "ok",
                "error_type": "",
                "error": "",
            }
            env = None
            try:
                partner_model_path = partner_run_dir / "models" / "alt.zip"
                if not partner_model_path.exists():
                    raise FileNotFoundError(f"Missing partner model: {partner_model_path}")
                partner_model = PPO.load(str(partner_model_path))
                env = gym.make(layout_config["env_id"], **env_config_from_config(layout_config))
                env.add_partner_agent(StaticPolicyAgent(partner_model.policy))
                episode_rows = [run_episode(env, ego_model, deterministic) for _ in range(episodes)]
                stats = summarize(episode_rows)
                row = {**base_row, **stats}
                for episode_idx, episode_row in enumerate(episode_rows):
                    detail_rows.append({
                        "ego_run": ego_run_dir.name,
                        "partner_run": partner_run_dir.name,
                        "layout": layout,
                        "episode": episode_idx,
                        "status": "ok",
                        "error_type": "",
                        "error": "",
                        **episode_row,
                    })
            except Exception as exc:
                row = {
                    **base_row,
                    "status": "error",
                    "mean_episode_reward": None,
                    "std_episode_reward": None,
                    "mean_sparse_reward": None,
                    "mean_soups_delivered": None,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                detail_rows.append({
                    "ego_run": ego_run_dir.name,
                    "partner_run": partner_run_dir.name,
                    "layout": layout,
                    "episode": "",
                    "episode_reward": None,
                    "sparse_reward": None,
                    "shaped_reward": None,
                    "soups_delivered": None,
                    "steps": None,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
            finally:
                if env is not None:
                    env.close()
            summary_rows.append(row)
            print(row)

    summary_path = metrics_dir / f"{output_name}.csv"
    detail_path = metrics_dir / f"{output_name}_episodes.csv"
    summary_fields = [
        "ego_run",
        "partner_run",
        "layout",
        "episodes",
        "deterministic",
        "status",
        "mean_episode_reward",
        "std_episode_reward",
        "mean_sparse_reward",
        "mean_soups_delivered",
        "error_type",
        "error",
    ]
    detail_fields = [
        "ego_run",
        "partner_run",
        "layout",
        "episode",
        "status",
        "episode_reward",
        "sparse_reward",
        "shaped_reward",
        "soups_delivered",
        "steps",
        "error_type",
        "error",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    with detail_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(detail_rows)

    payload = {
        "ego_run": ego_run_dir.name,
        "partner_runs": [p.name for p in partner_run_dirs],
        "layouts": layouts,
        "episodes": episodes,
        "deterministic": deterministic,
        "summary_csv": str(summary_path),
        "episode_csv": str(detail_path),
        "rows": summary_rows,
    }
    save_json(metrics_dir / f"{output_name}.json", payload)
    return payload


def main() -> None:
    args = build_parser().parse_args()
    evaluate_matrix(
        ego_run_dir=args.ego_run_dir,
        partner_run_dirs=args.partner_run_dir or [],
        layouts=args.layout or DEFAULT_LAYOUTS,
        episodes=args.episodes,
        deterministic=not args.stochastic,
        output_name=args.output_name,
    )


if __name__ == "__main__":
    main()
