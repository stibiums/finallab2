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
from .partner_conditioning import PartnerIDConditioningWrapper, partner_conditioning_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a partner-id conditioned ego against an unknown partner under each assumed id."
    )
    parser.add_argument("ego_run_dir", help="Conditioned run directory containing models/ego.zip.")
    parser.add_argument("--partner-run-dir", required=True, help="Unknown partner run containing models/alt.zip.")
    parser.add_argument("--layout", default="random1")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--output-name", default="unknown_partner_conditioning")
    parser.add_argument("--assumed-partner-id", type=int, action="append", help="Assumed id to test. Repeatable.")
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


def evaluate_conditioned_unknown(
    ego_run_dir: str | Path,
    partner_run_dir: str | Path,
    layout: str,
    episodes: int,
    deterministic: bool,
    output_name: str,
    assumed_partner_ids: list[int] | None = None,
) -> dict:
    register_envs()
    ego_run_dir = Path(ego_run_dir)
    partner_run_dir = Path(partner_run_dir)
    config = load_json(ego_run_dir / "config.resolved.json")
    conditioning = partner_conditioning_config(config)
    if not conditioning["enabled"]:
        raise ValueError(f"{ego_run_dir} is not a partner-id conditioned run")
    num_partners = int(conditioning.get("num_partners") or len(config.get("partner_run_dirs", [])))
    if num_partners <= 0:
        raise ValueError("Could not infer number of conditioned partner ids")
    if assumed_partner_ids is None:
        assumed_partner_ids = list(range(num_partners))

    layout_config = dict(config)
    layout_config["layout_name"] = layout
    layout_config["layout_names"] = [layout]
    layout_config.pop("layout_sampling_weights", None)

    ego_model = PPO.load(str(ego_run_dir / "models" / "ego.zip"))
    partner_model = PPO.load(str(partner_run_dir / "models" / "alt.zip"))
    metrics_dir = ego_run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    detail_rows = []
    for assumed_id in assumed_partner_ids:
        env = gym.make(layout_config["env_id"], **env_config_from_config(layout_config))
        env.add_partner_agent(StaticPolicyAgent(partner_model.policy))
        env = PartnerIDConditioningWrapper(env, num_partners=num_partners, fixed_partner_id=int(assumed_id))
        episode_rows = [run_episode(env, ego_model, deterministic) for _ in range(episodes)]
        env.close()
        stats = summarize(episode_rows)
        row = {
            "ego_run": ego_run_dir.name,
            "unknown_partner_run": partner_run_dir.name,
            "layout": layout,
            "episodes": episodes,
            "deterministic": deterministic,
            "assumed_partner_id": int(assumed_id),
            **stats,
        }
        summary_rows.append(row)
        for episode_idx, episode_row in enumerate(episode_rows):
            detail_rows.append(
                {
                    "ego_run": ego_run_dir.name,
                    "unknown_partner_run": partner_run_dir.name,
                    "layout": layout,
                    "episode": episode_idx,
                    "assumed_partner_id": int(assumed_id),
                    **episode_row,
                }
            )
        print(row)

    summary_path = metrics_dir / f"{output_name}.csv"
    detail_path = metrics_dir / f"{output_name}_episodes.csv"
    summary_fields = [
        "ego_run",
        "unknown_partner_run",
        "layout",
        "episodes",
        "deterministic",
        "assumed_partner_id",
        "mean_episode_reward",
        "std_episode_reward",
        "mean_sparse_reward",
        "mean_soups_delivered",
    ]
    detail_fields = [
        "ego_run",
        "unknown_partner_run",
        "layout",
        "episode",
        "assumed_partner_id",
        "episode_reward",
        "sparse_reward",
        "shaped_reward",
        "soups_delivered",
        "steps",
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
        "unknown_partner_run": partner_run_dir.name,
        "layout": layout,
        "episodes": episodes,
        "deterministic": deterministic,
        "num_conditioned_partner_ids": num_partners,
        "summary_csv": str(summary_path),
        "episode_csv": str(detail_path),
        "rows": summary_rows,
    }
    save_json(metrics_dir / f"{output_name}.json", payload)
    return payload


def main() -> None:
    args = build_parser().parse_args()
    evaluate_conditioned_unknown(
        ego_run_dir=args.ego_run_dir,
        partner_run_dir=args.partner_run_dir,
        layout=args.layout,
        episodes=args.episodes,
        deterministic=not args.stochastic,
        output_name=args.output_name,
        assumed_partner_ids=args.assumed_partner_id,
    )


if __name__ == "__main__":
    main()
