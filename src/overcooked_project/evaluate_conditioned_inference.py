from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import gym
from pantheonrl.common.util import action_from_policy, clip_actions
from stable_baselines3 import PPO

from . import register_envs
from .config import env_config_from_config, load_json, save_json
from .partner_conditioning import append_partner_id, partner_conditioning_config
from .trace_episode import action_text, state_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a partner-id conditioned ego with online known-partner id inference."
    )
    parser.add_argument("ego_run_dir", help="Conditioned run directory containing models/ego.zip.")
    parser.add_argument("--partner-run-dir", required=True, help="Unknown partner run containing models/alt.zip.")
    parser.add_argument("--layout", default="random1")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--output-name", default="unknown_partner_inferred_ids")
    parser.add_argument("--initial-assumed-id", type=int, action="append", help="Initial id to test. Repeatable.")
    return parser


def current_observations(env) -> tuple[Any, Any]:
    state = env.unwrapped.base_env.state
    ob_p0, ob_p1 = env.unwrapped.featurize_fn(state)
    if env.unwrapped.ego_agent_idx == 0:
        return ob_p0, ob_p1
    return ob_p1, ob_p0


def policy_action(model: PPO, obs) -> int:
    actions, _, _ = action_from_policy(obs, model.policy)
    return int(clip_actions(actions, model.policy)[0])


def best_id_with_sticky_tie(scores: list[int], current_id: int) -> int:
    best_score = max(scores)
    best_ids = [idx for idx, score in enumerate(scores) if score == best_score]
    if current_id in best_ids:
        return current_id
    return best_ids[0]


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    rewards = [float(row["episode_reward"]) for row in rows]
    sparse_rewards = [float(row["sparse_reward"]) for row in rows]
    soups = [float(row["soups_delivered"]) for row in rows]
    switches = [float(row["id_switches"]) for row in rows]
    return {
        "mean_episode_reward": mean(rewards),
        "std_episode_reward": pstdev(rewards) if len(rewards) > 1 else 0.0,
        "mean_sparse_reward": mean(sparse_rewards),
        "mean_soups_delivered": mean(soups),
        "mean_id_switches": mean(switches),
    }


def run_episode(
    env,
    ego_model: PPO,
    unknown_partner_model: PPO,
    known_partner_models: list[PPO],
    initial_assumed_id: int,
    deterministic: bool,
    record_trace: bool = False,
) -> dict[str, Any]:
    env.unwrapped.multi_reset()
    initial_state = state_summary(env)
    num_partners = len(known_partner_models)
    current_id = int(initial_assumed_id)
    match_scores = [0 for _ in known_partner_models]
    id_counts: Counter[int] = Counter()
    id_switches = 0
    rows = []
    total_reward = 0.0
    sparse_reward = 0.0
    shaped_reward = 0.0
    done = False
    steps = 0

    while not done and steps < int(getattr(env.unwrapped, "horizon", 400)):
        before = state_summary(env) if record_trace else None
        ego_obs, alt_obs = current_observations(env)
        conditioned_obs = append_partner_id(ego_obs, num_partners, current_id)
        ego_action, _ = ego_model.predict(conditioned_obs, deterministic=deterministic)
        unknown_action = policy_action(unknown_partner_model, alt_obs)
        known_actions = [policy_action(model, alt_obs) for model in known_partner_models]

        for partner_id, action in enumerate(known_actions):
            if int(action) == int(unknown_action):
                match_scores[partner_id] += 1

        next_id = best_id_with_sticky_tie(match_scores, current_id)
        if next_id != current_id:
            id_switches += 1
        id_used = current_id
        current_id = next_id

        _, rewards, done, info = env.unwrapped.multi_step(int(ego_action), int(unknown_action))
        row_sparse_reward = float(info.get("sparse_reward", 0.0))
        row_shaped_reward = float(info.get("shaped_reward", 0.0))
        row_reward = float(rewards[0])
        total_reward += row_reward
        sparse_reward += row_sparse_reward
        shaped_reward += row_shaped_reward
        id_counts[id_used] += 1

        if record_trace:
            rows.append(
                {
                    "step": steps,
                    "assumed_id_used": id_used,
                    "next_assumed_id": current_id,
                    "match_scores": list(match_scores),
                    "ego_action": int(ego_action),
                    "unknown_partner_action": int(unknown_action),
                    "known_partner_actions": [int(action) for action in known_actions],
                    "ego_action_text": action_text(int(ego_action)),
                    "unknown_partner_action_text": action_text(int(unknown_action)),
                    "known_partner_action_text": [action_text(int(action)) for action in known_actions],
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
        "initial_assumed_id": int(initial_assumed_id),
        "final_assumed_id": int(current_id),
        "id_switches": int(id_switches),
        "id_counts": dict(id_counts),
        "final_match_scores": list(match_scores),
    }
    if record_trace:
        result.update(
            {
                "initial_state": initial_state,
                "final_state": rows[-1]["after"] if rows else initial_state,
                "steps_detail": rows,
            }
        )
    return result


def evaluate_conditioned_inference(
    ego_run_dir: str | Path,
    partner_run_dir: str | Path,
    layout: str,
    episodes: int,
    deterministic: bool,
    output_name: str,
    initial_assumed_ids: list[int] | None = None,
) -> dict[str, Any]:
    register_envs()
    ego_run_dir = Path(ego_run_dir)
    partner_run_dir = Path(partner_run_dir)
    config = load_json(ego_run_dir / "config.resolved.json")
    conditioning = partner_conditioning_config(config)
    if not conditioning["enabled"]:
        raise ValueError(f"{ego_run_dir} is not a partner-id conditioned run")
    known_partner_run_dirs = [Path(path) for path in config.get("partner_run_dirs", [])]
    num_partners = int(conditioning.get("num_partners") or len(known_partner_run_dirs))
    if num_partners != len(known_partner_run_dirs):
        raise ValueError("num_partners must match partner_run_dirs for inference")
    if initial_assumed_ids is None:
        initial_assumed_ids = list(range(num_partners))

    layout_config = dict(config)
    layout_config["layout_name"] = layout
    layout_config["layout_names"] = [layout]
    layout_config.pop("layout_sampling_weights", None)

    ego_model = PPO.load(str(ego_run_dir / "models" / "ego.zip"))
    unknown_partner_model = PPO.load(str(partner_run_dir / "models" / "alt.zip"))
    known_partner_models = [PPO.load(str(path / "models" / "alt.zip")) for path in known_partner_run_dirs]
    metrics_dir = ego_run_dir / "metrics"
    trace_dir = ego_run_dir / "traces"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    detail_rows = []
    for initial_id in initial_assumed_ids:
        env = gym.make(layout_config["env_id"], **env_config_from_config(layout_config))
        episode_rows = []
        for episode_idx in range(episodes):
            row = run_episode(
                env=env,
                ego_model=ego_model,
                unknown_partner_model=unknown_partner_model,
                known_partner_models=known_partner_models,
                initial_assumed_id=int(initial_id),
                deterministic=deterministic,
                record_trace=episode_idx == 0,
            )
            if episode_idx == 0:
                trace_path = trace_dir / f"{output_name}_initial{initial_id}_episode0.json"
                save_json(trace_path, {"episode": episode_idx, **row})
            episode_rows.append(row)
            detail_rows.append({"episode": episode_idx, **row})
        env.close()

        stats = summarize(episode_rows)
        final_ids = Counter(int(row["final_assumed_id"]) for row in episode_rows)
        row = {
            "ego_run": ego_run_dir.name,
            "unknown_partner_run": partner_run_dir.name,
            "layout": layout,
            "episodes": episodes,
            "deterministic": deterministic,
            "initial_assumed_id": int(initial_id),
            "final_id_counts": dict(final_ids),
            **stats,
        }
        summary_rows.append(row)
        print(row)

    summary_path = metrics_dir / f"{output_name}.csv"
    detail_path = metrics_dir / f"{output_name}_episodes.csv"
    summary_fields = [
        "ego_run",
        "unknown_partner_run",
        "layout",
        "episodes",
        "deterministic",
        "initial_assumed_id",
        "mean_episode_reward",
        "std_episode_reward",
        "mean_sparse_reward",
        "mean_soups_delivered",
        "mean_id_switches",
        "final_id_counts",
    ]
    detail_fields = [
        "episode",
        "initial_assumed_id",
        "final_assumed_id",
        "id_switches",
        "episode_reward",
        "sparse_reward",
        "shaped_reward",
        "soups_delivered",
        "steps",
        "id_counts",
        "final_match_scores",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    with detail_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in detail_fields} for row in detail_rows)

    payload = {
        "ego_run": ego_run_dir.name,
        "unknown_partner_run": partner_run_dir.name,
        "known_partner_runs": [path.name for path in known_partner_run_dirs],
        "layout": layout,
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
    evaluate_conditioned_inference(
        ego_run_dir=args.ego_run_dir,
        partner_run_dir=args.partner_run_dir,
        layout=args.layout,
        episodes=args.episodes,
        deterministic=not args.stochastic,
        output_name=args.output_name,
        initial_assumed_ids=args.initial_assumed_id,
    )


if __name__ == "__main__":
    main()
