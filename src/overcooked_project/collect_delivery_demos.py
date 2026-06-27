from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import gym
from overcooked_ai_py.mdp.actions import Action, Direction

from . import register_envs
from .config import env_config_from_config, load_config, save_json
from .trace_episode import state_summary


def action_index(action) -> int:
    return int(Action.ACTION_TO_INDEX[action])


def scripted_action_for_player(player) -> int:
    if player.held_object is None or player.held_object.name != "soup":
        return action_index(Action.STAY)
    if player.position[0] > 1:
        return action_index(Direction.WEST)
    if player.position != (1, 3):
        return action_index(Action.STAY)
    if player.orientation != Direction.SOUTH:
        return action_index(Direction.SOUTH)
    return action_index(Action.INTERACT)


def collect_episode(env, max_steps: int) -> dict[str, Any]:
    env.unwrapped.multi_reset()
    initial_state = state_summary(env)
    rows = []
    total_reward = 0.0
    sparse_reward = 0.0
    action_counts = Counter()

    for step in range(max_steps):
        before = state_summary(env)
        state = env.unwrapped.base_env.state
        observations = env.unwrapped.featurize_fn(state)
        ego_action = scripted_action_for_player(state.players[0])
        alt_action = scripted_action_for_player(state.players[1])
        _, rewards, done, info = env.unwrapped.multi_step(ego_action, alt_action)
        after = state_summary(env)

        joint_action_text = [
            str(Action.INDEX_TO_ACTION[ego_action]),
            str(Action.INDEX_TO_ACTION[alt_action]),
        ]
        action_counts.update(joint_action_text)
        row_sparse_reward = float(info.get("sparse_reward", 0.0))
        row_reward = float(rewards[0])
        total_reward += row_reward
        sparse_reward += row_sparse_reward
        rows.append(
            {
                "step": step,
                "joint_action": [ego_action, alt_action],
                "joint_action_text": joint_action_text,
                "reward": row_reward,
                "sparse_reward": row_sparse_reward,
                "shaped_reward": float(info.get("shaped_reward", 0.0)),
                "player_observations": [
                    observations[0].astype(float).tolist(),
                    observations[1].astype(float).tolist(),
                ],
                "before": before,
                "after": after,
            }
        )
        if done or row_sparse_reward > 0.0:
            break

    return {
        "initial_state": initial_state,
        "final_state": rows[-1]["after"] if rows else initial_state,
        "steps": len(rows),
        "total_reward": total_reward,
        "sparse_reward": sparse_reward,
        "soups_delivered": sparse_reward / 20.0,
        "success": sparse_reward > 0.0,
        "action_counts": dict(action_counts),
        "steps_detail": rows,
    }


def collect_demos(config_path: str | Path, episodes: int, max_steps: int, output_path: str | Path):
    register_envs()
    config = load_config(config_path)
    env = gym.make(config["env_id"], **env_config_from_config(config))
    demos = [collect_episode(env, max_steps=max_steps) for _ in range(episodes)]
    env.close()

    successes = [demo for demo in demos if demo["success"]]
    summary = {
        "config": str(config_path),
        "episodes": episodes,
        "max_steps": max_steps,
        "successes": len(successes),
        "success_rate": len(successes) / episodes if episodes else 0.0,
        "mean_success_steps": (
            sum(demo["steps"] for demo in successes) / len(successes) if successes else None
        ),
        "demos": demos,
    }
    save_json(output_path, summary)
    print(
        {
            "output_path": str(output_path),
            "episodes": episodes,
            "successes": summary["successes"],
            "success_rate": summary["success_rate"],
            "mean_success_steps": summary["mean_success_steps"],
        },
        flush=True,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect scripted small_corridor delivery demos.")
    parser.add_argument("--config", default="configs/small_corridor_delivery_warmstart_from_v3.json")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument(
        "--output",
        default="outputs/demos/small_corridor_delivery_scripted.json",
        help="Output JSON path for scripted demonstrations.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    collect_demos(
        config_path=args.config,
        episodes=args.episodes,
        max_steps=args.max_steps,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
