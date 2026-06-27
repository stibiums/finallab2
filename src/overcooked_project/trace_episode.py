from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import gym
from overcooked_ai_py.mdp.actions import Action
from pantheonrl.common.agents import StaticPolicyAgent
from stable_baselines3 import PPO

from . import register_envs
from .config import env_config_from_config, load_json, save_json


def action_text(action_idx: int) -> str:
    return str(Action.INDEX_TO_ACTION[int(action_idx)])


def object_summary(obj) -> dict[str, Any] | None:
    if obj is None:
        return None
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return {"repr": repr(obj)}


def state_summary(env) -> dict[str, Any]:
    state = env.unwrapped.base_env.state
    players = []
    for player in state.players:
        players.append(
            {
                "position": list(player.position),
                "orientation": list(player.orientation),
                "held_object": object_summary(player.held_object),
            }
        )
    objects = []
    for position, obj in sorted(state.objects.items()):
        objects.append(
            {
                "position": list(position),
                "object": object_summary(obj),
            }
        )
    return {
        "layout_name": env.unwrapped.layout_name,
        "players": players,
        "objects": objects,
        "state_string": env.unwrapped.mdp.state_string(state),
    }


def trace_episode(
    run_dir: str | Path,
    max_steps: int = 400,
    deterministic: bool = True,
    output_name: str = "trace",
    layout: str | None = None,
) -> dict[str, Any]:
    register_envs()
    run_dir = Path(run_dir)
    config = load_json(run_dir / "config.resolved.json")
    if layout is not None:
        config = dict(config)
        config["layout_name"] = layout
        config["layout_names"] = [layout]
        config.pop("layout_sampling_weights", None)

    trace_dir = run_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(config["env_id"], **env_config_from_config(config))
    ego_model = PPO.load(str(run_dir / "models" / "ego"))
    alt_model = PPO.load(str(run_dir / "models" / "alt"))
    env.add_partner_agent(StaticPolicyAgent(alt_model.policy))

    obs = env.reset()
    rows = []
    total_reward = 0.0
    sparse_reward = 0.0
    shaped_reward = 0.0
    done = False
    steps = 0

    initial_state = state_summary(env)
    action_counts: Counter[str] = Counter()
    joint_action_counts: Counter[str] = Counter()

    while not done and steps < max_steps:
        before = state_summary(env)
        action, _ = ego_model.predict(obs, deterministic=deterministic)
        obs, reward, done, info = env.step(action)
        after = state_summary(env)

        ego_action = int(action)
        joint_action = [int(a) for a in info.get("joint_action", [])]
        ego_action_text = action_text(ego_action)
        joint_action_text = [action_text(a) for a in joint_action]
        action_counts[ego_action_text] += 1
        joint_action_counts[" | ".join(joint_action_text)] += 1

        row_sparse_reward = float(info.get("sparse_reward", 0.0))
        row_shaped_reward = float(info.get("shaped_reward", 0.0))
        row_reward = float(reward)
        total_reward += row_reward
        sparse_reward += row_sparse_reward
        shaped_reward += row_shaped_reward

        rows.append(
            {
                "step": steps,
                "ego_action": ego_action,
                "ego_action_text": ego_action_text,
                "joint_action": joint_action,
                "joint_action_text": joint_action_text,
                "reward": row_reward,
                "sparse_reward": row_sparse_reward,
                "shaped_reward": row_shaped_reward,
                "done": bool(done),
                "before": before,
                "after": after,
            }
        )
        steps += 1

    env.close()

    reward_events = [
        {
            "step": row["step"],
            "reward": row["reward"],
            "sparse_reward": row["sparse_reward"],
            "shaped_reward": row["shaped_reward"],
            "joint_action_text": row["joint_action_text"],
            "after": row["after"],
        }
        for row in rows
        if row["reward"] != 0.0 or row["sparse_reward"] != 0.0 or row["shaped_reward"] != 0.0
    ]
    summary = {
        "run_dir": str(run_dir),
        "layout_name": initial_state["layout_name"],
        "steps": steps,
        "deterministic": deterministic,
        "total_reward": total_reward,
        "sparse_reward": sparse_reward,
        "shaped_reward": shaped_reward,
        "soups_delivered": sparse_reward / 20.0,
        "initial_state": initial_state,
        "final_state": rows[-1]["after"] if rows else initial_state,
        "reward_event_count": len(reward_events),
        "first_reward_step": reward_events[0]["step"] if reward_events else None,
        "ego_action_counts": dict(action_counts),
        "joint_action_counts": dict(joint_action_counts),
        "reward_events": reward_events,
        "steps_detail": rows,
    }
    output_path = trace_dir / f"{output_name}.json"
    save_json(output_path, summary)
    print(f"Saved episode trace to {output_path}")
    print(
        {
            "steps": steps,
            "total_reward": total_reward,
            "sparse_reward": sparse_reward,
            "shaped_reward": shaped_reward,
            "soups_delivered": sparse_reward / 20.0,
            "reward_event_count": len(reward_events),
            "first_reward_step": reward_events[0]["step"] if reward_events else None,
        },
        flush=True,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trace one Overcooked episode from trained models.")
    parser.add_argument("--run-dir", required=True, help="Run directory under outputs/runs.")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy actions.")
    parser.add_argument("--output-name", default="trace")
    parser.add_argument("--layout", help="Force a single layout for tracing.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    trace_episode(
        args.run_dir,
        max_steps=args.max_steps,
        deterministic=not args.stochastic,
        output_name=args.output_name,
        layout=args.layout,
    )


if __name__ == "__main__":
    main()
