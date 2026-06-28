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


def build_full_chain_action_plan(cycles: int = 1) -> list[tuple[int, int]]:
    if cycles < 1:
        raise ValueError("cycles must be at least 1.")

    n, s, e, w, stay, interact = (
        Direction.NORTH,
        Direction.SOUTH,
        Direction.EAST,
        Direction.WEST,
        Action.STAY,
        Action.INTERACT,
    )

    def p0_trip(return_after: bool) -> list:
        actions = [w, w, w, s, s] + [e] * 8 + [s, interact]
        if return_after:
            actions += [w] * 8 + [n, n] + [e] * 3 + [n, interact]
        return actions

    def first_cycle_p0() -> list:
        return [e, e, n, interact] + p0_trip(return_after=True) + p0_trip(return_after=True) + p0_trip(
            return_after=False
        ) + [e]

    def first_cycle_p1(p0_cycle_len: int) -> list:
        actions = [w, w, n, interact, e, e, e, s]
        while len(actions) < p0_cycle_len:
            actions.append(stay)
        return actions + [s] + [stay] * 25 + [interact] * 5 + [w] * 9 + [s, interact]

    def later_cycle_p0() -> list:
        return [w] * 9 + [n, n] + [e] * 3 + [n, interact] + p0_trip(
            return_after=True
        ) + p0_trip(return_after=True) + p0_trip(return_after=False) + [e]

    def later_cycle_p1() -> list:
        return (
            [stay] * 90
            + [e] * 9
            + [n, n]
            + [w] * 3
            + [n, interact]
            + [e] * 3
            + [s, s]
            + [interact] * 5
            + [w] * 9
            + [s, interact]
        )

    def append_cycle(p0_cycle: list, p1_cycle: list) -> None:
        cycle_len = max(len(p0_cycle), len(p1_cycle))
        p0_actions.extend(p0_cycle + [stay] * (cycle_len - len(p0_cycle)))
        p1_actions.extend(p1_cycle + [stay] * (cycle_len - len(p1_cycle)))

    p0_actions: list = []
    p1_actions: list = []
    first_p0 = first_cycle_p0()
    append_cycle(first_p0, first_cycle_p1(len(first_p0)))
    for _ in range(cycles - 1):
        append_cycle(later_cycle_p0(), later_cycle_p1())

    return [(action_index(a0), action_index(a1)) for a0, a1 in zip(p0_actions, p1_actions)]


def validate_mode_config(config: dict[str, Any], mode: str) -> None:
    if mode != "full_chain":
        return
    if config.get("layout_name") != "small_corridor":
        raise ValueError("full_chain scripted demos are hard-coded for layout_name='small_corridor'.")
    if config.get("start_state_mode") not in (None, "standard"):
        raise ValueError("full_chain scripted demos require the standard small_corridor start state.")


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


def collect_episode(env, max_steps: int, mode: str, full_chain_cycles: int) -> dict[str, Any]:
    env.unwrapped.multi_reset()
    initial_state = state_summary(env)
    rows = []
    total_reward = 0.0
    sparse_reward = 0.0
    action_counts = Counter()
    full_chain_plan = build_full_chain_action_plan(full_chain_cycles) if mode == "full_chain" else None
    target_sparse_reward = 20.0 * full_chain_cycles if mode == "full_chain" else 20.0

    for step in range(max_steps):
        before = state_summary(env)
        state = env.unwrapped.base_env.state
        observations = env.unwrapped.featurize_fn(state)
        if mode == "delivery":
            ego_action = scripted_action_for_player(state.players[0])
            alt_action = scripted_action_for_player(state.players[1])
        elif mode == "full_chain":
            if full_chain_plan is None or step >= len(full_chain_plan):
                ego_action = action_index(Action.STAY)
                alt_action = action_index(Action.STAY)
            else:
                ego_action, alt_action = full_chain_plan[step]
        else:
            raise ValueError(f"Unsupported scripted demo mode: {mode}")
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
        if done or sparse_reward >= target_sparse_reward:
            break

    return {
        "initial_state": initial_state,
        "final_state": rows[-1]["after"] if rows else initial_state,
        "steps": len(rows),
        "total_reward": total_reward,
        "sparse_reward": sparse_reward,
        "soups_delivered": sparse_reward / 20.0,
        "success": sparse_reward >= target_sparse_reward,
        "action_counts": dict(action_counts),
        "steps_detail": rows,
    }


def collect_demos(config_path: str | Path, episodes: int, max_steps: int, output_path: str | Path):
    return collect_scripted_demos(
        config_path=config_path,
        episodes=episodes,
        max_steps=max_steps,
        output_path=output_path,
        mode="delivery",
        full_chain_cycles=1,
    )


def collect_scripted_demos(
    config_path: str | Path,
    episodes: int,
    max_steps: int,
    output_path: str | Path,
    mode: str,
    full_chain_cycles: int,
):
    register_envs()
    config = load_config(config_path)
    validate_mode_config(config, mode)
    env = gym.make(config["env_id"], **env_config_from_config(config))
    demos = [
        collect_episode(env, max_steps=max_steps, mode=mode, full_chain_cycles=full_chain_cycles)
        for _ in range(episodes)
    ]
    env.close()

    successes = [demo for demo in demos if demo["success"]]
    summary = {
        "config": str(config_path),
        "mode": mode,
        "full_chain_cycles": full_chain_cycles if mode == "full_chain" else None,
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
            "mode": mode,
            "full_chain_cycles": full_chain_cycles if mode == "full_chain" else None,
            "episodes": episodes,
            "successes": summary["successes"],
            "success_rate": summary["success_rate"],
            "mean_success_steps": summary["mean_success_steps"],
        },
        flush=True,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect scripted small_corridor demos.")
    parser.add_argument("--config", default="configs/small_corridor_delivery_warmstart_from_v3.json")
    parser.add_argument(
        "--mode",
        choices=["delivery", "full_chain"],
        default="delivery",
        help="Scripted policy to collect: isolated delivery or full standard-start chain.",
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument(
        "--full-chain-cycles",
        type=int,
        default=1,
        help="Number of standard-start soup cycles to script when --mode full_chain.",
    )
    parser.add_argument(
        "--output",
        default="outputs/demos/small_corridor_delivery_scripted.json",
        help="Output JSON path for scripted demonstrations.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    collect_scripted_demos(
        config_path=args.config,
        episodes=args.episodes,
        max_steps=args.max_steps,
        output_path=args.output,
        mode=args.mode,
        full_chain_cycles=args.full_chain_cycles,
    )


if __name__ == "__main__":
    main()
