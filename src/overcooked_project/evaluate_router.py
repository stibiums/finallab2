from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import gym
from pantheonrl.common.agents import StaticPolicyAgent
from stable_baselines3 import PPO

from . import register_envs
from .config import env_config_from_config, load_json, save_json
from .evaluate import run_episode
from .evaluate_matrix import DEFAULT_LAYOUTS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a layout router that selects a specialist run per Overcooked layout."
    )
    parser.add_argument("--config", help="Router JSON config.")
    parser.add_argument(
        "--route",
        action="append",
        default=[],
        metavar="LAYOUT=RUN_DIR",
        help="Map one layout to one trained run directory. Repeatable.",
    )
    parser.add_argument("--layout", action="append", help="Layout to evaluate. Repeatable.")
    parser.add_argument("--default-run-dir", help="Fallback run directory for layouts without a route.")
    parser.add_argument("--episodes", type=int, help="Evaluation episodes per routed layout.")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy actions.")
    parser.add_argument("--output-dir", help="Directory where router metrics are written.")
    parser.add_argument("--output-name", help="Output file prefix under metrics/.")
    return parser


def parse_route(route_text: str) -> tuple[str, str]:
    if "=" not in route_text:
        raise ValueError(f"Route must have the form LAYOUT=RUN_DIR, got: {route_text}")
    layout, run_dir = route_text.split("=", 1)
    layout = layout.strip()
    run_dir = run_dir.strip()
    if not layout or not run_dir:
        raise ValueError(f"Route must have non-empty layout and run dir, got: {route_text}")
    return layout, run_dir


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    rewards = [row["episode_reward"] for row in rows]
    sparse_rewards = [row["sparse_reward"] for row in rows]
    soups = [row["soups_delivered"] for row in rows]
    return {
        "mean_episode_reward": mean(rewards),
        "std_episode_reward": pstdev(rewards) if len(rewards) > 1 else 0.0,
        "mean_sparse_reward": mean(sparse_rewards),
        "mean_soups_delivered": mean(soups),
    }


def load_router_config(config_path: str | Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    return load_json(config_path)


def resolve_output_dir(config: dict[str, Any], output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir)
    if "output_dir" in config:
        return Path(str(config["output_dir"]))
    output_root = Path(str(config.get("output_root", "outputs/runs")))
    return output_root / str(config.get("run_name", "layout_router"))


def resolve_routes(config: dict[str, Any], cli_routes: list[str]) -> dict[str, str]:
    routes = {str(layout): str(run_dir) for layout, run_dir in config.get("routes", {}).items()}
    for route_text in cli_routes:
        layout, run_dir = parse_route(route_text)
        routes[layout] = run_dir
    return routes


def fixed_layout_config(run_dir: Path, layout: str) -> dict[str, Any]:
    config = load_json(run_dir / "config.resolved.json")
    config["layout_name"] = layout
    config["layout_names"] = [layout]
    config.pop("layout_sampling_weights", None)
    return config


def evaluate_router(
    routes: dict[str, str],
    layouts: list[str],
    episodes: int,
    deterministic: bool,
    output_dir: str | Path,
    output_name: str,
    default_run_dir: str | None = None,
) -> dict[str, Any]:
    if episodes < 1:
        raise ValueError("episodes must be at least 1.")
    if not layouts:
        raise ValueError("At least one layout must be configured.")

    register_envs()
    output_dir = Path(output_dir)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    model_cache: dict[Path, tuple[PPO, PPO]] = {}
    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []

    def load_models(run_dir: Path) -> tuple[PPO, PPO]:
        if run_dir not in model_cache:
            ego_path = run_dir / "models" / "ego.zip"
            alt_path = run_dir / "models" / "alt.zip"
            if not ego_path.exists():
                raise FileNotFoundError(f"Missing ego model: {ego_path}")
            if not alt_path.exists():
                raise FileNotFoundError(f"Missing partner model: {alt_path}")
            model_cache[run_dir] = (PPO.load(str(ego_path)), PPO.load(str(alt_path)))
        return model_cache[run_dir]

    for layout in layouts:
        selected_run = routes.get(layout) or default_run_dir
        route_source = "route" if layout in routes else "default"
        base_row = {
            "layout": layout,
            "selected_run": str(selected_run or ""),
            "selected_run_name": Path(selected_run).name if selected_run else "",
            "route_source": route_source if selected_run else "",
            "episodes": episodes,
            "deterministic": deterministic,
            "status": "ok",
            "error_type": "",
            "error": "",
        }

        if selected_run is None:
            row = {
                **base_row,
                "status": "skipped",
                "mean_episode_reward": None,
                "std_episode_reward": None,
                "mean_sparse_reward": None,
                "mean_soups_delivered": None,
                "error_type": "NoRoute",
                "error": f"No run route configured for layout {layout}.",
            }
            detail_rows.append(
                {
                    **base_row,
                    "episode": "",
                    "status": "skipped",
                    "episode_reward": None,
                    "sparse_reward": None,
                    "shaped_reward": None,
                    "soups_delivered": None,
                    "steps": None,
                    "error_type": row["error_type"],
                    "error": row["error"],
                }
            )
            summary_rows.append(row)
            print(row, flush=True)
            continue

        env = None
        try:
            run_dir = Path(selected_run)
            ego_model, alt_model = load_models(run_dir)
            layout_config = fixed_layout_config(run_dir, layout)
            env = gym.make(layout_config["env_id"], **env_config_from_config(layout_config))
            env.add_partner_agent(StaticPolicyAgent(alt_model.policy))
            episode_rows = [run_episode(env, ego_model, deterministic) for _ in range(episodes)]
            stats = summarize(episode_rows)
            row = {**base_row, **stats}
            for episode_idx, episode_row in enumerate(episode_rows):
                detail_rows.append(
                    {
                        **base_row,
                        "episode": episode_idx,
                        "status": "ok",
                        "error_type": "",
                        "error": "",
                        **episode_row,
                    }
                )
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
            detail_rows.append(
                {
                    **base_row,
                    "episode": "",
                    "status": "error",
                    "episode_reward": None,
                    "sparse_reward": None,
                    "shaped_reward": None,
                    "soups_delivered": None,
                    "steps": None,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        finally:
            if env is not None:
                env.close()

        summary_rows.append(row)
        print(row, flush=True)

    summary_path = metrics_dir / f"{output_name}.csv"
    detail_path = metrics_dir / f"{output_name}_episodes.csv"
    summary_fields = [
        "layout",
        "selected_run",
        "selected_run_name",
        "route_source",
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
        "layout",
        "selected_run",
        "selected_run_name",
        "route_source",
        "episodes",
        "deterministic",
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
        "routes": routes,
        "default_run_dir": default_run_dir,
        "layouts": layouts,
        "episodes": episodes,
        "deterministic": deterministic,
        "output_dir": str(output_dir),
        "summary_csv": str(summary_path),
        "episode_csv": str(detail_path),
        "rows": summary_rows,
    }
    save_json(metrics_dir / f"{output_name}.json", payload)
    return payload


def main() -> None:
    args = build_parser().parse_args()
    config = load_router_config(args.config)
    routes = resolve_routes(config, args.route)
    layouts = args.layout or list(config.get("layouts", [])) or list(routes.keys()) or DEFAULT_LAYOUTS
    episodes = args.episodes or int(config.get("episodes", 20))
    deterministic = bool(config.get("deterministic_eval", True)) and not args.stochastic
    output_dir = resolve_output_dir(config, args.output_dir)
    output_name = args.output_name or str(config.get("output_name", "router_eval"))
    default_run_dir = args.default_run_dir or config.get("default_run_dir")

    router_config = {
        "routes": routes,
        "layouts": layouts,
        "episodes": episodes,
        "deterministic_eval": deterministic,
        "output_dir": str(output_dir),
        "output_name": output_name,
        "default_run_dir": default_run_dir,
    }
    save_json(output_dir / "router_config.resolved.json", router_config)
    evaluate_router(
        routes=routes,
        layouts=layouts,
        episodes=episodes,
        deterministic=deterministic,
        output_dir=output_dir,
        output_name=output_name,
        default_run_dir=default_run_dir,
    )


if __name__ == "__main__":
    main()
