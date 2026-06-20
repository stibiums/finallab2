"""Utilities for the Overcooked MARL course project."""

from .envs import register_envs

register_envs()

__all__ = ["register_envs"]
