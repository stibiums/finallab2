from __future__ import annotations

from collections.abc import Mapping, Sequence

import gym
import numpy as np


def partner_conditioning_config(config: Mapping) -> dict:
    raw = config.get("partner_id_conditioning", False)
    if isinstance(raw, Mapping):
        return {
            "enabled": bool(raw.get("enabled", False)),
            "num_partners": raw.get("num_partners"),
            "unknown_partner_id": raw.get("unknown_partner_id"),
        }
    return {"enabled": bool(raw), "num_partners": None, "unknown_partner_id": None}


def append_partner_id(obs, num_partners: int, partner_id: int) -> np.ndarray:
    if num_partners <= 0:
        raise ValueError("num_partners must be positive")
    if partner_id < 0 or partner_id >= num_partners:
        raise ValueError(f"partner_id {partner_id} outside [0, {num_partners})")
    base = np.asarray(obs, dtype=np.float32).reshape(-1)
    partner_one_hot = np.zeros(int(num_partners), dtype=np.float32)
    partner_one_hot[int(partner_id)] = 1.0
    return np.concatenate([base, partner_one_hot]).astype(np.float32)


class PartnerIDConditioningWrapper(gym.Wrapper):
    """Append a one-hot partner id to the ego observation.

    During training, the id is read from PantheonRL's ``env.partnerids`` after
    each reset. During matrix evaluation, ``fixed_partner_id`` can pin the
    one-hot id for a single loaded partner model.
    """

    def __init__(self, env: gym.Env, num_partners: int, fixed_partner_id: int | None = None):
        super().__init__(env)
        if num_partners <= 0:
            raise ValueError("num_partners must be positive")
        self.num_partners = int(num_partners)
        self.fixed_partner_id = fixed_partner_id

        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("PartnerIDConditioningWrapper only supports Box observations")
        low = np.concatenate(
            [
                np.asarray(env.observation_space.low, dtype=np.float32),
                np.zeros(self.num_partners, dtype=np.float32),
            ]
        )
        high = np.concatenate(
            [
                np.asarray(env.observation_space.high, dtype=np.float32),
                np.ones(self.num_partners, dtype=np.float32),
            ]
        )
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _partner_id(self) -> int:
        if self.fixed_partner_id is not None:
            partner_id = int(self.fixed_partner_id)
        else:
            partnerids = getattr(self.env, "partnerids", [0])
            partner_id = int(partnerids[0])
        if partner_id < 0 or partner_id >= self.num_partners:
            raise ValueError(f"partner_id {partner_id} outside [0, {self.num_partners})")
        return partner_id

    def _augment(self, obs) -> np.ndarray:
        return append_partner_id(obs, self.num_partners, self._partner_id())

    def reset(self, **kwargs):
        return self._augment(self.env.reset(**kwargs))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._augment(obs), reward, done, info


def maybe_wrap_partner_id_conditioning(
    env: gym.Env,
    config: Mapping,
    partner_run_dirs: Sequence | None = None,
    fixed_partner_run_dir=None,
) -> gym.Env:
    conditioning = partner_conditioning_config(config)
    if not conditioning["enabled"]:
        return env

    configured_num_partners = conditioning.get("num_partners")
    if configured_num_partners is not None:
        num_partners = int(configured_num_partners)
    elif partner_run_dirs is not None:
        num_partners = len(partner_run_dirs)
    else:
        num_partners = len(getattr(env, "partners", [[]])[0])

    fixed_partner_id = None
    if fixed_partner_run_dir is not None:
        partner_names = [getattr(path, "name", str(path)) for path in (partner_run_dirs or [])]
        fixed_name = getattr(fixed_partner_run_dir, "name", str(fixed_partner_run_dir))
        if fixed_name in partner_names:
            fixed_partner_id = partner_names.index(fixed_name)
        elif conditioning.get("unknown_partner_id") is not None:
            fixed_partner_id = int(conditioning["unknown_partner_id"])
        else:
            raise ValueError(
                f"Partner {fixed_name!r} is not in conditioned partner pool {partner_names!r}"
            )

    return PartnerIDConditioningWrapper(
        env,
        num_partners=num_partners,
        fixed_partner_id=fixed_partner_id,
    )
