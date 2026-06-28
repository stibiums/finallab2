# Algorithm Comparison Plan

Last updated: 2026-06-29

This note turns the `random1` partner-robustness failures into a concrete
algorithm-comparison plan. It is intentionally conservative: the current project
already has a strong specialist-router story, so a new MARL stack should only be
added if it answers a specific weakness that the PPO/IPPO-style experiments
cannot answer.

## Current Baseline

The current neural training code is best described as independent PPO over the
PantheonRL simultaneous-agent interface:

- `src/overcooked_project/train.py` trains two PPO policies in self-play.
- `src/overcooked_project/train_diverse.py` trains one ego PPO policy against a
  fixed partner population, optionally with a learned partner.
- `src/overcooked_project/partner_conditioning.py` can append a one-hot partner
  id to the ego observation.

This setup is useful and reproducible, but it is not MAPPO or HARL. It has no
centralized critic over the full two-agent state, no recurrent partner-context
model, and no sequential heterogeneous-agent update scheme.

## Latest Diagnostic

The six-partner `random1` diagnostic trains a partner-id conditioned ego against
six fixed partners and holds out seed78:

- Training partners: `baseline_random1`, `baseline_random1_seed71`,
  `baseline_random1_seed72`, `baseline_random1_seed73`,
  `baseline_random1_seed76`, `baseline_random1_seed77`.
- Held-out partner: `baseline_random1_seed78`.
- Run: `outputs/runs/partner_conditioned_random1_six_partners_holdout78`.

Results:

| Evaluation | Mean soups |
| --- | ---: |
| Known partner `baseline_random1` | 2.10 |
| Known partner `baseline_random1_seed71` | 0.60 |
| Known partner `baseline_random1_seed72` | 0.25 |
| Known partner `baseline_random1_seed73` | 1.00 |
| Known partner `baseline_random1_seed76` | 0.00 |
| Known partner `baseline_random1_seed77` | 0.00 |
| Held-out seed78, best fixed assumed id | 0.00 |
| Held-out seed78, best online inferred id | 0.00 |

Interpretation: adding more fixed partners does not by itself solve partner
robustness. The next comparison should test a different representation or
training algorithm rather than only adding more seed-specific PPO partners.

## Candidate Algorithms

| Candidate | Why it is relevant | Integration cost | Suggested role |
| --- | --- | --- | --- |
| Recurrent/context-conditioned PPO | Lets the ego adapt from recent partner behavior without requiring a true partner id. | Medium; likely needs `sb3-contrib` or custom recurrent rollout handling. | Best first engineering target if we stay in the current SB3/PantheonRL stack. |
| MAPPO | Adds centralized training with decentralized execution and is a standard cooperative MARL baseline. | High; needs a multi-agent environment adapter exposing per-agent observations, actions, rewards, dones, and shared state. | Good report comparison if time allows adapter work. |
| HAPPO/HARL | Targets heterogeneous agents without parameter sharing and uses sequential agent updates. | High; requires adapting Overcooked to the HARL environment interface and runner/logger conventions. | Best as a serious stretch comparison, not a quick parameter tweak. |

## HARL Reference Value

HARL is worth citing and possibly using because Overcooked naturally has
heterogeneous roles: one teammate may specialize in onion/pot work while the
other handles dishes and delivery. The official HARL implementation describes
HAPPO/HATRPO and related algorithms as heterogeneous-agent methods that avoid
the restrictive parameter-sharing trick and use a sequential update scheme.

However, the official integration path for a new environment requires a
multi-agent interface with fields such as `n_agents`, per-agent observation and
action spaces, shared observations/state, and `step`/`reset` methods returning
multi-agent tensors. That is a real adapter project. It is not the same as
changing `PPO` to another SB3 class inside the current scripts.

Primary references:

- Official HARL repository: https://github.com/PKU-MARL/HARL
- JMLR 2024 paper: https://jmlr.org/papers/v25/23-0488.html

## Recommended Next Experiment

If more experiment time remains, use this order:

1. Build an Overcooked multi-agent adapter that can return both agents'
   observations, a shared global state vector, rewards, dones, and available
   actions.
2. Run a 50k-step smoke test with MAPPO or HAPPO on `simple` to validate the
   adapter before spending time on `random1`.
3. Run a small `random1` benchmark with the same partner-robustness protocol:
   known-partner matrix plus held-out seed76/77/78 or a leave-one-out split.
4. Compare against the current six-partner PPO result, not only against
   self-play.

Success should be defined by held-out partner soups, not rollout reward:

- Minimum meaningful signal: held-out `random1` improves from 0.00 to at least
  0.50 soups under the same 20-episode deterministic evaluation.
- Strong signal: held-out `random1` reaches at least 1.00 soups without
  sacrificing all known-partner performance.

