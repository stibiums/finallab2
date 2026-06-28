---
title: Overcooked MARL Specialist Router
theme: default
layout: cover
---

# Overcooked 多智能体协作策略研究

PPO baseline → failure analysis → specialists → layout router

---

# 任务设置

- 环境：Overcooked-AI + PantheonRL simultaneous-agent wrapper
- 智能体：两个 PPO agents，同时学习合作
- 每局 horizon：400 steps
- 主指标：`soups delivered = sparse reward / 20`
- 目标：从 baseline 出发，构建能覆盖多张 onion layouts 的实用方案

---

# 核心问题

1. PPO 能否在简单地图上学会合作？
2. Reward shaping 是否必要？
3. 单地图策略能否 zero-shot 泛化？
4. 多地图混训是否足够？
5. Specialist + router 能否更稳定？
6. `small_corridor` 这种长链条地图怎么处理？
7. 自博弈成功是否等于 partner 鲁棒？

---

# Baseline 结论

| Run | Layout | Mean soups | 结论 |
| --- | --- | ---: | --- |
| `no_shaping_simple` | `simple` | 0.00 | sparse-only 失败 |
| `baseline_simple` | `simple` | 7.50 | 默认 shaping 有效 |
| `distance_shaping_simple` | `simple` | 7.50 | 没超过默认 shaping |

结论：reward shaping 是必要条件，但简单加距离奖励不一定提升最终 sparse performance。

---

# 泛化失败

- `baseline_simple` 在 `simple` 上达到 7.50 soups。
- 迁移到 `small_corridor`、`random0`、`random1`、`unident_s` 后基本 0 sparse reward。
- 朴素多地图 curriculum 也没有解决问题，甚至损伤 `simple` 能力。

因此，当前阶段不把重点放在“单一 PPO policy 自然泛化”上。

---

# Specialist Router

![Router comparison](../docs/assets/router_comparison.svg)

---

# Router 结果

| Router | Average | Minimum | 解释 |
| --- | ---: | ---: | --- |
| PPO-only onion router | 9.23 | 5.80 | 强，但跳过 `small_corridor` |
| With 3-cycle BC `small_corridor` | 7.76 | 1.90 | 五图覆盖，但 small corridor 较弱 |
| With perturbed BC+PPO `small_corridor` | 7.98 | 3.00 | 当前最强五图覆盖 |

---

# Small Corridor 难点

![Small corridor progression](../docs/assets/small_corridor_progression.svg)

---

# Small Corridor 关键发现

- PPO from scratch：0 soups
- Structured shaping：能推进到 soup pickup，但无法上菜
- Delivery-only BC：只会最后一段，标准开局仍失败
- Full-chain BC：从标准开局做到 1 soup
- 3-cycle BC：平均 1.90 soups
- Wait-jittered BC：平均 2.50 soups
- Role-balanced BC：平均 2.40 soups，未超过固定分工 BC
- Subtask router：BC-only 到 2.60，但会把最佳 BC+PPO 降到 2.45
- Role-specific router：只解释了 delivery 多由 partner 持汤触发，未超过最佳策略
- Perturbed BC + PPO 25k checkpoint：稳定 3.00 soups
- PPO 50k final：退化到 0.85 soups

结论：checkpoint selection 是必要的。

---

# Partner Robustness

![Partner robustness](../docs/assets/partner_robustness.svg)

---

# Partner 结论

- Self-play 分数不能代表真实鲁棒性。
- `random1` self-play 有 5.80 soups，但 held-out partner 几乎归零。
- 两 partner 的 `partner_diversity_random1` 改善 seen partners，但 held-out seed72 仍只有 0.45 soups。
- 三 partner 训练把 seen-partner minimum 提到 0.65，但 held-out seed73 仍只有 1.00 soups。
- 固定三 partner + learned partner 混训平均只有 0.69 soups，是负结果。
- Partner-id conditioning 提升到 known avg 2.34 / min 0.80，但 unknown seed76 仍为 0。
- `unident_s` 在测试 seed 间更稳定。
- 报告中必须区分 self-play success 与 partner robustness。

---

# 最终主张

1. PPO + default shaping 能解决 `simple`。
2. Sparse-only、zero-shot、naive multi-layout 都失败。
3. Single-layout specialists 能解决多个 hard layouts。
4. Layout router 是当前最强 practical method。
5. `small_corridor` 需要 scripted BC、perturbation 和 checkpoint selection；简单 role-balanced BC 不够。
6. 后续重点是 stronger partner-conditioned/HARL-style 训练、learned option routing 和统一策略蒸馏。

---

# Demo 顺序

1. `baseline_simple/demo/demo.gif`
2. `no_shaping_simple/demo/demo.gif`
3. `multi_layout_curriculum/demo/simple_demo.gif`
4. `baseline_random0_long/demo/random0_long_demo.gif`
5. `baseline_random1/demo/random1_demo.gif`
6. `baseline_unident_s/demo/unident_s_demo.gif`
7. `small_corridor_full_chain_3cycle_jitter3_bc_ppo_best_demo.gif`

---

# 结论

当前项目最有说服力的结果不是“一个 PPO 策略泛化到所有地图”，而是：

> baseline works on simple → sparse / zero-shot / multi-layout fail → specialists solve hard layouts → router composes specialists → small corridor needs BC + checkpoint selection

这条链条能诚实解释失败，也能给出清晰改进。
