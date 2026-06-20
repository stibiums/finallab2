# Report Outline

建议报告题目：

基于奖励塑形的 Overcooked 双智能体协作策略研究

## 1. Introduction

说明 Overcooked 是一个完全合作的双智能体任务，两个智能体需要在有限时间内分工完成取洋葱、下锅、取盘、盛汤、上菜等子任务。本文关注 reward shaping 对协作学习效率和最终策略行为的影响。

## 2. Method

- Environment: Overcooked `simple` layout.
- Algorithm: PPO self-play with two learning agents.
- Observation: Overcooked-AI featurized state.
- Action space: six discrete actions: move up/down/left/right, stay, interact.
- Reward variants:
  - no shaping
  - default intermediate shaping
  - additional distance shaping

## 3. Experiments

建议至少跑：

- `no_shaping_simple`
- `baseline_simple`
- `distance_shaping_simple`

每个实验保存训练曲线、模型、评估 JSON/CSV 和 GIF。

## 4. Results

推荐图表：

- episode reward training curve
- mean sparse reward / soups delivered table
- behavior GIF screenshots
- failed coordination examples

## 5. Discussion

可以重点分析：

- 稀疏奖励下为什么探索难
- 中间奖励如何帮助形成子任务链条
- 过强的 dense reward 是否会造成局部最优
- 两个 PPO agent 同时学习时的非平稳性
