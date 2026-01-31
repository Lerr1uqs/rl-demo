---
Create Date: 2026-02-01
Type: feat
---

不是 DQN 不适合走迷宫，是你的"盲人版" DQN 不适合。
迷宫是 DQN 的经典 benchmark（比如 DeepMind 早年的很多迷宫实验），但有一个前提：Agent 得看得见墙。
为什么你现在觉得"DQN 不适合"？
你现在遇到的所有痛苦（震荡、撞墙、Q 值虚高），本质不是 DQN 的锅，是"部分可观测"（Partial Observability）的锅。
假设你用表格 Q-learning（Tabular Q-Learning）：
如果状态只是 (x, y) 坐标 → 同样会撞墙震荡，和 DQN 一模一样
如果状态是 (x, y, 上是否墙, 下是否墙, 左是否墙, 右是否墙) → 表格法 100 个 episode 就能收敛
DQN 只是表格法的神经网络近似，如果输入信息不足（只看坐标），它当然比表格法还惨（神经网络会过拟合、会震荡）。
DQN 走迷宫的正确打开方式
迷宫任务中，DQN 要 work，必须满足以下至少一条：
1. 完全可观测（Full Observability）——给地图
Python
复制
# 把地图作为 CNN 输入（就像人类玩家看屏幕）
state = maze_grid  # shape: (H, W, channels)
# walls=1, road=0, agent=0.5, goal=0.8
这是 DQN 的舒适区，收敛很稳。
2. 完全可观测 + 动作掩码（Action Masking）
Python
复制
# 即使只用坐标，但在输出层 mask 掉不能走的动作
q_values = network(coord)
q_values[wall_directions] = -inf  # 彻底禁止
这样网络虽然看不见墙，但永远不会选择撞墙动作（因为被 mask 了），剩下的就是正常路径规划。

所以要为agent添加迷宫环境感知 知道哪里不能走