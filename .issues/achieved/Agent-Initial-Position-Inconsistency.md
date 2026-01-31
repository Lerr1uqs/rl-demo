
agent有几次初始位置不一样

---

## Solution

# Solution for Issue #1: Agent-Initial-Position-Inconsistency

## 分析
回放数据的第一步是从`env.step`之后记录的，所以展示的是「执行第一步动作后的位置」，而不是`reset`后的起始位置。由于第一步动作存在随机探索，导致回放里“初始位置”看起来不一致。

## 修复
在训练开始后、首次行动前，新增记录初始状态的步骤（action=-1, action_name=INIT）。随后每一步动作记录的step索引从1开始，确保回放第0步始终对应真实起始位置。
同时将训练数据缓存改为使用Pydantic模型管理，避免用dict维护复杂结构。
