
env每次reset之后不一定能恢复之前的状态 奖励在后面几次epoch中被清除了。需要先保存初始化状态 之后reset的时候再从中恢复

---

## Solution

# Solution for Issue #0: Fix-reward-reset-issue-by-saving-and-restoring-initial-environment-state

## 进度记录
- 已在 `rlf/env/base.py` 保存迷宫初始地图副本，并在 `reset` 时恢复，确保奖励点不会因上次回合被清空。
- 新增 `tests/env/test_base.py` 覆盖奖励点恢复逻辑。
