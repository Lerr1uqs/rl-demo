"""
这是一个Agent基类模块，主要功能如下：
定义可插拔Agent基类接口
"""
from typing import Optional
from rlf.schemas import AgentStats, ActionScoresData, AlgorithmType


class BaseAgent:
    """可插拔Agent基类"""

    def __init__(self, state_dim: int, action_dim: int) -> None:
        self.state_dim: int = state_dim
        self.action_dim: int = action_dim
        self.episode_count: int = 0

    def select_action(self, state: int, training: bool = True) -> int:
        """选择动作"""
        raise NotImplementedError

    def store_transition(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool
    ) -> None:
        """存储转换"""
        raise NotImplementedError

    def train(self) -> Optional[float]:
        """训练"""
        raise NotImplementedError

    def action_distribution(self, state: int) -> ActionScoresData:
        """返回当前状态下动作分布"""
        raise NotImplementedError

    @property
    def stats(self) -> AgentStats:
        """统计信息"""
        raise NotImplementedError

    @property
    def policy_type(self) -> AlgorithmType:
        """算法类型"""
        raise NotImplementedError
