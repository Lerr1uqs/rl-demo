"""
这是一个传统Q-Learning算法模块，主要功能如下：
实现表格型Q-Learning
"""
import random
from typing import Optional, List

from rlf.agents.base import BaseAgent
from rlf.schemas import (
    TabularConfig,
    AgentStats,
    ActionScoresData,
    AlgorithmType,
    ScoreType,
    Transition,
    ACTION_ORDER
)
from pydantic import BaseModel, Field


class QTable(BaseModel):
    """Q表数据"""
    values: List[List[float]] = Field(default_factory=list)


class QLearningAgent(BaseAgent):
    """Q-Learning Agent（表格型）"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[TabularConfig] = None
    ) -> None:
        """
        初始化Q-Learning Agent。
        args:
            state_dim (int): 状态空间维度。
            action_dim (int): 动作空间维度。
            config (Optional[TabularConfig]): Q表学习配置。
        """
        super().__init__(state_dim, action_dim)

        self.config: TabularConfig = config if config else TabularConfig()

        self.q_table = QTable(
            values=[
                [self.config.initial_q for _ in range(action_dim)]
                for _ in range(state_dim)
            ]
        )
        self.epsilon: float = self.config.epsilon
        self._pending_transition: Optional[Transition] = None

        self.train_steps: int = 0
        self.total_td_error: float = 0.0
        self.update_count: int = 0

    def _valid_actions(self, action_mask: Optional[List[bool]]) -> List[int]:
        if action_mask is None:
            return list(range(self.action_dim))
        assert len(action_mask) == self.action_dim
        valid_actions = [
            index for index, allowed in enumerate(action_mask) if allowed
        ]
        assert len(valid_actions) > 0
        return valid_actions

    def _best_actions(self, state: int, valid_actions: List[int]) -> List[int]:
        q_values = self.q_table.values[state]
        max_q = max(q_values[action] for action in valid_actions)
        return [
            action for action in valid_actions if q_values[action] == max_q
        ]

    def _action_mask_for_state(self, state: int) -> Optional[List[bool]]:
        if self._action_mask_provider is None:
            return None
        mask = self._action_mask_provider(state)
        assert len(mask) == self.action_dim
        assert any(mask)
        return mask

    def select_action(
        self,
        state: int,
        training: bool = True,
        action_mask: Optional[List[bool]] = None
    ) -> int:
        """选择动作（epsilon-greedy）"""
        valid_actions = self._valid_actions(action_mask)
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)

        best_actions = self._best_actions(state, valid_actions)
        return random.choice(best_actions)

    def store_transition(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool
    ) -> None:
        """存储转换"""
        self._pending_transition = Transition.model_construct(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done
        )

    def train(self) -> Optional[float]:
        """episode级训练（表格型默认不使用）"""
        return None

    def step_train(self) -> Optional[float]:
        """step级训练"""
        if self._pending_transition is None:
            return None

        transition = self._pending_transition
        self._pending_transition = None

        current_q = self.q_table.values[transition.state][transition.action]
        if transition.done:
            target = transition.reward
        else:
            next_mask = self._action_mask_for_state(transition.next_state)
            next_valid_actions = self._valid_actions(next_mask)
            next_best_actions = self._best_actions(
                transition.next_state,
                next_valid_actions
            )
            next_best_action = random.choice(next_best_actions)
            next_q = self.q_table.values[transition.next_state][next_best_action]
            target = transition.reward + self.config.gamma * next_q

        td_error = target - current_q
        updated_q = current_q + self.config.learning_rate * td_error
        self.q_table.values[transition.state][transition.action] = updated_q

        self.train_steps += 1
        self.total_td_error += abs(td_error)
        self.update_count += 1

        if transition.done:
            # NOTE: 每一个episode结束时更新epsilon 而不是每一个step
            self.episode_count += 1
            self.epsilon = max(
                self.config.epsilon_min,
                self.epsilon * self.config.epsilon_decay
            )

        return abs(td_error)

    def action_distribution(self, state: int) -> ActionScoresData:
        """返回当前状态下动作分布"""
        assert self.action_dim == len(ACTION_ORDER)
        scores = [float(value) for value in self.q_table.values[state]]
        return ActionScoresData(
            score_type=ScoreType.Q_VALUES,
            scores=scores,
            action_order=ACTION_ORDER[:]
        )

    @property
    def stats(self) -> AgentStats:
        """统计信息"""
        avg_loss = self.total_td_error / max(1, self.update_count)
        return AgentStats(
            agent_type='Off-Policy (Q-Learning)',
            epsilon=self.epsilon,
            avg_loss=avg_loss,
            train_steps=self.train_steps,
            data_usage='1x (online)'
        )

    @property
    def policy_type(self) -> AlgorithmType:
        """算法类型"""
        return AlgorithmType.QLEARN
