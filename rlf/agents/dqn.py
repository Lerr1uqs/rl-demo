"""
这是一个DQN算法模块，主要功能如下：
实现DQN（Deep Q-Network）
"""
import random
from typing import Optional, List, Deque
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from rlf.agents.base import BaseAgent
from rlf.schemas import (
    DQNConfig,
    AgentStats,
    Transition,
    ActionScoresData,
    AlgorithmType,
    ScoreType,
    ACTION_ORDER
)


class QNetwork(nn.Module):
    """DQN的Q网络"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent(BaseAgent):
    """DQN Agent"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[DQNConfig] = None
    ) -> None:
        super().__init__(state_dim, action_dim)

        self.config: DQNConfig = config if config else DQNConfig()

        self.q_net: QNetwork = QNetwork(
            state_dim,
            action_dim,
            self.config.hidden_dim
        )
        self.target_net: QNetwork = QNetwork(
            state_dim,
            action_dim,
            self.config.hidden_dim
        )
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer: torch.optim.Adam = torch.optim.Adam(
            self.q_net.parameters(),
            lr=self.config.learning_rate
        )

        self.replay_buffer: Deque[Transition] = deque(maxlen=self.config.buffer_size)
        self.epsilon: float = self.config.epsilon

        self.train_steps: int = 0
        self.total_steps: int = 0
        self.last_target_update_step: int = 0
        self.total_loss: float = 0.0
        self.loss_count: int = 0

    def _valid_actions(self, action_mask: Optional[List[bool]]) -> List[int]:
        if action_mask is None:
            return list(range(self.action_dim))
        assert len(action_mask) == self.action_dim
        valid_actions = [
            index for index, allowed in enumerate(action_mask) if allowed
        ]
        assert len(valid_actions) > 0
        return valid_actions

    def select_action(
        self,
        state: int,
        training: bool = True,
        action_mask: Optional[List[bool]] = None
    ) -> int:
        valid_actions = self._valid_actions(action_mask)
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)

        state_tensor: torch.Tensor = F.one_hot(
            torch.tensor(state),
            self.state_dim
        ).float()
        with torch.no_grad():
            q_values: torch.Tensor = self.q_net(state_tensor)
        if action_mask is not None:
            masked_q_values = q_values.clone()
            for index, allowed in enumerate(action_mask):
                if not allowed:
                    masked_q_values[index] = -float("inf")
            q_values = masked_q_values
        return int(q_values.argmax().item())

    def store_transition(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool
    ) -> None:
        self.total_steps += 1
        transition = Transition.model_construct(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done
        )
        self.replay_buffer.append(transition)

    def train(self) -> Optional[float]:
        return self._train_from_buffer()

    def step_train(self) -> Optional[float]:
        return self._train_from_buffer()

    def _train_from_buffer(self) -> Optional[float]:
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        # 从经验回放中采样
        batch: List[Transition] = random.sample(
            self.replay_buffer,
            self.config.batch_size
        )

        states: List[int] = [t.state for t in batch]
        actions: List[int] = [t.action for t in batch]
        rewards: List[float] = [t.reward for t in batch]
        next_states: List[int] = [t.next_state for t in batch]
        dones: List[bool] = [t.done for t in batch]

        # 转换为tensor
        states_tensor: torch.Tensor = F.one_hot(
            torch.tensor(states),
            self.state_dim
        ).float()  # shape: (states_size, state_dim)
        actions_tensor: torch.Tensor = torch.tensor(actions).long()
        rewards_tensor: torch.Tensor = torch.tensor(rewards).float()
        next_states_tensor: torch.Tensor = F.one_hot(
            torch.tensor(next_states),
            self.state_dim
        ).float()
        dones_tensor: torch.Tensor = torch.tensor(dones).float()

        # 计算当前Q值
        # self.q_net(states_tensor): shape (states_size, action_dim) 返回每一个状态的全部action的q值
        # gather: 从全部action的q值中选中对应action的q值
        current_q: torch.Tensor = self.q_net(states_tensor).gather(
            1,
            actions_tensor.unsqueeze(1)
        ).squeeze()

        # 计算目标Q值
        with torch.no_grad():
            # shape 为 (batch_size, action_dim)
            next_q_values: torch.Tensor = self.target_net(next_states_tensor)
            if self._action_mask_provider is not None:
                # 下一个状态采样的四个action的True/False掩码
                # (batch_size, action_dim)
                masks: List[List[bool]] = [
                    self._action_mask_provider(state)
                    for state in next_states
                ]
                for mask in masks:
                    assert any(mask)
                mask_tensor = torch.tensor(
                    masks,
                    dtype=next_q_values.dtype,
                    device=next_q_values.device
                )
                # Loss = MSE(Q(s, a), r + γ * (1 - d) * max(Q(s', a')))
                # Q(s', a') 也要进行位置掩码处理
                masked_q_values = next_q_values.masked_fill(
                    mask_tensor == 0,
                    -float("inf")
                )
                next_q = masked_q_values.max(1)[0]
            else:
                next_q = next_q_values.max(1)[0]
            target_q: torch.Tensor = rewards_tensor + \
                self.config.gamma * next_q * (1 - dones_tensor)

        # 计算损失
        loss: torch.Tensor = F.mse_loss(current_q, target_q)

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新目标网络
        self.train_steps += 1
        if self.total_steps - self.last_target_update_step >= \
            self.config.update_target_freq:
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.last_target_update_step = self.total_steps

        # 衰减epsilon
        self.epsilon = max(
            self.config.epsilon_min,
            self.epsilon * self.config.epsilon_decay
        )

        loss_value: float = float(loss.item())
        self.total_loss += loss_value
        self.loss_count += 1

        return loss_value

    def action_distribution(self, state: int) -> ActionScoresData:
        assert self.action_dim == len(ACTION_ORDER)
        state_tensor: torch.Tensor = F.one_hot(
            torch.tensor(state),
            self.state_dim
        ).float()
        with torch.no_grad():
            q_values: torch.Tensor = self.q_net(state_tensor)
        scores: List[float] = [float(value) for value in q_values.tolist()]
        return ActionScoresData(
            score_type=ScoreType.Q_VALUES,
            scores=scores,
            action_order=ACTION_ORDER[:]
        )

    @property
    def stats(self) -> AgentStats:
        avg_loss: float = self.total_loss / max(1, self.loss_count)
        return AgentStats(
            agent_type='Off-Policy (DQN)',
            buffer_size=len(self.replay_buffer),
            epsilon=self.epsilon,
            avg_loss=avg_loss,
            train_steps=self.train_steps,
            data_usage='∞ (experience replay)'
        )

    @property
    def policy_type(self) -> AlgorithmType:
        return AlgorithmType.DQN
