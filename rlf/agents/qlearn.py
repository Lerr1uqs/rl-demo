"""
这是一个Q-Learning算法模块，主要功能如下：
实现DQN（Deep Q-Network）和Policy Gradient算法
"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Deque
from collections import deque

from rlf.agents.base import BaseAgent
from rlf.schemas import (
    DQNConfig,
    TrainingConfig,
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


class PolicyNetwork(nn.Module):
    """策略网络"""

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
        return F.softmax(self.net(x), dim=-1)


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
        self.total_loss: float = 0.0
        self.loss_count: int = 0

    def select_action(self, state: int, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_tensor: torch.Tensor = F.one_hot(
            torch.tensor(state),
            self.state_dim
        ).float()
        with torch.no_grad():
            q_values: torch.Tensor = self.q_net(state_tensor)
        return int(q_values.argmax().item())

    def store_transition(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool
    ) -> None:
        transition = Transition.model_construct(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done
        )
        self.replay_buffer.append(transition)

    def train(self) -> Optional[float]:
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
        ).float()
        actions_tensor: torch.Tensor = torch.tensor(actions).long()
        rewards_tensor: torch.Tensor = torch.tensor(rewards).float()
        next_states_tensor: torch.Tensor = F.one_hot(
            torch.tensor(next_states),
            self.state_dim
        ).float()
        dones_tensor: torch.Tensor = torch.tensor(dones).float()

        # 计算当前Q值
        current_q: torch.Tensor = self.q_net(states_tensor).gather(
            1,
            actions_tensor.unsqueeze(1)
        ).squeeze()

        # 计算目标Q值
        with torch.no_grad():
            next_q: torch.Tensor = self.target_net(next_states_tensor).max(1)[0]
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
        if self.train_steps % self.config.update_target_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

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


class PGAgent(BaseAgent):
    """Policy Gradient Agent"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[TrainingConfig] = None
    ) -> None:
        super().__init__(state_dim, action_dim)

        self.config: TrainingConfig = config if config else TrainingConfig()

        self.policy_net: PolicyNetwork = PolicyNetwork(
            state_dim,
            action_dim,
            self.config.hidden_dim
        )
        self.optimizer: torch.optim.Adam = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=self.config.learning_rate
        )

        self.episode_data: List[tuple] = []
        self.total_loss: float = 0.0
        self.loss_count: int = 0

    def select_action(self, state: int, training: bool = True) -> int:
        state_tensor: torch.Tensor = F.one_hot(
            torch.tensor(state),
            self.state_dim
        ).float()
        probs: torch.Tensor = self.policy_net(state_tensor)

        if training:
            dist = torch.distributions.Categorical(probs)
            action: torch.Tensor = dist.sample()
            return int(action.item())
        else:
            return int(probs.argmax().item())

    def store_transition(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool
    ) -> None:
        self.episode_data.append((state, action, reward))

    def train(self) -> Optional[float]:
        if len(self.episode_data) == 0:
            return None

        states: List[int] = [t[0] for t in self.episode_data]
        actions: List[int] = [t[1] for t in self.episode_data]
        rewards: List[float] = [t[2] for t in self.episode_data]

        # 计算回报
        returns: List[float] = []
        G: float = 0.0
        for r in reversed(rewards):
            G = r + self.config.gamma * G
            returns.insert(0, G)

        # 标准化回报
        returns_tensor: torch.Tensor = torch.tensor(returns).float()
        returns_tensor = (returns_tensor - returns_tensor.mean()) / \
            (returns_tensor.std() + 1e-8)

        # 计算策略梯度
        states_tensor: torch.Tensor = F.one_hot(
            torch.tensor(states),
            self.state_dim
        ).float()
        actions_tensor: torch.Tensor = torch.tensor(actions).long()

        probs: torch.Tensor = self.policy_net(states_tensor)
        log_probs: torch.Tensor = torch.log(
            probs.gather(1, actions_tensor.unsqueeze(1)).squeeze() + 1e-8
        )

        loss: torch.Tensor = -(log_probs * returns_tensor).mean()

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_value: float = float(loss.item())
        self.total_loss += loss_value
        self.loss_count += 1

        # On-Policy: 用完即丢
        self.episode_data.clear()

        return loss_value

    def action_distribution(self, state: int) -> ActionScoresData:
        assert self.action_dim == len(ACTION_ORDER)
        state_tensor: torch.Tensor = F.one_hot(
            torch.tensor(state),
            self.state_dim
        ).float()
        with torch.no_grad():
            probs: torch.Tensor = self.policy_net(state_tensor)
        scores: List[float] = [float(value) for value in probs.tolist()]
        return ActionScoresData(
            score_type=ScoreType.POLICY_PROBS,
            scores=scores,
            action_order=ACTION_ORDER[:]
        )

    @property
    def stats(self) -> AgentStats:
        avg_loss: float = self.total_loss / max(1, self.loss_count)
        return AgentStats(
            agent_type='On-Policy (PG)',
            current_buffer=len(self.episode_data),
            avg_loss=avg_loss,
            data_usage='1x (immediate discard)'
        )

    @property
    def policy_type(self) -> AlgorithmType:
        return AlgorithmType.PG
