"""
这是一个Policy Gradient算法模块，主要功能如下：
实现最基础的Policy Gradient（REINFORCE）
"""
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from rlf.agents.base import BaseAgent
from rlf.schemas import (
    TrainingConfig,
    AgentStats,
    ActionScoresData,
    AlgorithmType,
    ScoreType,
    ACTION_ORDER
)


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

    def _apply_action_mask(
        self,
        probs: torch.Tensor,
        action_mask: List[bool]
    ) -> torch.Tensor:
        assert len(action_mask) == self.action_dim
        mask_tensor = torch.tensor(action_mask, dtype=probs.dtype, device=probs.device)
        masked_probs = probs * mask_tensor
        total = masked_probs.sum()
        assert float(total.item()) > 0
        return masked_probs / total

    def select_action(
        self,
        state: int,
        training: bool = True,
        action_mask: Optional[List[bool]] = None
    ) -> int:
        state_tensor: torch.Tensor = F.one_hot(
            torch.tensor(state),
            self.state_dim
        ).float()
        probs: torch.Tensor = self.policy_net(state_tensor)
        if action_mask is not None:
            probs = self._apply_action_mask(probs, action_mask)

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
