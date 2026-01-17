"""
这是一个PPO算法模块，主要功能如下：
实现PPO（Proximal Policy Optimization）算法
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from rlf.agents.base import BaseAgent
from rlf.agents.qlearn import PolicyNetwork
from rlf.schemas import PPOConfig, AgentStats, PPOTransition


class ValueNetwork(nn.Module):
    """价值网络（用于Actor-Critic）"""

    def __init__(self, state_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PPOAgent(BaseAgent):
    """PPO Agent"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[PPOConfig] = None
    ) -> None:
        super().__init__(state_dim, action_dim)

        self.config: PPOConfig = config if config else PPOConfig()

        self.policy_net: PolicyNetwork = PolicyNetwork(
            state_dim,
            action_dim,
            self.config.hidden_dim
        )
        self.value_net: ValueNetwork = ValueNetwork(
            state_dim,
            self.config.hidden_dim
        )

        self.policy_optimizer: torch.optim.Adam = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=self.config.learning_rate
        )
        self.value_optimizer: torch.optim.Adam = torch.optim.Adam(
            self.value_net.parameters(),
            lr=self.config.learning_rate
        )

        self.episode_data: List[PPOTransition] = []
        self.total_policy_loss: float = 0.0
        self.total_value_loss: float = 0.0
        self.loss_count: int = 0
        self.reuse_count: int = 0

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
        state_tensor: torch.Tensor = F.one_hot(
            torch.tensor(state),
            self.state_dim
        ).float()

        with torch.no_grad():
            probs: torch.Tensor = self.policy_net(state_tensor)
            old_log_prob: torch.Tensor = torch.log(probs[action] + 1e-8)
            value: torch.Tensor = self.value_net(state_tensor)

        transition = PPOTransition.model_construct(
            state=state,
            action=action,
            reward=reward,
            old_log_prob=float(old_log_prob.item()),
            value=float(value.item())
        )
        self.episode_data.append(transition)

    def train(self) -> Optional[float]:
        if len(self.episode_data) == 0:
            return None

        states: List[int] = [t.state for t in self.episode_data]
        actions: List[int] = [t.action for t in self.episode_data]
        rewards: List[float] = [t.reward for t in self.episode_data]
        old_log_probs: List[float] = [t.old_log_prob for t in self.episode_data]
        values: List[float] = [t.value for t in self.episode_data]

        # 计算优势函数
        returns: List[float] = []
        advantages: List[float] = []
        G: float = 0.0
        A: float = 0.0

        for i in reversed(range(len(rewards))):
            G = rewards[i] + self.config.gamma * G
            delta: float = rewards[i] + self.config.gamma * \
                (values[i+1] if i+1 < len(values) else 0.0) - values[i]
            A = delta + self.config.gamma * self.config.gae_lambda * A

            returns.insert(0, G)
            advantages.insert(0, A)

        # 转换为tensor
        states_tensor: torch.Tensor = F.one_hot(
            torch.tensor(states),
            self.state_dim
        ).float()
        actions_tensor: torch.Tensor = torch.tensor(actions).long()
        returns_tensor: torch.Tensor = torch.tensor(returns).float()
        advantages_tensor: torch.Tensor = torch.tensor(advantages).float()
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / \
            (advantages_tensor.std() + 1e-8)
        old_log_probs_tensor: torch.Tensor = torch.tensor(old_log_probs).float()

        # PPO: 多轮训练（有限重用数据）
        total_policy_loss: float = 0.0
        total_value_loss: float = 0.0

        for epoch in range(self.config.ppo_epochs):
            # 计算当前策略的log_prob
            probs: torch.Tensor = self.policy_net(states_tensor)
            log_probs: torch.Tensor = torch.log(
                probs.gather(1, actions_tensor.unsqueeze(1)).squeeze() + 1e-8
            )

            # 计算重要性采样比率
            ratio: torch.Tensor = torch.exp(log_probs - old_log_probs_tensor)

            # PPO裁剪
            surr1: torch.Tensor = ratio * advantages_tensor
            surr2: torch.Tensor = torch.clamp(
                ratio,
                1 - self.config.clip_epsilon,
                1 + self.config.clip_epsilon
            ) * advantages_tensor
            policy_loss: torch.Tensor = -torch.min(surr1, surr2).mean()

            # 价值函数损失
            values_pred: torch.Tensor = self.value_net(states_tensor).squeeze()
            value_loss: torch.Tensor = F.mse_loss(values_pred, returns_tensor)

            # 优化策略网络
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # 优化价值网络
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())
            self.reuse_count += 1

        avg_policy_loss: float = total_policy_loss / self.config.ppo_epochs
        avg_value_loss: float = total_value_loss / self.config.ppo_epochs

        self.total_policy_loss += avg_policy_loss
        self.total_value_loss += avg_value_loss
        self.loss_count += 1

        # PPO: 重用后仍需清空
        self.episode_data.clear()

        return avg_policy_loss

    @property
    def stats(self) -> AgentStats:
        avg_policy_loss: float = self.total_policy_loss / max(1, self.loss_count)
        avg_value_loss: float = self.total_value_loss / max(1, self.loss_count)
        return AgentStats(
            agent_type='On-Policy (PPO)',
            current_buffer=len(self.episode_data),
            ppo_epochs=self.config.ppo_epochs,
            avg_policy_loss=avg_policy_loss,
            avg_value_loss=avg_value_loss,
            total_reuses=self.reuse_count,
            data_usage=f'{self.config.ppo_epochs}x (limited reuse)'
        )
