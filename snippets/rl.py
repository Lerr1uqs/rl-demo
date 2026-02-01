# import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from typing import List, Tuple, Optional, Deque, Any
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
import time
from rich import print
from rlf.schemas import ACTION_DELTAS, ACTION_ORDER, ACTION_SYMBOLS

# ==================== 数据类定义 ====================

@dataclass
class StepResult:
    """环境步进返回结果"""
    state: int
    reward: float
    done: bool
    info: 'StepInfo'


@dataclass
class StepInfo:
    """步进信息"""
    hit: str = ""
    timeout: bool = False


@dataclass
class Transition:
    """转换数据"""
    state: int
    action: int
    reward: float
    next_state: int
    done: bool


@dataclass
class PPOTransition:
    """PPO转换数据（包含旧策略的log_prob和value）"""
    state: int
    action: int
    reward: float
    old_log_prob: float
    value: float


class TrainingConfig(BaseModel):
    """训练配置"""
    learning_rate: float = Field(default=0.001, description="学习率")
    gamma: float = Field(default=0.99, description="折扣因子")
    hidden_dim: int = Field(default=128, description="隐藏层维度")
    
    class Config:
        arbitrary_types_allowed = True


class DQNConfig(TrainingConfig):
    """DQN配置"""
    buffer_size: int = Field(default=10000, description="经验回放缓冲区大小")
    batch_size: int = Field(default=64, description="批量大小")
    epsilon: float = Field(default=1.0, description="初始探索率")
    epsilon_decay: float = Field(default=0.995, description="探索率衰减")
    epsilon_min: float = Field(default=0.01, description="最小探索率")
    update_target_freq: int = Field(default=100, description="目标网络更新频率")


class PPOConfig(TrainingConfig):
    """PPO配置"""
    gae_lambda: float = Field(default=0.95, description="GAE lambda")
    clip_epsilon: float = Field(default=0.2, description="PPO裁剪参数")
    ppo_epochs: int = Field(default=4, description="PPO训练轮数")


@dataclass
class AgentStats:
    """Agent统计信息"""
    agent_type: str
    buffer_size: int = 0
    epsilon: float = 0.0
    avg_loss: float = 0.0
    train_steps: int = 0
    current_buffer: int = 0
    ppo_epochs: int = 0
    avg_policy_loss: float = 0.0
    avg_value_loss: float = 0.0
    total_reuses: int = 0
    data_usage: str = ""


@dataclass
class TrainingResult:
    """训练结果"""
    episode_rewards: List[float]
    episode_steps: List[int]
    total_time: float
    best_reward: float
    final_avg_reward: float


@dataclass
class ComparisonResult:
    """对比结果"""
    algorithm_name: str
    avg_last_50: float
    max_reward: float
    success_rate: float


# ==================== 迷宫环境 ====================

class MazeEnv:
    """
    迷宫环境定义：
    R = Road (可通行，奖励0)
    T = Trap (陷阱，奖励-10)
    W = Wall (墙壁，不可通行)
    G = Goal (目标，奖励+100，终止)
    B = Bonus (奖励点，奖励+10)
    """
    
    def __init__(self, maze_map: List[str]) -> None:
        self.maze_map: List[List[str]] = [list(row) for row in maze_map]
        self.height: int = len(self.maze_map)
        self.width: int = len(self.maze_map[0])
        
        # 找到起始位置（第一个R或第一个非W位置）
        self.start_pos: List[int] = self._find_start()
        self.agent_pos: List[int] = list(self.start_pos)
        
        # 动作空间：上、下、左、右
        self.action_space: int = 4
        # 状态空间：位置编码
        self.state_space: int = self.height * self.width
        
        self.step_count: int = 0
        self.max_steps: int = 200
        
    def _find_start(self) -> List[int]:
        """找到起始位置"""
        for i in range(self.height):
            for j in range(self.width):
                if self.maze_map[i][j] == 'R':
                    return [i, j]
        return [0, 0]
    
    def reset(self) -> int:
        """重置环境"""
        self.agent_pos = list(self.start_pos)
        self.step_count = 0
        return self._get_state()
    
    def _get_state(self) -> int:
        """获取当前状态（位置编码）"""
        return self.agent_pos[0] * self.width + self.agent_pos[1]
    
    def step(self, action: int) -> StepResult:
        """执行动作"""
        self.step_count += 1
        
        # 动作映射：0=上, 1=下, 2=左, 3=右
        moves = [
            ACTION_DELTAS[ord] 
            for ord in ACTION_ORDER
        ]
        next_pos: List[int] = [
            self.agent_pos[0] + moves[action][0],
            self.agent_pos[1] + moves[action][1]
        ]
        
        # 检查边界
        if not (0 <= next_pos[0] < self.height and 0 <= next_pos[1] < self.width):
            info = StepInfo(hit='boundary')
            return StepResult(
                state=self._get_state(),
                reward=-5.0,
                done=False,
                info=info
            )
        
        # 检查墙壁
        cell: str = self.maze_map[next_pos[0]][next_pos[1]]
        if cell == 'W':
            info = StepInfo(hit='wall')
            return StepResult(
                state=self._get_state(),
                reward=-2.0,
                done=False,
                info=info
            )
        
        # 移动到新位置
        self.agent_pos = next_pos
        
        # 计算奖励
        reward: float = 0.0
        done: bool = False
        info = StepInfo()
        
        if cell == 'R':
            reward = -1  # 小惩罚鼓励快速到达
            info.hit = 'road'
        elif cell == 'T':
            reward = -10.0
            info.hit = 'trap'
        elif cell == 'B':
            reward = 15.0
            info.hit = 'bonus'
            self.maze_map[next_pos[0]][next_pos[1]] = 'R'  # 奖励只能拿一次
        elif cell == 'G':
            reward = 100.0
            done = True
            info.hit = 'goal'
        
        # 检查是否超时
        if self.step_count >= self.max_steps:
            done = True
            info.timeout = True
        
        return StepResult(
            state=self._get_state(),
            reward=reward,
            done=done,
            info=info
        )
    
    def render(self) -> None:
        """渲染迷宫"""
        print("\n" + "="*40)
        for i in range(self.height):
            row: str = ""
            for j in range(self.width):
                if [i, j] == self.agent_pos:
                    row += "🤖 "
                else:
                    cell: str = self.maze_map[i][j]
                    symbols: List[Tuple[str, str]] = [
                        ('R', '⬜'), ('T', '💥'), ('W', '⬛'),
                        ('G', '🎯'), ('B', '💎')
                    ]
                    symbol: str = cell
                    for c, s in symbols:
                        if cell == c:
                            symbol = s
                            break
                    row += symbol + " "
            print(row)
        print("="*40)


# ==================== 神经网络 ====================

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


# ==================== 基础Agent接口 ====================

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
    
    def get_stats(self) -> AgentStats:
        """获取统计信息"""
        raise NotImplementedError


# ==================== DQN Agent (Off-Policy) ====================

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
        transition = Transition(
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
    
    def get_stats(self) -> AgentStats:
        avg_loss: float = self.total_loss / max(1, self.loss_count)
        return AgentStats(
            agent_type='Off-Policy (DQN)',
            buffer_size=len(self.replay_buffer),
            epsilon=self.epsilon,
            avg_loss=avg_loss,
            train_steps=self.train_steps,
            data_usage='∞ (experience replay)'
        )


# ==================== Policy Gradient Agent (On-Policy) ====================

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
        
        self.episode_data: List[Tuple[int, int, float]] = []
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
        
        # ✅ On-Policy: 用完即丢
        self.episode_data.clear()
        
        return loss_value
    
    def get_stats(self) -> AgentStats:
        avg_loss: float = self.total_loss / max(1, self.loss_count)
        return AgentStats(
            agent_type='On-Policy (PG)',
            current_buffer=len(self.episode_data),
            avg_loss=avg_loss,
            data_usage='1x (immediate discard)'
        )


# ==================== PPO Agent (On-Policy with Limited Reuse) ====================

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
        
        transition = PPOTransition(
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
        
        # ⚠️ PPO: 多轮训练（有限重用数据）
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
        
        # ⚠️ PPO: 重用后仍需清空
        self.episode_data.clear()
        
        return avg_policy_loss
    
    def get_stats(self) -> AgentStats:
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


# ==================== 训练框架 ====================

class MazeTrainer:
    """可插拔的训练框架"""
    
    def __init__(self, env: MazeEnv, agent: BaseAgent) -> None:
        self.env: MazeEnv = env
        self.agent: BaseAgent = agent
        self.episode_rewards: List[float] = []
        self.episode_steps: List[int] = []
        
    def train(
        self, 
        num_episodes: int = 500, 
        print_freq: int = 50, 
        render_freq: int = 100
    ) -> TrainingResult:
        print(f"\n{'='*60}")
        print(f"🚀 开始训练: {self.agent.get_stats().agent_type}")
        print(f"{'='*60}\n")
        
        start_time: float = time.time()
        
        for episode in range(num_episodes):
            state: int = self.env.reset()
            episode_reward: float = 0.0
            episode_steps: int = 0
            done: bool = False
            
            # 收集一个episode的数据
            while not done:
                action: int = self.agent.select_action(state, training=True)
                step_result: StepResult = self.env.step(action)
                
                self.agent.store_transition(
                    state, 
                    action, 
                    step_result.reward, 
                    step_result.state, 
                    step_result.done
                )
                
                state = step_result.state
                episode_reward += step_result.reward
                episode_steps += 1
                done = step_result.done
            
            # 训练
            loss: Optional[float] = self.agent.train()
            
            self.episode_rewards.append(episode_reward)
            self.episode_steps.append(episode_steps)
            
            # 打印训练信息
            if (episode + 1) % print_freq == 0:
                avg_reward: float = float(
                    np.mean(self.episode_rewards[-print_freq:])
                )
                avg_steps: float = float(
                    np.mean(self.episode_steps[-print_freq:])
                )
                elapsed_time: float = time.time() - start_time
                
                print(f"\n{'─'*60}")
                print(f"📊 Episode {episode + 1}/{num_episodes}")
                print(f"{'─'*60}")
                print(f"⏱️  Time: {elapsed_time:.2f}s")
                print(f"🎯 Avg Reward (last {print_freq}): {avg_reward:.2f}")
                print(f"👣 Avg Steps: {avg_steps:.2f}")
                print(f"📉 Loss: {loss:.4f}" if loss else "📉 Loss: warming up...")
                
                stats: AgentStats = self.agent.get_stats()
                print(f"\n📈 Agent Stats:")
                self._print_stats(stats)
                print(f"{'─'*60}")
            
            # 渲染
            if (episode + 1) % render_freq == 0:
                print(f"\n🎮 Episode {episode + 1} 演示:")
                self.demo(render=True)
        
        total_time: float = time.time() - start_time
        best_reward: float = float(max(self.episode_rewards))
        final_avg: float = float(np.mean(self.episode_rewards[-50:]))
        
        print(f"\n{'='*60}")
        print(f"✅ 训练完成!")
        print(f"⏱️  Total Time: {total_time:.2f}s")
        print(f"🏆 Best Reward: {best_reward:.2f}")
        print(f"📊 Final Avg Reward (last 50): {final_avg:.2f}")
        print(f"{'='*60}\n")
        
        return TrainingResult(
            episode_rewards=self.episode_rewards,
            episode_steps=self.episode_steps,
            total_time=total_time,
            best_reward=best_reward,
            final_avg_reward=final_avg
        )
    
    def _print_stats(self, stats: AgentStats) -> None:
        """打印统计信息"""
        if stats.buffer_size > 0:
            print(f"   buffer_size: {stats.buffer_size}")
        if stats.epsilon > 0:
            print(f"   epsilon: {stats.epsilon:.3f}")
        if stats.avg_loss > 0:
            print(f"   avg_loss: {stats.avg_loss:.4f}")
        if stats.train_steps > 0:
            print(f"   train_steps: {stats.train_steps}")
        if stats.current_buffer > 0:
            print(f"   current_buffer: {stats.current_buffer}")
        if stats.ppo_epochs > 0:
            print(f"   ppo_epochs: {stats.ppo_epochs}")
        if stats.avg_policy_loss > 0:
            print(f"   avg_policy_loss: {stats.avg_policy_loss:.4f}")
        if stats.avg_value_loss > 0:
            print(f"   avg_value_loss: {stats.avg_value_loss:.4f}")
        if stats.total_reuses > 0:
            print(f"   total_reuses: {stats.total_reuses}")
        if stats.data_usage:
            print(f"   data_usage: {stats.data_usage}")
    
    def demo(self, render: bool = True) -> float:
        """演示训练好的agent"""
        state: int = self.env.reset()
        done: bool = False
        total_reward: float = 0.0
        steps: int = 0
        
        if render:
            self.env.render()
        
        while not done and steps < 50:
            action: int = self.agent.select_action(state, training=False)
            step_result: StepResult = self.env.step(action)
            state = step_result.state
            total_reward += step_result.reward
            steps += 1
            done = step_result.done
            
            if render:
                time.sleep(0.1)
                self.env.render()
                print(
                    f"Action: {ACTION_SYMBOLS[ACTION_ORDER[action]]}, "
                    f"Reward: {step_result.reward:.1f}, "
                    f"Hit: {step_result.info.hit}"
                )
        
        print(f"\n🎯 Demo Result: Reward={total_reward:.2f}, Steps={steps}")
        return total_reward


# ==================== 主程序 ====================

def main() -> None:
    # 定义迷宫
    maze_design: List[str] = [
        "RWWWWWWWW",
        "RRRRTRRWG",
        "WBWWWRRWW",
        "WRRRWRRTR",
        "WRWRWWWRR",
        "WRRRRRWRB",
        "WRRWWRRRR",
    ]
    
    print("\n" + "="*60)
    print("🎮 迷宫走位 - On-Policy vs Off-Policy 对比实验")
    print("="*60)
    print("\n迷宫图例:")
    print("  ⬜ R = Road (可通行)")
    print("  💥 T = Trap (陷阱, -10)")
    print("  ⬛ W = Wall (墙壁)")
    print("  🎯 G = Goal (目标, +100)")
    print("  💎 B = Bonus (奖励, +10)")
    print("  🤖 = Agent")
    
    # 创建环境
    env: MazeEnv = MazeEnv(maze_design)
    env.render()
    
    # 训练参数
    num_episodes: int = 300
    
    # ==================== 实验1: DQN (Off-Policy) ====================
    print("\n\n" + "🔵"*30)
    print("实验 1: DQN (Off-Policy)")
    print("🔵"*30)
    
    env1: MazeEnv = MazeEnv(maze_design)
    dqn_config: DQNConfig = DQNConfig(learning_rate=0.001)
    agent1: DQNAgent = DQNAgent(env1.state_space, env1.action_space, dqn_config)
    trainer1: MazeTrainer = MazeTrainer(env1, agent1)
    result1: TrainingResult = trainer1.train(
        num_episodes=num_episodes, 
        print_freq=50, 
        render_freq=150
    )

    import pdb; pdb.set_trace()
    
    # ==================== 实验2: Policy Gradient (On-Policy) ====================
    print("\n\n" + "🟢"*30)
    print("实验 2: Policy Gradient (On-Policy)")
    print("🟢"*30)
    
    env2: MazeEnv = MazeEnv(maze_design)
    pg_config: TrainingConfig = TrainingConfig(learning_rate=0.001)
    agent2: PGAgent = PGAgent(env2.state_space, env2.action_space, pg_config)
    trainer2: MazeTrainer = MazeTrainer(env2, agent2)
    result2: TrainingResult = trainer2.train(
        num_episodes=num_episodes, 
        print_freq=50, 
        render_freq=150
    )
    
    # ==================== 实验3: PPO (On-Policy with Limited Reuse) ====================
    print("\n\n" + "🟡"*30)
    print("实验 3: PPO (On-Policy with Limited Reuse)")
    print("🟡"*30)
    
    env3: MazeEnv = MazeEnv(maze_design)
    ppo_config: PPOConfig = PPOConfig(learning_rate=0.0003)
    agent3: PPOAgent = PPOAgent(env3.state_space, env3.action_space, ppo_config)
    trainer3: MazeTrainer = MazeTrainer(env3, agent3)
    result3: TrainingResult = trainer3.train(
        num_episodes=num_episodes, 
        print_freq=50, 
        render_freq=150
    )
    
    # ==================== 最终对比 ====================
    print("\n\n" + "="*80)
    print("🏁 最终对比")
    print("="*80)
    
    results: List[Tuple[str, TrainingResult]] = [
        ("DQN (Off-Policy)", result1),
        ("PG (On-Policy)", result2),
        ("PPO (On-Policy)", result3)
    ]
    
    comparison_results: List[ComparisonResult] = []
    
    for name, result in results:
        avg_reward: float = float(np.mean(result.episode_rewards[-50:]))
        max_reward: float = float(max(result.episode_rewards))
        success_rate: float = float(
            sum(1 for r in result.episode_rewards[-50:] if r > 50) / 50 * 100
        )
        
        comparison_results.append(ComparisonResult(
            algorithm_name=name,
            avg_last_50=avg_reward,
            max_reward=max_reward,
            success_rate=success_rate
        ))
    
    print(f"\n{'Algorithm':<25} {'Avg Last 50':<15} {'Max Reward':<15} {'Success Rate':<15}")
    print("-"*80)
    
    for comp_result in comparison_results:
        print(
            f"{comp_result.algorithm_name:<25} "
            f"{comp_result.avg_last_50:>10.2f}     "
            f"{comp_result.max_reward:>10.2f}     "
            f"{comp_result.success_rate:>10.1f}%"
        )
    
    print("="*80)
    
    print("\n\n" + "📚"*30)
    print("关键差异总结")
    print("📚"*30)
    
    stats1: AgentStats = agent1.get_stats()
    stats2: AgentStats = agent2.get_stats()
    stats3: AgentStats = agent3.get_stats()
    
    print("\n🔵 DQN (Off-Policy):")
    print(f"   ✅ 使用经验回放，数据效率高")
    print(f"   ✅ 可以重复使用历史数据数千次")
    print(f"   ✅ Buffer Size: {stats1.buffer_size}")
    print(f"   ✅ Total Train Steps: {stats1.train_steps}")
    
    print("\n🟢 Policy Gradient (On-Policy):")
    print(f"   ⚠️  只使用当前策略的新数据")
    print(f"   ⚠️  数据用完即丢，数据效率低")
    print(f"   ⚠️  {stats2.data_usage}")
    
    print("\n🟡 PPO (On-Policy with Limited Reuse):")
    print(f"   ⭐ 通过重要性采样实现有限重用")
    print(f"   ⭐ 每批数据重用 {stats3.ppo_epochs} 次")
    print(f"   ⭐ 总重用次数: {stats3.total_reuses}")
    print(f"   ⭐ {stats3.data_usage}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
