"""
这是一个数据模型定义模块，主要功能如下：
定义RL框架中使用的所有Pydantic数据模型，包括环境步进结果、Agent统计信息、训练结果等
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class StepInfo(BaseModel):
    """步进信息"""
    hit: str = ""
    timeout: bool = False


class StepResult(BaseModel):
    """环境步进返回结果"""
    state: int
    reward: float
    done: bool
    info: StepInfo = Field(default_factory=StepInfo)

    class Config:
        arbitrary_types_allowed = True


class TrainingConfig(BaseModel):
    """训练配置"""
    learning_rate: float = Field(default=0.001, description="学习率")
    gamma: float = Field(default=0.99, description="折扣因子")
    hidden_dim: int = Field(default=128, description="隐藏层维度")


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


class AgentStats(BaseModel):
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


class TrainingResult(BaseModel):
    """训练结果"""
    episode_rewards: List[float]
    episode_steps: List[int]
    total_time: float
    best_reward: float
    final_avg_reward: float


class ComparisonResult(BaseModel):
    """对比结果"""
    algorithm_name: str
    avg_last_50: float
    max_reward: float
    success_rate: float


class StepInfoData(BaseModel):
    """步进信息数据"""
    hit: str
    timeout: bool


class StepData(BaseModel):
    """步数据"""
    step: int
    state: int
    action: int
    action_name: str
    reward: float
    cumulative_reward: float
    agent_pos: List[int]
    maze_state: List[List[str]]
    info: StepInfoData


class EpisodeData(BaseModel):
    """Episode数据"""
    episode: int
    steps: List[StepData]
    total_reward: float
    total_steps: int
    success: bool
    loss: Optional[float] = None
    agent_stats: Optional[AgentStats] = None


class Transition(BaseModel):
    """转换数据"""
    state: int
    action: int
    reward: float
    next_state: int
    done: bool


class PPOTransition(BaseModel):
    """PPO转换数据（包含旧策略的log_prob和value）"""
    state: int
    action: int
    reward: float
    old_log_prob: float
    value: float
