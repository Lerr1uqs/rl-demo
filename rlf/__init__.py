"""
这是一个RL框架包初始化文件
主要功能：提供强化学习框架的核心模块
"""
from rlf.env import MazeEnv
from rlf.agents import (
    BaseAgent,
    DQNAgent,
    PGAgent,
    PPOAgent,
    QLearningAgent,
    SarsaAgent
)
from rlf.trainer import MazeTrainer
from rlf.schemas import (
    TrainingConfig,
    DQNConfig,
    TabularConfig,
    PPOConfig,
    AgentStats,
    TrainingResult,
    ComparisonResult
)

__version__ = "0.1.0"
__all__ = [
    'MazeEnv',
    'BaseAgent',
    'DQNAgent',
    'PGAgent',
    'PPOAgent',
    'QLearningAgent',
    'SarsaAgent',
    'MazeTrainer',
    'TrainingConfig',
    'DQNConfig',
    'TabularConfig',
    'PPOConfig',
    'AgentStats',
    'TrainingResult',
    'ComparisonResult'
]
