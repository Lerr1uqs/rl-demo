"""
这是一个Agent模块包初始化文件
"""
from rlf.agents.base import BaseAgent
from rlf.agents.qlearn import DQNAgent, PGAgent
from rlf.agents.ppo import PPOAgent

__all__ = ['BaseAgent', 'DQNAgent', 'PGAgent', 'PPOAgent']