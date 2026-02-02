"""
这是迷宫环境单元测试模块，主要功能如下：
验证 reset 能恢复奖励点的初始状态
"""
from __future__ import annotations

import unittest

from rlf.env.base import MazeEnv


class TestMazeEnvReset(unittest.TestCase):
    """迷宫环境 reset 行为测试"""

    def test_reset_restores_bonus(self) -> None:
        """奖励点被取走后 reset 需要恢复"""
        maze_map = ["RB"]
        env = MazeEnv(maze_map)

        step = env.step(3)
        self.assertEqual(step.reward, 20.0)
        self.assertEqual(env.maze_map[0][1], "R")

        env.reset()
        self.assertEqual(env.maze_map[0][1], "B")
