"""
这是算法单元测试模块，主要功能如下：
验证DQN/PG/PPO/Q-Learning/SARSA的基础行为与数据导出流程可用
"""
from __future__ import annotations

import unittest
from pathlib import Path

from rlf.agents.dqn import DQNAgent
from rlf.agents.pg import PGAgent
from rlf.agents.ppo import PPOAgent
from rlf.agents.qlearn import QLearningAgent
from rlf.agents.sarsa import SarsaAgent
from rlf.data_export import TrainingDataSaver
from rlf.schemas import (
    DQNConfig,
    TrainingConfig,
    PPOConfig,
    TabularConfig,
    AlgorithmType,
    ScoreType,
    ActionScoresData,
    StepInfo,
    TrainingSessionData,
    ACTION_ORDER
)

class TestAlgorithmsTiny(unittest.TestCase):
    """算法基础行为的tiny测试"""

    def test_qlearning_step_update(self) -> None:
        """Q-Learning 单步更新可运行且返回loss"""
        config = TabularConfig(
            learning_rate=1.0,
            gamma=0.99,
            epsilon=0.0,
            epsilon_decay=1.0,
            epsilon_min=0.0,
            initial_q=0.0
        )
        agent = QLearningAgent(state_dim=3, action_dim=2, config=config)
        agent.store_transition(
            state=0,
            action=1,
            reward=5.0,
            next_state=1,
            done=True
        )
        loss = agent.step_train()
        self.assertIsNotNone(loss)
        self.assertEqual(agent.q_table.values[0][1], 5.0)

    def test_sarsa_step_update(self) -> None:
        """SARSA 单步更新可运行且返回loss"""
        config = TabularConfig(
            learning_rate=1.0,
            gamma=0.99,
            epsilon=0.0,
            epsilon_decay=1.0,
            epsilon_min=0.0,
            initial_q=0.0
        )
        agent = SarsaAgent(state_dim=3, action_dim=2, config=config)
        agent.store_transition(
            state=0,
            action=0,
            reward=3.0,
            next_state=2,
            done=True
        )
        loss = agent.step_train()
        self.assertIsNotNone(loss)
        self.assertEqual(agent.q_table.values[0][0], 3.0)

    def test_dqn_step_update(self) -> None:
        """DQN 单步更新可运行（经验回放不足时返回None）"""
        config = DQNConfig(
            learning_rate=0.01,
            gamma=0.99,
            hidden_dim=8,
            buffer_size=10,
            batch_size=1,
            epsilon=0.0,
            epsilon_decay=1.0,
            epsilon_min=0.0,
            update_target_freq=1
        )
        agent = DQNAgent(state_dim=3, action_dim=2, config=config)
        agent.store_transition(
            state=0,
            action=1,
            reward=1.0,
            next_state=1,
            done=False
        )
        loss = agent.step_train()
        self.assertIsNotNone(loss)

    def test_pg_episode_update(self) -> None:
        """PG episode 更新可运行并返回loss"""
        config = TrainingConfig(
            learning_rate=0.01,
            gamma=0.99,
            hidden_dim=8
        )
        agent = PGAgent(state_dim=3, action_dim=2, config=config)
        agent.store_transition(
            state=0,
            action=1,
            reward=1.0,
            next_state=1,
            done=True
        )
        loss = agent.train()
        self.assertIsNotNone(loss)

    def test_ppo_episode_update(self) -> None:
        """PPO episode 更新可运行并返回loss"""
        config = PPOConfig(
            learning_rate=0.01,
            gamma=0.99,
            hidden_dim=8,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            ppo_epochs=1
        )
        agent = PPOAgent(state_dim=3, action_dim=2, config=config)
        agent.store_transition(
            state=0,
            action=1,
            reward=1.0,
            next_state=1,
            done=True
        )
        loss = agent.train()
        self.assertIsNotNone(loss)

    def test_export_data_schema(self) -> None:
        """训练导出包含必要字段"""
        action_scores = ActionScoresData(
            score_type=ScoreType.Q_VALUES,
            scores=[0.0 for _ in ACTION_ORDER],
            action_order=ACTION_ORDER[:]
        )
        step_info = StepInfo(hit="road", timeout=False)

        tmp_root = Path(__file__).resolve().parents[2] / "tmp" / "tests"
        tmp_root.mkdir(parents=True, exist_ok=True)
        export_dir = tmp_root / "export"
        export_dir.mkdir(parents=True, exist_ok=True)

        saver = TrainingDataSaver(
            policy_type=AlgorithmType.QLEARN,
            save_dir=str(export_dir)
        )
        saver.record_initial_state(
            episode=0,
            state=0,
            maze_state=[["R"]],
            agent_pos=[0, 0],
            action_scores=action_scores
        )
        saver.record_step(
            episode=0,
            step=1,
            state=0,
            action=0,
            reward=1.0,
            maze_state=[["R"]],
            agent_pos=[0, 0],
            info=step_info,
            cumulative_reward=1.0,
            action_scores=action_scores
        )
        saver.finalize_episode(
            episode=0,
            total_reward=1.0,
            total_steps=1,
            success=True,
            loss=0.1
        )
        filename = saver.save("TinyAgent")

        with open(filename, "r", encoding="utf-8") as f:
            raw = f.read()

        session = TrainingSessionData.model_validate_json(raw)
        self.assertEqual(session.total_episodes, 1)
        self.assertEqual(session.type, AlgorithmType.QLEARN)
        self.assertEqual(len(session.episodes), 1)
        self.assertEqual(session.episodes[0].steps[0].step, 0)
