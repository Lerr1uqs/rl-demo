"""
这是一个训练器模块，主要功能如下：
实现可插拔的训练框架，支持多种Agent的训练和演示
"""
import time
import numpy as np
from typing import Optional

from rlf.env.base import MazeEnv
from rlf.agents.base import BaseAgent
from rlf.schemas import TrainingResult, AgentStats, StepResult
from rlf.data_export import TrainingDataSaver


class MazeTrainer:
    """可插拔的训练框架"""

    def __init__(self, env: MazeEnv, agent: BaseAgent, save_data: bool = True) -> None:
        self.env: MazeEnv = env
        self.agent: BaseAgent = agent
        self.episode_rewards: list[float] = []
        self.episode_steps: list[int] = []
        self.save_data = save_data
        self.data_saver = None
        if save_data:
            self.data_saver = TrainingDataSaver()

    def train(
        self,
        num_episodes: int = 500,
        print_freq: int = 50,
        render_freq: int = 100
    ) -> TrainingResult:
        print(f"\n{'='*60}")
        print(f"🚀 开始训练: {self.agent.stats.agent_type}")
        print(f"{'='*60}\n")

        start_time: float = time.time()

        for episode in range(num_episodes):
            state: int = self.env.reset()
            episode_reward: float = 0.0
            episode_steps: int = 0
            done: bool = False
            if self.data_saver:
                self.data_saver.record_initial_state(
                    episode=episode,
                    state=state,
                    maze_state=[row[:] for row in self.env.maze_map],
                    agent_pos=list(self.env.agent_pos)
                )

            # 收集一个episode的数据
            while not done:
                action: int = self.agent.select_action(state, training=True)
                step_result: StepResult = self.env.step(action)

                # 保存步骤数据
                if self.data_saver:
                    self.data_saver.record_step(
                        episode=episode,
                        step=episode_steps + 1,
                        state=state,
                        action=action,
                        reward=step_result.reward,
                        maze_state=[row[:] for row in self.env.maze_map],  # 深拷贝
                        agent_pos=list(self.env.agent_pos),
                        info=step_result.info,
                        cumulative_reward=episode_reward + step_result.reward
                    )

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

            agent_stats: AgentStats = self.agent.stats

            # 完成episode记录
            if self.data_saver:
                self.data_saver.finalize_episode(
                    episode=episode,
                    total_reward=episode_reward,
                    total_steps=episode_steps,
                    success=episode_reward > 50,
                    loss=loss,
                    agent_stats=agent_stats
                )

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

                print(f"\n📈 Agent Stats:")
                self._print_stats(agent_stats)
                print(f"{'─'*60}")

            # 渲染
            if (episode + 1) % render_freq == 0:
                # print(f"\n🎮 Episode {episode + 1} 演示:")
                # self.demo(render=True)
                pass

        # 保存数据
        if self.data_saver:
            agent_name = self.agent.stats.agent_type.replace(' ', '_')
            self.data_saver.save(agent_name)

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

        return total_reward
