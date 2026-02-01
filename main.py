"""
这是一个RL框架的主程序入口文件
主要功能：演示如何使用RL框架训练不同的算法
"""
from rich import print
from rich.traceback import Traceback
from rich.console import Console
from loguru import logger
from typing import List
from enum import StrEnum
import typer

from rlf import (
    MazeEnv,
    DQNAgent,
    PGAgent,
    PPOAgent,
    MazeTrainer,
    DQNConfig,
    PPOConfig
)
from rlf.agents.base import BaseAgent
from rlf.schemas import TrainingConfig


app = typer.Typer(add_completion=False)


class SupportedAlgorithm(StrEnum):
    """支持的算法类型"""
    DQN = "dqn"
    PG = "pg"
    PPO = "ppo"


@app.command()
def main(
    algorithm: SupportedAlgorithm = typer.Option(
        SupportedAlgorithm.DQN,
        "--algorithm",
        "-a",
        help="选择要训练的算法: dqn/pg/ppo"
    ),
    export_dir: str = typer.Option(
        "./training_data",
        "--export-dir",
        "-o",
        help="训练数据导出目录"
    )
) -> None:
    """主函数"""
    assert export_dir.strip()

    # 定义迷宫地图
    maze_map: List[str] = [
        "RWWWWWWWW",
        "RRRRTRRRR",
        "WRWRWWRWG",
        "WRRRBRRRR",
        "WWWWWWWWW",
    ]

    # 创建环境
    env = MazeEnv(maze_map)
    print(f"🌍 环境创建成功: {env.height}x{env.width} 迷宫")
    print(f"   状态空间: {env.state_space}")
    print(f"   动作空间: {env.action_space}\n")

    # 选择要训练的算法
    print("🎯 选择要训练的算法:")
    print("   dqn: DQN (Off-Policy)")
    print("   pg: Policy Gradient (On-Policy)")
    print("   ppo: PPO (On-Policy with Limited Reuse)")
    print(f"   当前选择: {algorithm.value}")

    agent: BaseAgent
    if algorithm == SupportedAlgorithm.DQN:
        # 创建DQN Agent
        dqn_config = DQNConfig(
            learning_rate=0.001,
            gamma=0.99,
            hidden_dim=128,
            buffer_size=50000,
            batch_size=64,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            update_target_freq=100
        )
        agent = DQNAgent(
            state_dim=env.state_space,
            action_dim=env.action_space,
            config=dqn_config
        )
    elif algorithm == SupportedAlgorithm.PG:
        # 创建Policy Gradient Agent
        pg_config = TrainingConfig(
            learning_rate=0.001,
            gamma=0.99,
            hidden_dim=128
        )
        agent = PGAgent(
            state_dim=env.state_space,
            action_dim=env.action_space,
            config=pg_config
        )
    elif algorithm == SupportedAlgorithm.PPO:
        # 创建PPO Agent
        ppo_config = PPOConfig(
            learning_rate=0.001,
            gamma=0.99,
            hidden_dim=128,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            ppo_epochs=4
        )
        agent = PPOAgent(
            state_dim=env.state_space,
            action_dim=env.action_space,
            config=ppo_config
        )
    else:
        raise ValueError("无效的选择")

    # 创建训练器
    trainer = MazeTrainer(
        env,
        agent,
        save_data=True,
        save_dir=export_dir
    )

    # 训练
    result = trainer.train(
        num_episodes=500,
        print_freq=50,
        render_freq=100
    )

    print("\n" + "="*60)
    print("🎉 训练完成！")
    print("="*60)
    print(f"📊 训练结果:")
    print(f"   总时间: {result.total_time:.2f}s")
    print(f"   最佳奖励: {result.best_reward:.2f}")
    print(f"   最后50轮平均奖励: {result.final_avg_reward:.2f}")
    print("="*60)

    trainer.plot_training_curves()

    # 最终演示
    print("\n🎮 最终演示:")
    # trainer.demo(render=True)


if __name__ == "__main__":
    console = Console()

    try:
        app()
    except Exception as e:
        t = Traceback.from_exception(type(e), e, e.__traceback__)
        with console.capture() as capture:
            console.print(t)
        rich_output = capture.get()
        logger.info("\n" + rich_output)
