"""
这是一个数据导出模块，主要功能如下：
导出训练数据给TUI使用，包含数据保存和schema定义
"""
import json
import os
from typing import List, Optional
from datetime import datetime

from rlf.schemas import StepInfoData, StepData, EpisodeData, AgentStats, StepInfo


class TrainingDataSaver:
    """训练数据保存器"""

    def __init__(self, save_dir: str = "./training_data") -> None:
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.episodes_data: List[EpisodeData] = []

    def _add_step_data(self, episode: int, step_data: StepData) -> None:
        """添加步数据并确保episode结构存在"""
        if episode >= len(self.episodes_data):
            self.episodes_data.append(
                EpisodeData(
                    episode=episode,
                    steps=[],
                    total_reward=0.0,
                    total_steps=0,
                    success=False
                )
            )
        self.episodes_data[episode].steps.append(step_data)

    def record_initial_state(
        self,
        episode: int,
        state: int,
        maze_state: List[List[str]],
        agent_pos: List[int]
    ) -> None:
        """记录episode的初始状态"""
        step_info_data = StepInfoData(
            hit="",
            timeout=False
        )

        step_data = StepData(
            step=0,
            state=state,
            action=-1,
            action_name="INIT",
            reward=0.0,
            cumulative_reward=0.0,
            agent_pos=agent_pos,
            maze_state=maze_state,
            info=step_info_data
        )

        self._add_step_data(episode, step_data)

    def record_step(
        self,
        episode: int,
        step: int,
        state: int,
        action: int,
        reward: float,
        maze_state: List[List[str]],
        agent_pos: List[int],
        info: StepInfo,
        cumulative_reward: float
    ) -> None:
        """记录单步数据"""
        action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']

        step_info_data = StepInfoData(
            hit=info.hit if hasattr(info, 'hit') else "",
            timeout=info.timeout if hasattr(info, 'timeout') else False
        )

        step_data = StepData(
            step=step,
            state=state,
            action=action,
            action_name=action_names[action],
            reward=reward,
            cumulative_reward=cumulative_reward,
            agent_pos=agent_pos,
            maze_state=maze_state,
            info=step_info_data
        )

        self._add_step_data(episode, step_data)

    def finalize_episode(
        self,
        episode: int,
        total_reward: float,
        total_steps: int,
        success: bool,
        loss: Optional[float] = None,
        agent_stats: Optional[AgentStats] = None
    ) -> None:
        """完成一个episode的记录"""
        if episode < len(self.episodes_data):
            episode_data = self.episodes_data[episode]
            episode_data.total_reward = total_reward
            episode_data.total_steps = total_steps
            episode_data.success = success
            episode_data.loss = loss
            episode_data.agent_stats = agent_stats
        else:
            raise RuntimeError(f"Episode {episode} is unreachable.")
        

    def save(self, agent_name: str) -> str:
        """保存到文件"""
        filename = f"{self.save_dir}/{self.session_id}_{agent_name}.json"

        data = {
            "session_id": self.session_id,
            "agent_name": agent_name,
            "timestamp": datetime.now().isoformat(),
            "total_episodes": len(self.episodes_data),
            "episodes": [episode.model_dump() for episode in self.episodes_data]
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\n💾 训练数据已保存到: {filename}")
        return filename
