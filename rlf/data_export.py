"""
这是一个数据导出模块，主要功能如下：
导出训练数据给TUI使用，包含数据保存和schema定义
"""
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

from rlf.schemas import StepInfoData, StepData, EpisodeData


class TrainingDataSaver:
    """训练数据保存器"""

    def __init__(self, save_dir: str = "./training_data") -> None:
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.episodes_data: List[Dict[str, Any]] = []

    def record_step(
        self,
        episode: int,
        step: int,
        state: int,
        action: int,
        reward: float,
        maze_state: List[List[str]],
        agent_pos: List[int],
        info: Any,
        cumulative_reward: float
    ) -> None:
        """记录单步数据"""
        if episode >= len(self.episodes_data):
            self.episodes_data.append({
                "episode": episode,
                "steps": [],
                "total_reward": 0.0,
                "total_steps": 0,
                "success": False
            })

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

        self.episodes_data[episode]["steps"].append(step_data.model_dump())

    def finalize_episode(
        self,
        episode: int,
        total_reward: float,
        total_steps: int,
        success: bool,
        loss: Optional[float] = None,
        agent_stats: Optional[Dict[str, Any]] = None
    ) -> None:
        """完成一个episode的记录"""
        if episode < len(self.episodes_data):
            self.episodes_data[episode].update({
                "total_reward": total_reward,
                "total_steps": total_steps,
                "success": success,
                "loss": loss,
                "agent_stats": agent_stats
            })

    def save(self, agent_name: str) -> str:
        """保存到文件"""
        filename = f"{self.save_dir}/{self.session_id}_{agent_name}.json"

        data = {
            "session_id": self.session_id,
            "agent_name": agent_name,
            "timestamp": datetime.now().isoformat(),
            "total_episodes": len(self.episodes_data),
            "episodes": self.episodes_data
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\n💾 训练数据已保存到: {filename}")
        return filename
