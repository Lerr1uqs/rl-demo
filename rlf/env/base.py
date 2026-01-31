"""
这是一个环境基类模块，主要功能如下：
定义迷宫环境，包括状态空间、动作空间、奖励机制等
"""
from typing import List, Tuple
from rlf.schemas import StepResult, StepInfo


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
        raise ValueError("迷宫地图中没有找到起始位置(R)")

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
        moves: List[Tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        next_pos: List[int] = [
            self.agent_pos[0] + moves[action][0],
            self.agent_pos[1] + moves[action][1]
        ]

        # 检查边界
        if not (0 <= next_pos[0] < self.height and 0 <= next_pos[1] < self.width):
            info = StepInfo.model_construct(hit='boundary')
            return StepResult(
                state=self._get_state(),
                reward=-5,
                done=False,
                info=info
            )

        # 检查墙壁
        cell: str = self.maze_map[next_pos[0]][next_pos[1]]
        if cell == 'W':
            info = StepInfo.model_construct(hit='wall')
            return StepResult(
                state=self._get_state(),
                reward=-2,
                done=False,
                info=info
            )

        # 移动到新位置
        self.agent_pos = next_pos

        # 计算奖励
        reward: float = 0.0
        done: bool = False
        info = StepInfo.model_construct()

        if cell == 'R':
            reward = -0.1  # 小惩罚鼓励快速到达
            info.hit = 'road'
        elif cell == 'T':
            reward = -10.0
            info.hit = 'trap'
        elif cell == 'B':
            reward = 15.0
            info.hit = 'bonus'
            self.maze_map[next_pos[0]][next_pos[1]] = 'R'  # 奖励只能拿一次
        elif cell == 'G':
            reward = 200.0
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
