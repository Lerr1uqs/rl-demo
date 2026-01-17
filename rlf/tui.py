"""
训练数据回放TUI应用
使用Textual实现训练数据的可视化回放
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, Container
from textual.widgets import (
    Header,
    Footer,
    Static,
    DataTable,
    Label,
    Button,
    ProgressBar
)
from textual.reactive import reactive
from textual import on
from textual.binding import Binding


class MazeDisplay(Static):
    """迷宫显示组件"""

    def __init__(self, maze_state: List[List[str]], agent_pos: List[int], **kwargs) -> None:
        super().__init__(**kwargs)
        self.maze_state = maze_state
        self.agent_pos = agent_pos
        self.update_display()

    def update_display(self) -> None:
        """更新迷宫显示"""
        if not self.maze_state:
            self.update("无迷宫数据")
            return

        # 符号映射
        symbols = {
            'R': '⬜',  # Road
            'T': '💥',  # Trap
            'W': '⬛',  # Wall
            'G': '🎯',  # Goal
            'B': '💎'   # Bonus
        }
        agent_symbol = '🤖'

        # 创建迷宫的文本表示
        maze_text = []
        for y, row in enumerate(self.maze_state):
            row_text = ""
            for x, cell in enumerate(row):
                # 如果是agent位置，显示Agent
                if x == self.agent_pos[1] and y == self.agent_pos[0]:
                    row_text += agent_symbol
                else:
                    # 根据格子类型显示对应符号
                    row_text += symbols.get(cell, cell)
            maze_text.append(row_text)

        self.update("\n".join(maze_text))


class EpisodeInfo(Static):
    """Episode信息显示组件"""

    def __init__(self, episode_data: Dict[str, Any], **kwargs) -> None:
        super().__init__(**kwargs)
        self.episode_data = episode_data
        self.update_info()

    def update_info(self) -> None:
        """更新episode信息"""
        if not self.episode_data:
            self.update("无Episode数据")
            return

        info_text = f"""
[bold cyan]Episode {self.episode_data['episode']}[/bold cyan]

[bold]基本信息:[/bold]
  总奖励: {self.episode_data.get('total_reward', 0):.2f}
  总步数: {self.episode_data.get('total_steps', 0)}
  成功: {'[green]✓[/green]' if self.episode_data.get('success', False) else '[red]✗[/red]'}

[bold]Agent统计:[/bold]:
"""
        agent_stats = self.episode_data.get('agent_stats', {})
        if agent_stats:
            info_text += f"  Agent类型: {agent_stats.get('agent_type', 'N/A')}\n"
            info_text += f"  Buffer大小: {agent_stats.get('buffer_size', 0)}\n"
            info_text += f"  Epsilon: {agent_stats.get('epsilon', 0):.4f}\n"
            if 'avg_loss' in agent_stats:
                info_text += f"  平均损失: {agent_stats['avg_loss']:.4f}\n"
            if 'avg_policy_loss' in agent_stats:
                info_text += f"  策略损失: {agent_stats['avg_policy_loss']:.4f}\n"
                info_text += f"  价值损失: {agent_stats['avg_value_loss']:.4f}\n"

        self.update(info_text)


class StepInfo(Static):
    """Step信息显示组件"""

    def __init__(self, step_data: Dict[str, Any], **kwargs) -> None:
        super().__init__(**kwargs)
        self.step_data = step_data
        self.update_info()

    def update_info(self) -> None:
        """更新step信息"""
        if not self.step_data:
            self.update("无Step数据")
            return

        info_text = f"""
[bold cyan]Step {self.step_data['step']}[/bold cyan]

[bold]动作信息:[/bold]
  动作: {self.step_data.get('action', -1)}
  动作名称: {self.step_data.get('action_name', 'N/A')}

[bold]奖励信息:[/bold]
  即时奖励: {self.step_data.get('reward', 0):.2f}
  累计奖励: {self.step_data.get('cumulative_reward', 0):.2f}

[bold]状态信息:[/bold]
  状态: {self.step_data.get('state', 0)}
  Agent位置: ({self.step_data.get('agent_pos', [0, 0])[0]}, {self.step_data.get('agent_pos', [0, 0])[1]})

[bold]附加信息:[/bold]
"""
        step_info = self.step_data.get('info', {})
        if step_info.get('hit'):
            info_text += f"  命中: {step_info['hit']}\n"
        if step_info.get('timeout'):
            info_text += "  超时: [red]是[/red]\n"

        self.update(info_text)


class TrainingReplayApp(App):
    """训练数据回放应用"""

    CSS = """
    Screen {
        layout: vertical;
    }
    
    #main-container {
        height: 1fr;
    }
    
    #left-panel {
        width: 30%;
        dock: left;
    }
    
    #center-panel {
        width: 40%;
    }
    
    #right-panel {
        width: 30%;
    }
    
    MazeDisplay {
        height: 1fr;
        background: $panel;
        padding: 1;
        border: solid $primary;
    }
    
    EpisodeInfo, StepInfo {
        height: 1fr;
        background: $panel;
        padding: 1;
        border: solid $primary;
    }
    
    #status-bar {
        height: 3;
        background: $surface;
        padding: 1;
    }
    
    #controls-hint {
        text-align: center;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("up", "prev_episode", "上一个Episode"),
        Binding("down", "next_episode", "下一个Episode"),
        Binding("left", "prev_step", "上一个Step"),
        Binding("right", "next_step", "下一个Step"),
        Binding("q", "quit", "退出"),
    ]

    current_episode: reactive[int] = reactive(0)
    current_step: reactive[int] = reactive(0)

    def __init__(self, data_file: str) -> None:
        super().__init__()
        self.data_file = data_file
        self.data: Dict[str, Any] = {}
        self.episodes: List[Dict[str, Any]] = []
        self.session_info: Dict[str, Any] = {}

    def load_data(self) -> None:
        """加载训练数据"""
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            
            self.episodes = self.data.get('episodes', [])
            self.session_info = {
                'session_id': self.data.get('session_id', ''),
                'agent_name': self.data.get('agent_name', ''),
                'timestamp': self.data.get('timestamp', ''),
                'total_episodes': self.data.get('total_episodes', 0)
            }
            
            if not self.episodes:
                self.notify("没有找到Episode数据", severity="error")
                self.exit()
                return
            
            self.current_episode = 0
            self.current_step = 0
            
        except FileNotFoundError:
            self.notify(f"文件不存在: {self.data_file}", severity="error")
            self.exit()
        except json.JSONDecodeError as e:
            self.notify(f"JSON解析错误: {e}", severity="error")
            self.exit()
        except Exception as e:
            self.notify(f"加载数据时出错: {e}", severity="error")
            self.exit()

    def compose(self) -> ComposeResult:
        """构建UI"""
        yield Header()
        
        with Horizontal(id="main-container"):
            # 左侧面板：Episode信息
            with Vertical(id="left-panel"):
                yield Label("[bold]Episode 信息[/bold]", id="episode-label")
                yield EpisodeInfo({}, id="episode-info")
            
            # 中间面板：迷宫显示
            with Vertical(id="center-panel"):
                yield Label("[bold]迷宫状态[/bold]", id="maze-label")
                yield MazeDisplay([], [], id="maze-display")
            
            # 右侧面板：Step信息
            with Vertical(id="right-panel"):
                yield Label("[bold]Step 信息[/bold]", id="step-label")
                yield StepInfo({}, id="step-info")
        
        # 底部状态栏
        yield Container(
            Static(id="controls-hint"),
            id="status-bar"
        )
        
        yield Footer()

    def on_mount(self) -> None:
        """应用启动时执行"""
        self.load_data()
        self.update_display()

    def update_display(self) -> None:
        """更新显示"""
        if not self.episodes:
            return
        
        # 确保索引有效
        self.current_episode = max(0, min(self.current_episode, len(self.episodes) - 1))
        episode_data = self.episodes[self.current_episode]
        steps = episode_data.get('steps', [])
        
        # 确保step索引有效
        self.current_step = max(0, min(self.current_step, len(steps) - 1))
        step_data = steps[self.current_step] if steps else {}
        
        # 更新各个组件
        episode_info = self.query_one(EpisodeInfo)
        episode_info.episode_data = episode_data
        episode_info.update_info()
        
        maze_display = self.query_one(MazeDisplay)
        maze_display.maze_state = step_data.get('maze_state', [])
        maze_display.agent_pos = step_data.get('agent_pos', [0, 0])
        maze_display.update_display()
        
        step_info = self.query_one(StepInfo)
        step_info.step_data = step_data
        step_info.update_info()
        
        # 更新状态栏
        controls_hint = self.query_one("#controls-hint", Static)
        controls_hint.update(
            f"[bold]Session:[/bold] {self.session_info.get('session_id', '')} | "
            f"[bold]Agent:[/bold] {self.session_info.get('agent_name', '')} | "
            f"[bold]Episode:[/bold] {self.current_episode + 1}/{len(self.episodes)} | "
            f"[bold]Step:[/bold] {self.current_step + 1}/{len(steps)} | "
            f"[dim]↑↓: 切换Episode | ←→: 切换Step | q: 退出[/dim]"
        )

    def watch_current_episode(self, old_value: int, new_value: int) -> None:
        """监听episode变化"""
        self.current_step = 0
        self.update_display()

    def watch_current_step(self, old_value: int, new_value: int) -> None:
        """监听step变化"""
        self.update_display()

    def action_prev_episode(self) -> None:
        """上一个episode"""
        if self.current_episode > 0:
            self.current_episode -= 1

    def action_next_episode(self) -> None:
        """下一个episode"""
        if self.current_episode < len(self.episodes) - 1:
            self.current_episode += 1

    def action_prev_step(self) -> None:
        """上一个step"""
        if self.current_step > 0:
            self.current_step -= 1

    def action_next_step(self) -> None:
        """下一个step"""
        if self.episodes:
            steps = self.episodes[self.current_episode].get('steps', [])
            if self.current_step < len(steps) - 1:
                self.current_step += 1


def main() -> None:
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: uv run python rlf/tui.py <训练数据文件路径>")
        print("示例: uv run python rlf/tui.py ./training_data/xxx.json")
        sys.exit(1)
    
    data_file = sys.argv[1]
    
    if not Path(data_file).exists():
        print(f"错误: 文件不存在 - {data_file}")
        sys.exit(1)
    
    app = TrainingReplayApp(data_file)
    app.run()


if __name__ == "__main__":
    main()