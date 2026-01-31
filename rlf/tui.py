"""
训练数据回放TUI应用
使用Textual实现训练数据的可视化回放
"""
import sys
from pathlib import Path
from typing import List, Optional
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, Container
from textual.screen import ModalScreen
from textual.widgets import (
    Header,
    Footer,
    Static,
    Label,
    Input
)
from textual.reactive import reactive
from textual.binding import Binding

from rlf.schemas import (
    TrainingSessionData,
    EpisodeData as EpisodeDataModel,
    StepData as StepDataModel,
    ScoreType,
    ACTION_SYMBOLS
)

SCORE_TYPE_LABELS = {
    ScoreType.Q_VALUES: "Q值",
    ScoreType.POLICY_PROBS: "策略概率"
}


class EpisodeJumpScreen(ModalScreen):
    """Episode跳转浮窗"""

    BINDINGS = [
        ("escape", "close", "关闭"),
        ("q", "close", "关闭"),
    ]

    CSS = """
    EpisodeJumpScreen {
        background: rgba(0, 0, 0, 0.35);
        align: center middle;
    }

    #episode-dialog {
        width: 40;
        height: auto;
        padding: 1 2;
        border: round $primary;
        background: $panel;
    }

    #episode-input-title {
        text-align: center;
    }

    #episode-input-error {
        color: $error;
        text-align: center;
        height: 1;
    }
    """

    def __init__(self, max_episode: int) -> None:
        super().__init__()
        self.max_episode = max_episode

    def compose(self) -> ComposeResult:
        with Vertical(id="episode-dialog"):
            yield Label("[bold]跳转Episode[/bold]", id="episode-input-title")
            yield Input(placeholder="输入Episode编号", id="episode-input")
            yield Static("", id="episode-input-error")

    def on_mount(self) -> None:
        input_widget = self.query_one("#episode-input", Input)
        self.set_focus(input_widget)

    def action_close(self) -> None:
        self.app.pop_screen()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "episode-input":
            return
        input_text = event.value.strip()
        error_label = self.query_one("#episode-input-error", Static)
        if not input_text:
            error_label.update("请输入Episode编号")
            return
        if not input_text.isdigit():
            error_label.update("请输入数字")
            return

        target_episode = int(input_text)
        if target_episode < 1 or target_episode > self.max_episode:
            error_label.update(f"范围应为 1 - {self.max_episode}")
            return

        app = self.app
        if isinstance(app, TrainingReplayApp):
            app.jump_to_episode(target_episode)
        self.app.pop_screen()


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
                    row_text += symbols[cell] if cell in symbols else cell
            maze_text.append(row_text)

        self.update("\n".join(maze_text))


class ActionDistribution(Static):
    """动作分布显示组件"""

    def __init__(self, step_data: Optional[StepDataModel], **kwargs) -> None:
        super().__init__(**kwargs)
        self.step_data = step_data
        self.update_info()

    def update_info(self) -> None:
        """更新动作分布显示"""
        if self.step_data is None:
            self.update("无动作分布数据")
            return

        action_scores = self.step_data.action_scores
        score_label = SCORE_TYPE_LABELS[action_scores.score_type]
        lines: List[str] = [f"[bold]类型:[/bold] {score_label}"]
        selected_index = self.step_data.action if self.step_data.action >= 0 else None

        for index, action_name in enumerate(action_scores.action_order):
            symbol = ACTION_SYMBOLS[action_name]
            score_value = action_scores.scores[index]
            marker = "*" if selected_index == index else " "
            lines.append(f"{marker} {symbol}: {score_value:.2f}")

        self.update("\n".join(lines))


class EpisodeInfo(Static):
    """Episode信息显示组件"""

    def __init__(self, episode_data: Optional[EpisodeDataModel], **kwargs) -> None:
        super().__init__(**kwargs)
        self.episode_data = episode_data
        self.update_info()

    def update_info(self) -> None:
        """更新episode信息"""
        if self.episode_data is None:
            self.update("无Episode数据")
            return

        info_text = f"""
[bold cyan]Episode {self.episode_data.episode}[/bold cyan]

[bold]基本信息:[/bold]
  总奖励: {self.episode_data.total_reward:.2f}
  总步数: {self.episode_data.total_steps}
  成功: {'[green]✓[/green]' if self.episode_data.success else '[red]✗[/red]'}

[bold]Agent统计:[/bold]:
"""
        agent_stats = self.episode_data.agent_stats
        if agent_stats is not None:
            info_text += f"  Agent类型: {agent_stats.agent_type}\n"
            info_text += f"  Buffer大小: {agent_stats.buffer_size}\n"
            info_text += f"  Epsilon: {agent_stats.epsilon:.4f}\n"
            info_text += f"  平均损失: {agent_stats.avg_loss:.4f}\n"
            if agent_stats.avg_policy_loss > 0:
                info_text += f"  策略损失: {agent_stats.avg_policy_loss:.4f}\n"
                info_text += f"  价值损失: {agent_stats.avg_value_loss:.4f}\n"

        self.update(info_text)


class StepInfo(Static):
    """Step信息显示组件"""

    def __init__(self, step_data: Optional[StepDataModel], **kwargs) -> None:
        super().__init__(**kwargs)
        self.step_data = step_data
        self.update_info()

    def update_info(self) -> None:
        """更新step信息"""
        if self.step_data is None:
            self.update("无Step数据")
            return

        info_text = f"""
[bold cyan]Step {self.step_data.step}[/bold cyan]

[bold]动作信息:[/bold]
  动作: {self.step_data.action}
  动作名称: {self.step_data.action_name}

[bold]奖励信息:[/bold]
  即时奖励: {self.step_data.reward:.2f}
  累计奖励: {self.step_data.cumulative_reward:.2f}

[bold]状态信息:[/bold]
  状态: {self.step_data.state}
  Agent位置: ({self.step_data.agent_pos[0]}, {self.step_data.agent_pos[1]})

[bold]附加信息:[/bold]
"""
        step_info = self.step_data.info
        if step_info.hit:
            info_text += f"  命中: {step_info.hit}\n"
        if step_info.timeout:
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
    
    EpisodeInfo, StepInfo, ActionDistribution {
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
        Binding("e", "toggle_episode_input", "输入Episode"),
        Binding("q", "quit", "退出"),
    ]

    current_episode: reactive[int] = reactive(0)
    current_step: reactive[int] = reactive(0)

    def __init__(self, data_file: str) -> None:
        super().__init__()
        self.data_file = data_file
        self.session_data: Optional[TrainingSessionData] = None
        self.episodes: List[EpisodeDataModel] = []

    def load_data(self) -> None:
        """加载训练数据"""
        with open(self.data_file, 'r', encoding='utf-8') as f:
            raw_json = f.read()

        self.session_data = TrainingSessionData.model_validate_json(raw_json)
        self.episodes = self.session_data.episodes

        assert len(self.episodes) > 0
        assert self.session_data.total_episodes == len(self.episodes)

        self.current_episode = 0
        self.current_step = 0

    def compose(self) -> ComposeResult:
        """构建UI"""
        yield Header()
        
        with Horizontal(id="main-container"):
            # 左侧面板：Episode信息
            with Vertical(id="left-panel"):
                yield Label("[bold]Episode 信息[/bold]", id="episode-label")
                yield EpisodeInfo(None, id="episode-info")
            
            # 中间面板：迷宫显示
            with Vertical(id="center-panel"):
                yield Label("[bold]迷宫状态[/bold]", id="maze-label")
                yield MazeDisplay([], [], id="maze-display")
                yield Label("[bold]动作分布[/bold]", id="action-label")
                yield ActionDistribution(None, id="action-distribution")
            
            # 右侧面板：Step信息
            with Vertical(id="right-panel"):
                yield Label("[bold]Step 信息[/bold]", id="step-label")
                yield StepInfo(None, id="step-info")
        
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
        assert self.session_data is not None
        
        # 确保索引有效
        self.current_episode = max(0, min(self.current_episode, len(self.episodes) - 1))
        episode_data = self.episodes[self.current_episode]
        steps = episode_data.steps
        
        # 确保step索引有效
        self.current_step = max(0, min(self.current_step, len(steps) - 1))
        step_data = steps[self.current_step] if steps else None
        
        # 更新各个组件
        episode_info = self.query_one(EpisodeInfo)
        episode_info.episode_data = episode_data
        episode_info.update_info()
        
        maze_display = self.query_one(MazeDisplay)
        if step_data is None:
            maze_display.maze_state = []
            maze_display.agent_pos = [0, 0]
        else:
            maze_display.maze_state = step_data.maze_state
            maze_display.agent_pos = step_data.agent_pos
        maze_display.update_display()
        
        step_info = self.query_one(StepInfo)
        step_info.step_data = step_data
        step_info.update_info()

        action_distribution = self.query_one(ActionDistribution)
        action_distribution.step_data = step_data
        action_distribution.update_info()
        
        # 更新状态栏
        self._update_status_bar(total_steps=len(steps))

    def _update_status_bar(self, total_steps: int) -> None:
        """更新状态栏显示"""
        assert self.session_data is not None
        controls_hint = self.query_one("#controls-hint", Static)
        controls_hint.update(
            f"[bold]Session:[/bold] {self.session_data.session_id} | "
            f"[bold]Agent:[/bold] {self.session_data.agent_name} | "
            f"[bold]Algo:[/bold] {self.session_data.type} | "
            f"[bold]Episode:[/bold] {self.current_episode + 1}/{len(self.episodes)} | "
            f"[bold]Step:[/bold] {self.current_step + 1}/{total_steps} | "
            f"[dim]↑↓: 切换Episode | ←→: 切换Step | e: 输入Episode | q: 退出[/dim]"
        )

    def action_toggle_episode_input(self) -> None:
        """切换Episode输入框"""
        if not self.episodes:
            return
        if isinstance(self.screen, EpisodeJumpScreen):
            return
        self.push_screen(EpisodeJumpScreen(len(self.episodes)))

    def jump_to_episode(self, target_episode: int) -> None:
        """跳转到指定Episode"""
        if not self.episodes:
            return
        assert 1 <= target_episode <= len(self.episodes)
        self.current_episode = target_episode - 1

    def watch_current_episode(self, old_value: int, new_value: int) -> None:
        """监听episode变化"""
        self.current_step = 0
        self.update_display()

    def watch_current_step(self, old_value: int, new_value: int) -> None:
        """监听step变化"""
        self.update_display()

    def action_prev_episode(self) -> None:
        """上一个episode"""
        if not self.episodes:
            return
        if self.current_episode == 0:
            self.current_episode = len(self.episodes) - 1
            return
        self.current_episode -= 1

    def action_next_episode(self) -> None:
        """下一个episode"""
        if not self.episodes:
            return
        if self.current_episode >= len(self.episodes) - 1:
            self.current_episode = 0
            return
        self.current_episode += 1

    def action_prev_step(self) -> None:
        """上一个step"""
        if self.current_step > 0:
            self.current_step -= 1

    def action_next_step(self) -> None:
        """下一个step"""
        if self.episodes:
            steps = self.episodes[self.current_episode].steps
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
