from textual.app import App, ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static
from textual.containers import Vertical
from textual.events import Key


class PopupWindow(ModalScreen):
    BINDINGS = [
        ("q", "close", "Close popup"),
        ("escape", "close", "Close popup"),
    ]

    CSS = """
    PopupWindow {
        background: rgba(0, 0, 0, 0.35);
        align: center middle;
    }

    #dialog {
        width: 40;
        height: 10;
        padding: 1 2;
        border: round white;
        background: blue;
    }

    #dialog Label { color: white; }
    #close-btn { margin-top: 1; width: 10; }
    """

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label("这是一个浮窗"),
            Label("按 'q' 或 Esc 关闭"),
            Button("关闭", id="close-btn"),
            id="dialog",
        )

    def action_close(self) -> None:
        self.app.pop_screen()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close-btn":
            self.action_close()


class MainApp(App):
    CSS = """
    Screen { background: red; }
    Static { color: white; padding: 1 2; }
    """

    BINDINGS = [
        ("q", "quit", "Quit app"),
    ]

    def compose(self) -> ComposeResult:
        yield Static("按 'e' 打开浮窗；浮窗里按 'q' 关闭；主界面按 'q' 退出程序")

    def on_key(self, event: Key) -> None:
        if event.key == "e":
            # ✅ 如果当前顶层已经是 PopupWindow，就别再 push 了
            if isinstance(self.screen, PopupWindow):
                return
            self.push_screen(PopupWindow())


if __name__ == "__main__":
    MainApp().run()
