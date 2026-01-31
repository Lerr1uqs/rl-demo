
添加loss和reward的曲线支持 plot 弹出

---

## Solution

# Solution for Issue #6: Add-support-for-plotting-loss-and-reward-curves-in-pop-up-windows

## 进度记录
- 新增训练阶段的 loss 轨迹记录，训练结果包含 episode_losses。
- 在训练器中加入弹窗绘图方法，分别展示 reward 与 loss 曲线。
- 主入口训练结束后自动弹出曲线窗口。
- 添加 matplotlib 依赖以支持绘图。

## 运行验证
- 命令: `$env:PYTHONUTF8=1; $env:MPLBACKEND='Agg'; .\.venv\Scripts\activate.ps1; @'...script...'@ | uv run python -`
- 结果: 训练与绘图流程完成（Agg后端提示非交互式窗口警告）。
