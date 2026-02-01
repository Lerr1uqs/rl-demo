
可以增加一个命令行参数 让typer选择最后导出的训练数据位置

---

## Solution

# Solution for Issue #3: Add-command-line-argument-for-specifying-training-data-export-location

进展记录：
- 已在 `main.py` 中引入 Typer CLI，支持通过 `--algorithm/-a` 选择算法。
- 新增 `--export-dir/-o` 参数，可指定训练数据导出目录。
- `MazeTrainer` 新增 `save_dir` 参数，传递到 `TrainingDataSaver` 完成导出路径控制。
