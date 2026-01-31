项目需求：agent-doc/demand.md
python项目规范：agent-doc/constitutions.md

完成后一定要运行确保运行成功无误再交付。
@python-code-auditor 负责最后一步的审计。



<!-- ISSUE-MAKE:START -->
## Task: Add-π-policy-distribution-printing-feature

**Issue ID:** 2
**Type:** feat
**Created:** Thu Jan 15 2026 08:00:00 GMT+0800 (中国标准时间)

### Description

能够打印π策略分布 比如一个epoch中位于一个step的时候 状态是确定的 在这个状态的action分布也是确定的

导出文件中要标注使用的策略-agent 这样重放的时候 根据算法不同 显示的类型也不同
- dqn 选择action基于Q网络 每一步的时候打印四个action的q值
- 策略梯度 选择action基于策略概率分布 每一个step打印四个 概率分布的值
比如：
```
⬆️: 0.31
⬇️: 0.21
⬅️: 0.10
➡️: 0.38
```

这就要求导出的json中 要多一个type来标识算法 然后有专门的新的field来存储每一个step的action选择状态

### Instructions
- Work on this issue and implement the solution
- Document your progress in F:\coding-workspace\rl-framework\.issues\solution.md
- When complete, use `issue-make close 2` to archive

<!-- ISSUE-MAKE:END -->
