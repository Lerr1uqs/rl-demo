
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

---

## Solution

# Solution for Issue #2: Add-π-policy-distribution-printing-feature

## 进展记录
- 增加算法类型、动作分布与会话导出模型（StrEnum + Pydantic）
- Agent层提供动作分布输出，训练与导出链路写入每步action分布
- TUI改为Pydantic解析并展示动作分布与算法类型

## 验证
- `uv run python -m pytest`
