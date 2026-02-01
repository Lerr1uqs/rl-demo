import torch
import torch.distributions as distributions

# 1. 创建类别分布 (比如3个动作的选择概率)
probs = torch.tensor([0.2, 0.5, 0.3])  # 动作0:20%, 动作1:50%, 动作2:30%
dist = distributions.Categorical(probs=probs)

print(f"概率分布: {probs}")
print(f"类别数: {dist.param_shape}")  # torch.Size([3])

# 2. 采样 (模拟选择动作)
samples = dist.sample((10,))  # 采样10次
print(f"\n采样10次: {samples}") # 采样10次: tensor([1, 2, 2, 0, 0, 0, 1, 2, 1, 1])

# 3. 计算对数概率 log_prob (强化学习中计算loss常用)
actions = torch.tensor([0, 1, 2, 1])
log_probs = dist.log_prob(actions)
from math import log
print(f"log(0.2) = {log(0.2)}") # log(0.2) = -1.6094379124341003
# 动作 [0, 1, 2, 1] 的 log_prob: tensor([-1.6094, -0.6931, -1.2040, -0.6931])
print(f"\n动作 {actions.tolist()} 的 log_prob: {log_probs}")

# 4. 实际概率值
probs_val = torch.exp(log_probs)
print(f"对应概率值: {probs_val}") # tensor([0.2000, 0.5000, 0.3000, 0.5000])
