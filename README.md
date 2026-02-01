RL Framework
============

一个用于学习与验证常见强化学习算法的可视化实验框架。

Training Notes
--------------
- Off-policy 算法（如 DQN）支持 step 级训练：`step_train()` 会在每一步被调用。
- On-policy 算法（如 PG、PPO）保持 episode 级训练，仅实现 `train()`。
- 训练器会把 step 级 loss 求均值作为该 episode 的 loss，并记录到训练结果中。


# NOTE
在DQN中 通过对环境感知 提供了action_mask 来规避掉一些走不了的Position 避免一直鬼打墙
action做mask的时候 计算损失函数也要对未来的Q'的全部action 做mask