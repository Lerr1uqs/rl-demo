RL Framework
============

一个用于学习与验证常见强化学习算法的可视化实验框架。

Training Notes
--------------
- Off-policy 算法（如 DQN）支持 step 级训练：`step_train()` 会在每一步被调用。
- On-policy 算法（如 PG、PPO）保持 episode 级训练，仅实现 `train()`。
- 训练器会把 step 级 loss 求均值作为该 episode 的 loss，并记录到训练结果中。
