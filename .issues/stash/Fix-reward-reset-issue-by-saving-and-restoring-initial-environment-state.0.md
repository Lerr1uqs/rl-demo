---
Create Date: 2026-01-15
Type: bug
---

env每次reset之后不一定能恢复之前的状态 奖励在后面几次epoch中被清除了。需要先保存初始化状态 之后reset的时候再从中恢复