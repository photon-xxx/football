# CumTB 三视角共享球员 Session

启动：

```powershell
& E:\miniconda3\envs\mcs\python.exe experiment/mcbyte-track-repair-ui/code/run_track_repair_ui_multiview.py --view view1
```

界面顶部的 `VIEW1 / VIEW2 / VIEW3` 按钮用于切换视角，也可以直接按 `1 / 2 / 3`。切换只更换视频、MOT 轨迹及该视角的修复历史；右侧逻辑球员对象在整个 session 中共享。当前球员会跟随到新视角，便于给同一球员继续添加该视角的轨迹。

首次启动会自动读取已有的 `outputs/view1_mask_conditioned_repaired.players.json`，将 26 个 view1 球员迁移到：

```text
outputs/cumtb_mask_conditioned.session.json
```

首次进入后请按一次 `S` 完成迁移保存。以后 view1、view2、view3 都读写这一个 session 文件。每个视角拥有独立的 track ID 命名空间，因此 `view1/T10` 和 `view2/T10` 可以属于不同球员。

球员列表支持：

- 在列表区域滚动鼠标滚轮；
- 拖动最右侧滚动条滑块；
- 点击滚动条空白位置快速翻页；
- 新建或点击球员时自动滚动到该球员。

其余操作保持不变：`P` 新建球员，`T` 给当前球员添加轨迹，`V` 查看未归档池，`X` 撤回当前视角最近添加的轨迹，`N/B/W/U/S` 用于检查、二分、修复、撤销和保存。

注意：`S` 会保存共享 session，并保存当前视角以及存在未保存修复的其他视角 MOT；原始 McByte 文件不会被覆盖。
