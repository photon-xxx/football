# 启动球员对象管理版本

```powershell
& E:\miniconda3\envs\mcs\python.exe experiment/mcbyte-track-repair-ui/code/run_track_repair_ui_players.py --view view1
```

## 简单流程

1. 点击右侧 `NEW PLAYER` 创建逻辑球员。
2. 点击画面中的未归档 bbox，把整条 tracker ID 轨迹归档给该球员。
3. 归档轨迹不再出现在可操作 bbox 池中；在普通球员聚焦视图里也不能点击。
4. 点击球员栏中的某个球员，只显示该球员拥有的轨迹 bbox，其他球员和未归档轨迹全部隐藏。
5. 点击 `ADD TRACK`，临时显示未归档 bbox；点击后归档给当前球员。
6. 点击 `UNARCHIVED` 查看未归档池；在这里直接点击 bbox 会自动创建新球员并归档。
7. `REMOVE LAST` 将当前球员最近加入的轨迹退回未归档池。
8. `S` 同时保存修复 MOT、修复操作记录和 `*.players.json`。

快捷键：`P` 新建球员、`T` 添加轨迹、`V` 未归档池、`X` 移除最近轨迹。原来的 `N/B/W/U/S` 状态机仍保留。

当前“轨迹”按完整 tracker ID 归档，不是任意起止帧区间。一个 tracker ID 只能属于一个逻辑球员。
