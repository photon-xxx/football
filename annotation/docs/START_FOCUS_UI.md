# 启动“关注轨迹”显示版本

选中一条轨迹后，选中 bbox 保持黄色，其余 bbox 与标签统一变为 38% 不透明度的灰色：

```powershell
& E:\miniconda3\envs\mcs\python.exe experiment/mcbyte-track-repair-ui/code/run_track_repair_ui_focus.py --view view1
```

`view1` 可以替换为 `view2` 或 `view3`；其余按键和修复状态机不变。
