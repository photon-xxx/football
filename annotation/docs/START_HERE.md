# 启动 McByte 轨迹修复界面

推荐使用带“轨迹缺帧感知”的正式入口：

```powershell
& E:\miniconda3\envs\mcs\python.exe experiment/mcbyte-track-repair-ui/code/run_track_repair_ui.py --view view1
```

可选 `view1`、`view2`、`view3`；默认读取对应的 `mask_conditioned.txt`。如需查看 Locked-IoU 轨迹：

```powershell
& E:\miniconda3\envs\mcs\python.exe experiment/mcbyte-track-repair-ui/code/run_track_repair_ui.py --view view1 --tracker locked_iou
```

完整按键和状态机说明见 [RUN.md](RUN.md)。
