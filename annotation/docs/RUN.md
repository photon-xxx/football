# McByte 轨迹修复界面：运行说明

## 启动

在 `E:\paper` 下运行：

```powershell
& E:\miniconda3\envs\mcs\python.exe experiment/mcbyte-track-repair-ui/code/track_repair_ui.py --view view1
```

把 `view1` 换成 `view2` 或 `view3` 即可打开另外两个视角。默认使用 `mask_conditioned.txt`；加 `--tracker locked_iou` 可检查另一套轨迹。

## 最短流程

1. 在任意帧点击关注球员的 bbox，选中 ID 会显示为黄色。
2. 按 `N` 跳到该轨迹最后一帧。
3. 末帧仍正确：按 `N` 接受，进入下一帧选择新轨迹。
4. 末帧错误：按 `B` 跳到正确起点与错误终点的中间帧。
5. 中间帧正确按 `N`，错误按 `B`，直到定位首个错误帧。
6. 按 `W`，再点击真正属于关注球员的 bbox；程序从当前帧到结尾交换两个 tracker ID。
7. 按 `N` 检查修复后的末帧；`U` 撤销，`S` 保存。

## 按键

| 操作 | 功能 |
|---|---|
| 点击 bbox | 选择轨迹；`W` 后点击表示正确轨迹 |
| `N` | 当前判断正确 / 跳末帧 / 接受轨迹 |
| `B` | 当前判断错误，收缩二分区间 |
| `W` | 进入交换修复状态 |
| `A` / `D` | 前后 1 帧 |
| `J` / `L` | 前后 10 帧 |
| 拖动时间轴 | 任意帧跳转 |
| `R` | 取消当前选择/二分搜索 |
| `U` | 撤销最后一次交换 |
| `S` | 保存新 MOT 和 `repairs.json` |
| `Q` | 退出；有未保存修改时大写 `Q` 强制放弃 |

默认保存到 `experiment/mcbyte-track-repair-ui/outputs/`，不会覆盖原始 MOT。

## 第一版边界

- 当前修复是从错误帧开始一直交换到视频结尾，适合两个同时存在的轨迹发生 ID switch。
- 正确球员在错误帧没有 bbox 时，暂时无法点击修复。
- 暂不保存 mask、模型 memory 或跨视角身份。
- 一次 suffix swap 可能影响之后的其他错误，因此需要继续检查末帧，必要时再次修复或撤销。
