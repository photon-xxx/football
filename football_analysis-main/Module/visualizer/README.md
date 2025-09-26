# Visualizer 模块

## 模块功能
Visualizer模块负责生成足球比赛分析的可视化图表，包括控球率统计、球员运动距离和速度分析等。该模块使用matplotlib库创建专业的统计图表，为教练和分析师提供直观的数据展示。

## 核心算法

### 1. 控球率计算
- **算法**: 基于时间序列的累计统计
- **功能**: 计算两队控球时间比例
- **特点**: 实时更新，动态显示

### 2. 运动数据分析
- **算法**: 基于轨迹数据的时间序列分析
- **功能**: 分析球员运动距离和速度变化
- **特点**: 支持多球员对比分析

### 3. 数据可视化
- **算法**: matplotlib图表绘制
- **功能**: 生成专业的统计图表
- **特点**: 支持多种图表类型

## 主要函数

### `plot_team_ball_control(team_ball_control, save_dir="figures")`
**功能**: 绘制两队控球率随时间变化的折线图
- **输入**: 
  - `team_ball_control` (np.ndarray): 控球权数组，逐帧标记哪一方控球（1或2）
  - `save_dir` (str): 保存目录，默认为"figures"
- **输出**: 无（直接保存图片文件）
- **核心算法**:
  1. 计算累计控球率
  2. 绘制时间序列折线图
  3. 保存为PNG文件

**详细步骤**:
```python
# 1. 计算累计控球率
num_frames = len(team_ball_control)
time = np.arange(num_frames)

team1_possession = np.cumsum(team_ball_control == 1) / (np.arange(num_frames) + 1) * 100
team2_possession = np.cumsum(team_ball_control == 2) / (np.arange(num_frames) + 1) * 100

# 2. 绘制图表
plt.figure(figsize=(12, 6))
plt.plot(time, team1_possession, color="red", label="Team 1")
plt.plot(time, team2_possession, color="blue", label="Team 2")
plt.xlabel("Frame")
plt.ylabel("Possession (%)")
plt.title("Ball Possession Over Time")
plt.legend()

# 3. 保存文件
plt.savefig(os.path.join(save_dir, "possession_over_time.png"))
plt.close()
```

### `plot_players_speed_distance(tracks, save_dir="figures")`
**功能**: 绘制所有球员的运动距离和速度分析图
- **输入**: 
  - `tracks` (dict): 跟踪结果字典
  - `save_dir` (str): 保存目录
- **输出**: 无（直接保存图片文件）
- **核心算法**:
  1. 提取所有球员的运动数据
  2. 绘制距离和速度时间序列
  3. 保存为PNG文件

**详细步骤**:
```python
# 1. 距离分析
plt.figure(figsize=(12, 6))
for player_id in set.union(*[set(f.keys()) for f in tracks["players"]]):
    distances = [
        tracks["players"][f].get(player_id, {}).get("distance", np.nan)
        for f in range(num_frames)
    ]
    plt.plot(time, distances, label=f"Player {player_id}")
plt.xlabel("Frame")
plt.ylabel("Distance (m)")
plt.title("Players Running Distance Over Time")
plt.legend()
plt.savefig(os.path.join(save_dir, "players_distance.png"))

# 2. 速度分析
plt.figure(figsize=(12, 6))
for player_id in set.union(*[set(f.keys()) for f in tracks["players"]]):
    speeds = [
        tracks["players"][f].get(player_id, {}).get("speed", np.nan)
        for f in range(num_frames)
    ]
    plt.plot(time, speeds, label=f"Player {player_id}")
plt.xlabel("Frame")
plt.ylabel("Speed (km/h)")
plt.title("Players Instant Speed Over Time")
plt.legend()
plt.savefig(os.path.join(save_dir, "players_speed.png"))
```

### `plot_players_speed_distance_by_team(tracks, save_dir="figures")`
**功能**: 按队伍分别绘制球员运动分析图
- **输入**: 
  - `tracks` (dict): 跟踪结果字典
  - `save_dir` (str): 保存目录
- **输出**: 无（直接保存图片文件）
- **核心算法**:
  1. 识别球员队伍归属
  2. 按队伍分组绘制图表
  3. 保存为PNG文件

**详细步骤**:
```python
# 1. 收集球员队伍信息
player_team_map = {}
for frame in tracks["players"]:
    for pid, pdata in frame.items():
        if "team" in pdata:
            player_team_map[pid] = pdata["team"]

# 2. 按队伍绘制图表
for team_id in [1, 2]:
    # 距离分析
    plt.figure(figsize=(12, 6))
    for pid, team in player_team_map.items():
        if team != team_id:
            continue
        distances = [
            tracks["players"][f].get(pid, {}).get("distance", np.nan)
            for f in range(num_frames)
        ]
        plt.plot(time, distances, label=f"Player {pid}")
    plt.xlabel("Frame")
    plt.ylabel("Distance (m)")
    plt.title(f"Team {team_id} Players Running Distance Over Time")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"team{team_id}_players_distance.png"))
```

## 核心算法详解

### 1. 控球率计算
```python
# 累计控球率计算
team1_possession = np.cumsum(team_ball_control == 1) / (np.arange(num_frames) + 1) * 100
team2_possession = np.cumsum(team_ball_control == 2) / (np.arange(num_frames) + 1) * 100
```
- **原理**: 基于时间序列的累计统计
- **公式**: 控球率 = 累计控球帧数 / 总帧数 × 100%
- **特点**: 实时更新，动态显示

### 2. 数据提取
```python
# 提取球员运动数据
distances = [
    tracks["players"][f].get(player_id, {}).get("distance", np.nan)
    for f in range(num_frames)
]
```
- **原理**: 从轨迹数据中提取运动信息
- **特点**: 处理缺失数据（NaN值）
- **优势**: 支持不完整轨迹

### 3. 图表绘制
```python
# 时间序列绘制
plt.plot(time, data, label=label)
plt.xlabel("Frame")
plt.ylabel("Value")
plt.title("Title")
plt.legend()
```
- **原理**: 基于matplotlib的图表绘制
- **特点**: 支持多线对比
- **优势**: 专业的数据可视化

## 参数调优

### 关键参数
- `figsize=(12, 6)`: 图表尺寸
- `color="red"`: 线条颜色
- `label="Team 1"`: 图例标签
- `save_dir="figures"`: 保存目录

### 性能优化
1. **批量处理**: 一次性处理所有数据
2. **内存管理**: 及时关闭图表释放内存
3. **文件管理**: 自动创建保存目录
4. **错误处理**: 处理缺失数据

## 使用示例

```python
# 绘制控球率图表
plot_team_ball_control(team_ball_control, save_dir="IO/figures")

# 绘制球员运动分析
plot_players_speed_distance(tracks, save_dir="IO/figures")

# 按队伍绘制分析
plot_players_speed_distance_by_team(tracks, save_dir="IO/figures")
```

## 数据结构

### 输入数据结构
```python
team_ball_control = np.array([1, 2, 1, 1, 2, ...])  # 控球权数组

tracks = {
    "players": [
        {
            player_id: {
                "distance": 120.5,    # 累计距离（米）
                "speed": 15.2,        # 瞬时速度（km/h）
                "team": 1/2,          # 队伍ID
                ...
            }
        }
    ]
}
```

### 输出数据结构
- `possession_over_time.png`: 控球率时间序列图
- `players_distance.png`: 球员距离分析图
- `players_speed.png`: 球员速度分析图
- `team1_players_distance.png`: 队伍1距离分析图
- `team1_players_speed.png`: 队伍1速度分析图
- `team2_players_distance.png`: 队伍2距离分析图
- `team2_players_speed.png`: 队伍2速度分析图

## 应用场景

1. **比赛分析**: 控球率和运动表现分析
2. **训练监控**: 运动员训练强度监控
3. **战术分析**: 球队战术和阵型分析
4. **数据报告**: 生成专业的分析报告

## 局限性

1. **数据依赖**: 依赖轨迹数据的完整性
2. **格式限制**: 输出为PNG格式
3. **交互性**: 静态图表，缺乏交互功能
4. **定制性**: 图表样式相对固定

## 改进建议

1. **交互图表**: 使用plotly等库创建交互图表
2. **实时更新**: 支持实时数据更新
3. **样式定制**: 提供更多图表样式选项
4. **数据导出**: 支持数据导出功能

## 依赖项

- matplotlib: 图表绘制
- numpy: 数组运算
- os: 文件操作
- 自定义数据: 通过其他模块导入
