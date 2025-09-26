# Team Assigner 模块

## 模块功能
Team Assigner模块负责根据球员球衣颜色自动分配球员到不同队伍。该模块使用K-means聚类算法分析球员球衣颜色，自动识别两个队伍，并为每个球员分配队伍标签。

## 核心算法

### 1. K-means聚类
- **算法**: K-means聚类算法
- **功能**: 将球员球衣颜色分为两个聚类
- **特点**: 无监督学习，自动识别队伍

### 2. 颜色特征提取
- **算法**: 基于边界框的颜色采样
- **功能**: 从球员图像中提取代表性颜色
- **特点**: 聚焦球衣区域，减少背景干扰

### 3. 聚类中心计算
- **算法**: 基于聚类中心的颜色表示
- **功能**: 计算每个队伍的代表性颜色
- **特点**: 提供队伍颜色标识

## 主要函数

### `__init__()`
**功能**: 初始化队伍分配器
- **输入**: 无
- **输出**: 无
- **核心算法**: 初始化队伍颜色字典和球员队伍映射

### `get_clustering_model(image)`
**功能**: 创建K-means聚类模型
- **输入**: 
  - `image` (np.ndarray): 输入图像 (H, W, 3)
- **输出**: 
  - `kmeans` (KMeans): 训练好的聚类模型
- **核心算法**:
  ```python
  # 图像重塑为2D数组
  image_2d = image.reshape(-1, 3)
  
  # K-means聚类，2个聚类
  kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
  kmeans.fit(image_2d)
  ```

### `get_player_color(frame, bbox)`
**功能**: 提取球员球衣颜色
- **输入**: 
  - `frame` (np.ndarray): 视频帧
  - `bbox` (list): 球员边界框 [x1, y1, x2, y2]
- **输出**: 
  - `player_color` (np.ndarray): 球员颜色 [R, G, B]
- **核心算法**:
  1. 裁剪球员区域
  2. 取上半身图像（球衣区域）
  3. K-means聚类分析
  4. 识别球员聚类（非背景聚类）
  5. 返回聚类中心颜色

**详细步骤**:
```python
# 1. 裁剪球员区域
image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

# 2. 取上半身（球衣区域）
top_half_image = image[0:int(image.shape[0]/2), :]

# 3. K-means聚类
kmeans = self.get_clustering_model(top_half_image)
labels = kmeans.labels_

# 4. 识别球员聚类
corner_clusters = [clustered_image[0,0], clustered_image[0,-1], 
                   clustered_image[-1,0], clustered_image[-1,-1]]
non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
player_cluster = 1 - non_player_cluster

# 5. 返回球员颜色
player_color = kmeans.cluster_centers_[player_cluster]
```

### `assign_team_color(frame, player_detections)`
**功能**: 为所有球员分配队伍颜色
- **输入**: 
  - `frame` (np.ndarray): 视频帧
  - `player_detections` (dict): 球员检测结果
- **输出**: 无（直接设置self.team_colors）
- **核心算法**:
  1. 提取所有球员颜色
  2. 全局K-means聚类（2个队伍）
  3. 计算队伍颜色中心
  4. 存储队伍颜色映射

**详细步骤**:
```python
# 1. 提取所有球员颜色
player_colors = []
for _, player_detection in player_detections.items():
    bbox = player_detection["bbox"]
    player_color = self.get_player_color(frame, bbox)
    player_colors.append(player_color)

# 2. 全局聚类
kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
kmeans.fit(player_colors)

# 3. 存储队伍颜色
self.team_colors[1] = kmeans.cluster_centers_[0]
self.team_colors[2] = kmeans.cluster_centers_[1]
```

### `get_player_team(frame, player_bbox, player_id)`
**功能**: 获取单个球员的队伍
- **输入**: 
  - `frame` (np.ndarray): 视频帧
  - `player_bbox` (list): 球员边界框
  - `player_id` (int): 球员ID
- **输出**: 
  - `team_id` (int): 队伍ID (1或2)
- **核心算法**:
  1. 检查是否已分配队伍
  2. 提取球员颜色
  3. 使用训练好的聚类模型预测
  4. 缓存结果

**详细步骤**:
```python
# 1. 检查缓存
if player_id in self.player_team_dict:
    return self.player_team_dict[player_id]

# 2. 提取颜色并预测
player_color = self.get_player_color(frame, player_bbox)
team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
team_id += 1  # 转换为1-based索引

# 3. 缓存结果
self.player_team_dict[player_id] = team_id
```

## 核心算法详解

### 1. 球员聚类识别
```python
# 通过四个角落的聚类标签判断背景
corner_clusters = [clustered_image[0,0], clustered_image[0,-1], 
                   clustered_image[-1,0], clustered_image[-1,-1]]
non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
player_cluster = 1 - non_player_cluster
```
- **原理**: 图像角落通常是背景，通过统计角落聚类标签确定背景聚类
- **优势**: 自动识别球员区域，无需手动标注

### 2. 全局队伍分配
```python
# 使用所有球员颜色进行全局聚类
kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
kmeans.fit(player_colors)
```
- **原理**: 将所有球员颜色放在一起聚类，确保队伍一致性
- **优势**: 避免单个球员颜色偏差影响整体分配

### 3. 缓存机制
```python
if player_id in self.player_team_dict:
    return self.player_team_dict[player_id]
```
- **目的**: 避免重复计算，提高效率
- **原理**: 球员队伍分配在视频中相对稳定

## 参数调优

### 关键参数
- `n_clusters=2`: 聚类数量（两个队伍）
- `init="k-means++"`: 初始化方法
- `n_init=10`: 全局聚类初始化次数
- `n_init=1`: 单球员聚类初始化次数

### 性能优化
1. **区域限制**: 只分析上半身图像
2. **缓存机制**: 避免重复计算
3. **聚类优化**: 使用k-means++初始化
4. **多次初始化**: 全局聚类使用多次初始化提高稳定性

## 使用示例

```python
# 初始化队伍分配器
team_assigner = TeamAssigner()

# 分配队伍颜色（在第一帧调用）
team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

# 获取球员队伍
for frame_num, player_track in enumerate(tracks['players']):
    for player_id, track in player_track.items():
        team = team_assigner.get_player_team(
            video_frames[frame_num], 
            track['bbox'], 
            player_id
        )
        tracks['players'][frame_num][player_id]['team'] = team
```

## 数据结构

### 队伍颜色存储
```python
self.team_colors = {
    1: [R1, G1, B1],  # 队伍1的代表颜色
    2: [R2, G2, B2]   # 队伍2的代表颜色
}
```

### 球员队伍映射
```python
self.player_team_dict = {
    player_id: team_id,  # 球员ID -> 队伍ID
    ...
}
```

## 应用场景

1. **足球比赛分析**: 区分主客队
2. **篮球比赛**: 区分不同队伍
3. **团队运动**: 任何需要区分队伍的体育项目
4. **人群分析**: 区分不同群体

## 局限性

1. **颜色依赖**: 依赖球衣颜色差异
2. **光照敏感**: 光照变化可能影响颜色识别
3. **队伍数量**: 目前只支持两个队伍
4. **颜色相似**: 球衣颜色过于相似时可能误分

## 依赖项

- scikit-learn: K-means聚类算法
- numpy: 数组运算
- opencv-python: 图像处理（通过其他模块）
