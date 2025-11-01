"""
基础跟踪类模块

定义了跟踪轨迹的基本状态和接口，为具体的跟踪算法提供基础框架。

主要包含：
1. TrackState: 轨迹状态枚举
2. BaseTrack: 基础轨迹类，定义通用属性和接口
"""

import numpy as np
from collections import OrderedDict


class TrackState(object):
    """
    轨迹状态枚举类
    
    定义了轨迹在整个生命周期中可能的状态：
    - New: 新创建的轨迹，尚未激活
    - Tracked: 正在跟踪的轨迹
    - Lost: 丢失的轨迹，可能重新找到
    - LongLost: 长时间丢失的轨迹
    - Removed: 已移除的轨迹，不再跟踪
    """
    New = 0        # 新轨迹
    Tracked = 1    # 正在跟踪
    Lost = 2       # 丢失
    LongLost = 3   # 长时间丢失
    Removed = 4    # 已移除


class BaseTrack(object):
    """
    基础轨迹类
    
    定义了所有轨迹的通用属性和接口，为具体的跟踪算法提供基础框架。
    具体的跟踪算法应该继承此类并实现相应的抽象方法。
    
    Attributes:
        _count: 轨迹ID计数器，用于生成唯一ID
        track_id: 当前轨迹的唯一标识符
        is_activated: 轨迹是否已激活
        state: 当前轨迹状态
        history: 轨迹历史记录
        features: 外观特征列表
        curr_feature: 当前帧特征
        score: 轨迹置信度分数
        start_frame: 轨迹开始帧
        frame_id: 当前帧ID
        time_since_update: 距离上次更新的时间
        location: 多相机系统中的位置信息
    """
    _count = 0  # 轨迹ID计数器

    # 轨迹基本信息
    track_id = 0
    is_activated = False
    state = TrackState.New

    # 轨迹数据
    history = OrderedDict()  # 轨迹历史记录
    features = []            # 外观特征列表
    curr_feature = None      # 当前帧特征
    score = 0                # 置信度分数
    
    # 时间信息
    start_frame = 0          # 轨迹开始帧
    frame_id = 0             # 当前帧ID
    time_since_update = 0    # 距离上次更新的时间

    # 多相机系统
    location = (np.inf, np.inf)  # 位置信息

    @property
    def end_frame(self):
        """
        获取轨迹结束帧
        
        Returns:
            int: 轨迹结束帧ID
        """
        return self.frame_id

    @staticmethod
    def next_id():
        """
        生成下一个轨迹ID
        
        使用静态计数器确保每个轨迹ID的唯一性
        
        Returns:
            int: 新的轨迹ID
        """
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        """
        激活轨迹的抽象方法
        
        子类必须实现此方法来定义轨迹激活的具体逻辑
        
        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError

    def predict(self):
        """
        预测轨迹下一帧状态的抽象方法
        
        子类必须实现此方法来定义状态预测的具体逻辑
        
        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError

    def update(self, *args, **kwargs):
        """
        更新轨迹状态的抽象方法
        
        子类必须实现此方法来定义轨迹更新的具体逻辑
        
        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError

    def mark_lost(self):
        """
        标记轨迹为丢失状态
        
        当轨迹在连续几帧中都没有匹配到检测时调用
        """
        self.state = TrackState.Lost

    def mark_long_lost(self):
        """
        标记轨迹为长时间丢失状态
        
        当轨迹丢失时间超过阈值时调用
        """
        self.state = TrackState.LongLost

    def mark_removed(self):
        """
        标记轨迹为已移除状态
        
        当轨迹确定不再需要跟踪时调用
        """
        self.state = TrackState.Removed

    @staticmethod
    def clear_count():
        """
        重置轨迹ID计数器
        
        在开始新的跟踪会话时调用，确保ID从1开始
        """
        BaseTrack._count = 0
