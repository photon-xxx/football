"""
卡尔曼滤波器模块

实现了用于多目标跟踪的卡尔曼滤波器，用于预测和更新目标的状态。

主要功能：
1. 目标状态预测和更新
2. 马哈拉诺比斯距离计算
3. 批量处理优化

状态空间：8维 [x, y, w, h, vx, vy, vw, vh]
- (x, y): 边界框中心位置
- (w, h): 边界框宽度和高度
- (vx, vy): 位置速度
- (vw, vh): 尺寸速度

运动模型：恒定速度模型
观测模型：线性观测模型
"""

import numpy as np
# 修复numpy版本兼容性问题 (numpy >= 1.24移除了np.object)
if not hasattr(np, "object"):
    np.object = object

import scipy.linalg


"""
卡方分布0.95分位数值表（Chi-squared 0.95 quantile）

用于马哈拉诺比斯距离的门控阈值，包含1-9自由度的值。
取自MATLAB/Octave的chi2inv函数，用于马哈拉诺比斯门控。

使用说明：
    - 在数据关联时，计算检测与轨迹的马哈拉诺比斯距离
    - 如果距离平方小于对应的卡方分布0.95分位数，则认为匹配有效
    - 例如：如果计算的是4维距离（位置+尺寸），使用 chi2inv95[4] = 9.4877
    - 如果只计算2维位置距离，使用 chi2inv95[2] = 5.9915
    
数学含义：
    卡方分布的0.95分位数表示：在给定的自由度下，95%的样本的马哈拉诺比斯距离
    会小于这个值。超过这个值的匹配被认为是不太可能的（5%显著性水平）。

自由度说明：
    - 自由度 = 观测空间的维度
    - 4维观测（x, y, w, h）：4自由度
    - 2维观测（x, y）：2自由度
"""
chi2inv95 = {
    1: 3.8415,   # 1自由度（单维度）
    2: 5.9915,   # 2自由度（如仅位置：x, y）
    3: 7.8147,   # 3自由度
    4: 9.4877,   # 4自由度（完整观测：x, y, w, h）- 最常用
    5: 11.070,   # 5自由度
    6: 12.592,   # 6自由度
    7: 14.067,   # 7自由度
    8: 15.507,   # 8自由度
    9: 16.919    # 9自由度
}


class KalmanFilter(object):
    """
    用于图像空间中边界框跟踪的卡尔曼滤波器
    
    实现了8维状态空间的卡尔曼滤波器：
    - 状态向量: [x, y, w, h, vx, vy, vw, vh]
    - (x, y): 边界框中心位置
    - (w, h): 边界框宽度和高度  
    - (vx, vy): 位置速度
    - (vw, vh): 尺寸速度
    
    运动模型：恒定速度模型
    观测模型：线性观测模型，直接观测位置和尺寸
    
    Attributes:
        _motion_mat: 状态转移矩阵 (8x8)
        _update_mat: 观测矩阵 (4x8)
        _std_weight_position: 位置不确定性权重
        _std_weight_velocity: 速度不确定性权重
    """

    def __init__(self):
        """
        初始化卡尔曼滤波器
        
        设置状态转移矩阵、观测矩阵和不确定性权重参数。
        这些参数定义了卡尔曼滤波器的运动模型和观测模型。
        
        运动模型：恒定速度模型
            - 假设目标以恒定速度运动
            - 位置变化 = 速度 × 时间步长
            - 速度保持不变（可能受噪声影响）
        
        观测模型：线性观测
            - 只能观测到位置和尺寸（边界框）
            - 不能直接观测速度，需要通过位置变化推断
        """
        ndim, dt = 4, 1.  # 观测维度=4 (x, y, w, h), 时间步长=1（单位帧间隔）

        # 创建状态转移矩阵 (8x8)
        # 实现恒定速度模型: x_{k+1} = x_k + v_k * dt
        # 矩阵结构：
        # [I_4   dt*I_4]  其中 I_4 是4x4单位矩阵
        # [0_4   I_4   ]   0_4 是4x4零矩阵
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            # 在位置和对应速度之间建立联系
            # 例如：x_{k+1} = x_k + vx_k * dt
            self._motion_mat[i, ndim + i] = dt  # 位置 += 速度 * dt

        # 创建观测矩阵 (4x8)
        # 只观测位置和尺寸（前4维），不直接观测速度（后4维）
        # 矩阵结构：[I_4  0_4]，提取前4维（位置和尺寸）
        self._update_mat = np.eye(ndim, 2 * ndim)

        # 运动不确定性权重
        # 这些权重决定了模型对目标运动不确定性的估计
        # 权重越大，表示不确定性越小（噪声越小）
        self._std_weight_position = 1. / 20   # 位置不确定性权重（相对较小，位置较稳定）
        self._std_weight_velocity = 1. / 160   # 速度不确定性权重（更小，速度变化较慢）

    def initiate(self, measurement):
        """
        从未关联的检测创建新轨迹
        
        初始化卡尔曼滤波器的状态均值和协方差矩阵。
        当检测到新目标时（如新出现的球员），调用此方法初始化跟踪轨迹。
        
        Parameters
        ----------
        measurement : ndarray
            边界框坐标 (x, y, w, h)，其中：
            - (x, y): 边界框中心位置（像素坐标）
            - w: 边界框宽度（像素）
            - h: 边界框高度（像素）

        Returns
        -------
        (ndarray, ndarray)
            mean : 8维均值向量 [x, y, w, h, vx, vy, vw, vh]
                - 前4维（位置和尺寸）直接使用测量值
                - 后4维（速度）初始化为0（因为我们不知道目标的初始速度）
            covariance : 8x8维协方差矩阵
                - 表示初始状态的不确定性
                - 位置不确定性较小（因为我们有观测值）
                - 速度不确定性较大（因为我们不知道速度）
            返回新轨迹的均值向量和协方差矩阵
        """
        # 位置状态（前4维）直接使用测量值
        # 因为我们有实际的检测结果，所以初始位置是已知的
        mean_pos = measurement
        
        # 速度状态（后4维）初始化为0
        # 由于没有速度的直接观测，我们假设初始速度为0
        # 后续通过卡尔曼滤波会自动估计真实速度
        mean_vel = np.zeros_like(mean_pos)
        
        # 组合成8维状态向量：位置在前，速度在后
        mean = np.r_[mean_pos, mean_vel]

        # 计算初始协方差的标准差
        # 不确定性正比于边界框的尺寸（w, h）
        # 较大的边界框通常对应较大的不确定性
        std = [
            # 位置不确定性：使用2倍权重，因为初始位置是已知的，不确定性较小
            2 * self._std_weight_position * measurement[2],  # x位置不确定性（正比于宽度w）
            2 * self._std_weight_position * measurement[3],  # y位置不确定性（正比于高度h）
            2 * self._std_weight_position * measurement[2],  # w尺寸不确定性（正比于宽度w）
            2 * self._std_weight_position * measurement[3],  # h尺寸不确定性（正比于高度h）
            
            # 速度不确定性：使用10倍权重，因为初始速度未知，不确定性较大
            10 * self._std_weight_velocity * measurement[2], # x速度不确定性（正比于宽度w）
            10 * self._std_weight_velocity * measurement[3], # y速度不确定性（正比于高度h）
            10 * self._std_weight_velocity * measurement[2], # w速度不确定性（正比于宽度w）
            10 * self._std_weight_velocity * measurement[3]  # h速度不确定性（正比于高度h）
        ]
        
        # 创建对角协方差矩阵
        # 假设各个维度之间相互独立，所以协方差矩阵是对角的
        # 对角线元素是各维度的方差（标准差的平方）
        covariance = np.diag(np.square(std))
        
        return mean, covariance

    def predict(self, mean, covariance):
        """
        运行卡尔曼滤波预测步骤
        
        根据恒定速度运动模型预测下一时刻的目标状态分布。
        预测步骤在没有新观测时进行，用于估计目标的新位置。
        
        Parameters
        ----------
        mean : ndarray
            上一时刻目标状态的8维均值向量 [x, y, w, h, vx, vy, vw, vh]
        covariance : ndarray
            上一时刻目标状态的8x8维协方差矩阵
            表示上一时刻状态估计的不确定性

        Returns
        -------
        (ndarray, ndarray)
            mean : 8维均值向量，预测后的状态均值
            covariance : 8x8维协方差矩阵，预测后的状态协方差
            返回预测状态的均值向量和协方差矩阵
        
        算法原理：
            根据恒定速度模型，预测公式为：
            - 位置预测: pos_{k+1} = pos_k + vel_k * dt
            - 速度预测: vel_{k+1} = vel_k（速度保持不变，但受噪声影响）
            - 不确定性会增加，因为预测过程引入了不确定性（运动噪声）
        """
        # 计算位置不确定性标准差
        # 不确定性正比于边界框尺寸（w, h）
        # mean[2]是宽度w，mean[3]是高度h
        std_pos = [
            self._std_weight_position * mean[2],  # x位置不确定性（正比于宽度）
            self._std_weight_position * mean[3],  # y位置不确定性（正比于高度）
            self._std_weight_position * mean[2],  # w尺寸不确定性（正比于宽度）
            self._std_weight_position * mean[3]   # h尺寸不确定性（正比于高度）
        ]
        
        # 计算速度不确定性标准差
        # 速度的不确定性也正比于边界框尺寸
        std_vel = [
            self._std_weight_velocity * mean[2],  # x速度不确定性（正比于宽度）
            self._std_weight_velocity * mean[3],  # y速度不确定性（正比于高度）
            self._std_weight_velocity * mean[2],  # w速度不确定性（正比于宽度）
            self._std_weight_velocity * mean[3]   # h速度不确定性（正比于高度）
        ]
        
        # 创建运动噪声协方差矩阵 Q
        # 运动噪声表示模型的不确定性（目标可能不严格按恒定速度运动）
        # 使用对角矩阵，假设各维度的噪声相互独立
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        # 状态预测: x_{k|k-1} = F * x_{k-1|k-1}
        # 根据状态转移矩阵更新状态
        # 例如：x_new = x_old + vx_old * dt
        mean = np.dot(mean, self._motion_mat.T)
        
        # 协方差预测: P_{k|k-1} = F * P_{k-1|k-1} * F^T + Q
        # 预测过程会增加不确定性（因为运动噪声）
        # 第一项：状态转移对不确定性的影响
        # 第二项：运动噪声Q（motion_cov）
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """
        将状态分布投影到观测空间
        
        将8维状态空间投影到4维观测空间（位置和尺寸），用于：
        1. 计算观测似然（在数据关联时）
        2. 卡尔曼滤波的更新步骤
        
        观测只能看到位置和尺寸，不能直接看到速度，
        所以需要将8维状态投影到4维观测空间。
        
        Parameters
        ----------
        mean : ndarray
            状态的8维均值向量 [x, y, w, h, vx, vy, vw, vh]
        covariance : ndarray
            状态的8x8维协方差矩阵

        Returns
        -------
        (ndarray, ndarray)
            mean : 4维均值向量 [x, y, w, h]，投影后的观测空间均值
            covariance : 4x4维协方差矩阵，投影后的观测空间协方差
            返回给定状态估计在观测空间中的投影均值和协方差矩阵
        
        数学原理：
            投影公式：
            - 均值投影: z = H * x（H是观测矩阵，提取前4维）
            - 协方差投影: S = H * P * H^T + R
              其中 R 是观测噪声协方差矩阵
        """
        # 计算观测噪声的标准差
        # 观测噪声表示检测器的不确定性（检测可能不准确）
        # 噪声大小正比于边界框尺寸
        std = [
            self._std_weight_position * mean[2],  # x位置观测噪声（正比于宽度）
            self._std_weight_position * mean[3],  # y位置观测噪声（正比于高度）
            self._std_weight_position * mean[2],  # w尺寸观测噪声（正比于宽度）
            self._std_weight_position * mean[3]   # h尺寸观测噪声（正比于高度）
        ]
        # 创建观测噪声协方差矩阵 R
        innovation_cov = np.diag(np.square(std))

        # 状态投影: z = H * x
        # 使用观测矩阵提取位置和尺寸（前4维），丢弃速度（后4维）
        mean = np.dot(self._update_mat, mean)
        
        # 协方差投影: S = H * P * H^T + R
        # 第一项：将8x8协方差矩阵投影到4x4观测空间
        # 第二项：加上观测噪声R
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance):
        """
        批量运行卡尔曼滤波预测步骤（向量化版本）
        
        这是 `predict` 方法的批量优化版本，可以同时处理多个目标的预测，
        大大提高了计算效率。该方法使用向量化操作来并行处理多个轨迹。
        
        Parameters
        ----------
        mean : ndarray
            Nx8维的均值矩阵，包含N个目标在上一步的状态均值向量
            每一行对应一个目标的状态向量 [x, y, w, h, vx, vy, vw, vh]
        covariance : ndarray
            Nx8x8维的协方差矩阵数组，包含N个目标在上一步的状态协方差矩阵
            第一个维度对应不同的目标，后两个维度是8x8的协方差矩阵

        Returns
        -------
        (ndarray, ndarray)
            mean : Nx8维均值矩阵，预测后的状态均值
            covariance : Nx8x8维协方差矩阵数组，预测后的状态协方差
            返回N个目标预测状态的均值向量和协方差矩阵
        
        性能优化：
            - 使用向量化操作替代循环，显著提高计算速度
            - 适合批量处理大量轨迹，如足球比赛中多个球员的同时跟踪
        """
        # 批量计算位置不确定性标准差（向量化）
        # mean[:, 2] 是所有目标的宽度 w
        # mean[:, 3] 是所有目标的高度 h
        # 位置和尺寸的不确定性正比于边界框的尺寸
        std_pos = [
            self._std_weight_position * mean[:, 2],  # x位置不确定性（所有目标）
            self._std_weight_position * mean[:, 3],  # y位置不确定性（所有目标）
            self._std_weight_position * mean[:, 2],  # w尺寸不确定性（所有目标）
            self._std_weight_position * mean[:, 3]   # h尺寸不确定性（所有目标）
        ]
        
        # 批量计算速度不确定性标准差（向量化）
        std_vel = [
            self._std_weight_velocity * mean[:, 2],  # x速度不确定性（所有目标）
            self._std_weight_velocity * mean[:, 3],  # y速度不确定性（所有目标）
            self._std_weight_velocity * mean[:, 2],  # w速度不确定性（所有目标）
            self._std_weight_velocity * mean[:, 3]   # h速度不确定性（所有目标）
        ]
        
        # 将所有标准差合并并平方，得到方差
        # np.r_[std_pos, std_vel] 将位置和速度标准差拼接成8xN矩阵
        # .T 转置为 Nx8，每一行对应一个目标的8个标准差
        sqr = np.square(np.r_[std_pos, std_vel]).T

        # 为每个目标创建对角运动噪声协方差矩阵
        # 注意：这里仍使用循环，因为每个目标的协方差矩阵是独立的
        motion_cov = []
        for i in range(len(mean)):
            # 为第i个目标创建对角协方差矩阵
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov)  # 转换为Nx8x8数组

        # 批量状态预测：x_{k|k-1} = F * x_{k-1|k-1} （所有目标）
        # mean是Nx8矩阵，每一行是一个目标的状态
        mean = np.dot(mean, self._motion_mat.T)
        
        # 批量协方差预测：P_{k|k-1} = F * P_{k-1|k-1} * F^T + Q
        # covariance是Nx8x8数组，需要逐个处理每个目标的协方差矩阵
        # left = F * P_{k-1|k-1}，结果形状为Nx8x8
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        # 计算完整的协方差预测，并加上运动噪声
        # 维度变换是为了匹配矩阵乘法的要求
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean, covariance, measurement):
        """
        运行卡尔曼滤波更新步骤（校正步骤）
        
        使用新的观测值（检测结果）来校正预测状态。这是卡尔曼滤波的核心步骤，
        通过结合预测状态和实际观测值，得到更准确的状态估计。
        
        Parameters
        ----------
        mean : ndarray
            预测状态的8维均值向量 [x, y, w, h, vx, vy, vw, vh]
            这是经过预测步骤得到的先验状态估计
        covariance : ndarray
            预测状态的8x8维协方差矩阵
            表示预测状态的不确定性
        measurement : ndarray
            4维观测向量 [x, y, w, h]，表示实际检测到的边界框
            - (x, y): 边界框中心位置
            - w: 边界框宽度
            - h: 边界框高度

        Returns
        -------
        (ndarray, ndarray)
            new_mean : 8维均值向量，更新后的状态估计
            new_covariance : 8x8维协方差矩阵，更新后的不确定性
            返回经过观测校正后的状态分布
        
        算法原理：
            1. 将预测状态投影到观测空间
            2. 计算卡尔曼增益 K
            3. 计算创新（观测残差）：innovation = measurement - projected_mean
            4. 更新状态：new_mean = mean + K * innovation
            5. 更新协方差：new_covariance = covariance - K * S * K^T
            其中 S 是投影协方差矩阵
        
        数学公式：
            K = P * H^T * (H * P * H^T + R)^(-1)  （卡尔曼增益）
            x_new = x_pred + K * (z - H * x_pred)  （状态更新）
            P_new = P_pred - K * H * P_pred        （协方差更新）
        """
        # 步骤1：将预测状态投影到观测空间（4维）
        # 得到在观测空间中的均值和协方差
        projected_mean, projected_cov = self.project(mean, covariance)

        # 步骤2：计算卡尔曼增益 K = P * H^T * (H * P * H^T + R)^(-1)
        # 为了提高数值稳定性，使用Cholesky分解来求解线性方程组
        # cho_factor: 对投影协方差矩阵进行Cholesky分解，得到下三角矩阵
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        
        # cho_solve: 使用Cholesky分解求解线性方程组 (H*P*H^T+R) * K^T = H*P^T
        # 这里计算的是 K^T，所以最后需要转置
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        # kalman_gain形状：8x4，用于将4维观测误差映射到8维状态空间
        
        # 步骤3：计算创新（innovation），即观测残差
        # innovation = measurement - projected_mean
        # 表示观测值与预测值之间的差异
        innovation = measurement - projected_mean

        # 步骤4：更新状态均值
        # new_mean = mean + K * innovation
        # 将观测误差通过卡尔曼增益加权后加到预测状态上
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        
        # 步骤5：更新协方差矩阵
        # new_covariance = covariance - K * S * K^T
        # 观测降低了状态的不确定性
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False, metric='maha'):
        """
        计算状态分布与观测值之间的门控距离（Gate Distance）
        
        门控距离用于数据关联阶段，过滤掉不匹配的检测结果。
        只有距离小于阈值的检测才会被考虑与轨迹关联。
        合适的距离阈值可以从 `chi2inv95` 字典中获取。
        
        Parameters
        ----------
        mean : ndarray
            状态的8维均值向量，表示预测的目标状态
        covariance : ndarray
            状态的8x8维协方差矩阵，表示状态的不确定性
        measurements : ndarray
            Nx4维观测矩阵，包含N个检测结果
            每个检测的格式为 [x, y, w, h]，其中：
            - (x, y): 边界框中心位置
            - w: 边界框宽度（注意：注释中写的是a即aspect ratio，但实际代码中使用的是w）
            - h: 边界框高度
        only_position : bool, optional
            如果为True，只计算边界框中心位置的距离（2维），忽略尺寸信息
            如果为False，计算完整4维距离（位置+尺寸）
            默认False
        metric : str, optional
            距离度量方式：
            - 'maha': 马哈拉诺比斯距离（默认，推荐）
            - 'gaussian': 高斯距离（欧氏距离）
            默认'maha'
        
        Returns
        -------
        ndarray
            长度为N的一维数组，第i个元素是第i个观测值与状态分布之间的
            平方马哈拉诺比斯距离（或高斯距离，取决于metric参数）
        
        使用场景：
            在数据关联时，使用此函数计算所有检测与轨迹的距离，
            然后使用卡方分布的0.95分位数作为阈值进行过滤：
            - 如果 only_position=False: 使用 chi2inv95[4] 作为阈值（4自由度）
            - 如果 only_position=True: 使用 chi2inv95[2] 作为阈值（2自由度）
        
        数学原理：
            马哈拉诺比斯距离考虑了数据分布的形状和方向：
            d² = (x - μ)ᵀ * Σ⁻¹ * (x - μ)
            其中 x 是观测值，μ 是均值，Σ 是协方差矩阵
            这个距离与欧氏距离的区别在于，它考虑了不同维度之间的相关性
        """
        # 步骤1：将8维状态投影到4维观测空间
        # 因为我们只能观测到位置和尺寸，不能直接观测速度
        mean, covariance = self.project(mean, covariance)
        
        # 步骤2：如果只需要位置距离，提取位置相关的均值和协方差（前2维）
        if only_position:
            mean = mean[:2]              # 只保留x, y位置
            covariance = covariance[:2, :2]  # 只保留x, y的2x2协方差矩阵
            measurements = measurements[:, :2]  # 只保留检测的位置信息

        # 步骤3：计算观测值与预测值的差异（残差）
        # d形状：Nx4（如果only_position=False）或Nx2（如果only_position=True）
        # 每一行是一个检测与预测状态的差值向量
        d = measurements - mean
        
        # 步骤4：根据选择的度量方式计算距离
        if metric == 'gaussian':
            # 高斯距离：简单的欧氏距离的平方
            # 不考虑协方差矩阵，直接计算 d^T * d
            # 适用于各维度独立且方差相等的情况
            return np.sum(d * d, axis=1)
        
        elif metric == 'maha':
            # 马哈拉诺比斯距离：考虑协方差矩阵的距离度量
            # 公式：d² = (x - μ)ᵀ * Σ⁻¹ * (x - μ)
            
            # 为了提高数值稳定性和计算效率，使用Cholesky分解
            # 将协方差矩阵分解为：Σ = L * Lᵀ，其中L是下三角矩阵
            cholesky_factor = np.linalg.cholesky(covariance)
            
            # 求解线性方程组 L * z = d，得到 z = L⁻¹ * d
            # 这样 dᵀ * Σ⁻¹ * d = zᵀ * z（因为 Σ⁻¹ = (L * Lᵀ)⁻¹ = L⁻ᵀ * L⁻¹）
            # solve_triangular求解下三角线性方程组，比直接求逆更稳定高效
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False,
                overwrite_b=True)
            # z形状：4xN（或2xN），每一列是一个观测对应的z向量
            
            # 计算平方马哈拉诺比斯距离：zᵀ * z
            squared_maha = np.sum(z * z, axis=0)
            # 返回长度为N的数组，每个元素是一个观测的距离
            return squared_maha
        
        else:
            raise ValueError('invalid distance metric: {}'.format(metric))