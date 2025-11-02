"""
全局运动补偿 (Global Motion Compensation) 模块

用于估计和补偿相机运动对跟踪的影响，提高跟踪稳定性。

支持的方法：
1. sparseOptFlow: 稀疏光流法 (默认)
2. orb: ORB特征匹配
3. sift: SIFT特征匹配
4. ecc: 增强相关系数
5. file: 从文件读取预计算的变换矩阵

主要功能：
- 估计帧间相机运动
- 补偿相机运动对轨迹的影响
- 提高跟踪的鲁棒性
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
# 修复numpy版本兼容性问题 (numpy >= 1.24移除了np.object)
if not hasattr(np, "object"):
    np.object = object

import copy
import time


class GMC:
    """
    全局运动补偿类
    
    估计连续帧之间的相机运动，并用于补偿轨迹位置
    """
    
    def __init__(self, method='sparseOptFlow', downscale=2, verbose=None):
        """
        初始化全局运动补偿器
        
        Args:
            method (str): 运动估计方法，可选值：
                - 'sparseOptFlow': 稀疏光流法（默认方法，速度最快）
                - 'orb': ORB特征匹配法（速度较快，对旋转和缩放有一定鲁棒性）
                - 'sift': SIFT特征匹配法（精度高但速度慢，对旋转、缩放、光照变化鲁棒）
                - 'ecc': 增强相关系数法（基于整幅图像的相关性）
                - 'file': 从文件读取预计算的变换矩阵
                - 'none': 不进行运动补偿
            downscale (int): 图像下采样倍数，用于加速计算。数值越大计算越快但精度降低
            verbose (list, optional): 详细输出设置，当method='file'时使用
                - verbose[0]: 序列名称
                - verbose[1]: 是否为ablation实验
        
        初始化各种方法所需的检测器、提取器、匹配器等工具
        """
        super(GMC, self).__init__()

        self.method = method
        self.downscale = max(1, int(downscale))  # 确保下采样倍数至少为1

        # 初始化ORB方法所需工具
        if self.method == 'orb':
            # FAST特征检测器，阈值20用于控制特征点质量
            self.detector = cv2.FastFeatureDetector_create(20)
            # ORB特征提取器，用于生成二进制特征描述符
            self.extractor = cv2.ORB_create()
            # 暴力匹配器，使用汉明距离（适合二进制描述符）
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        # 初始化SIFT方法所需工具
        elif self.method == 'sift':
            # SIFT同时作为检测器和提取器
            # nOctaveLayers=3: 每个八度的层数
            # contrastThreshold=0.02: 对比度阈值，过滤低对比度特征
            # edgeThreshold=20: 边缘阈值，过滤边缘响应强的特征
            self.detector = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20)
            self.extractor = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20)
            # SIFT使用L2范数（欧氏距离）进行匹配
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)

        # 初始化ECC方法所需参数
        elif self.method == 'ecc':
            number_of_iterations = 5000  # 最大迭代次数
            termination_eps = 1e-6  # 收敛精度阈值
            # 欧几里得变换模式：允许旋转和平移（3个自由度）
            self.warp_mode = cv2.MOTION_EUCLIDEAN
            # 终止条件：达到精度阈值或达到最大迭代次数
            self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

        # 初始化稀疏光流法参数（默认方法）
        elif self.method == 'sparseOptFlow':
            # goodFeaturesToTrack的参数配置
            self.feature_params = dict(
                maxCorners=1000,           # 最大特征点数量
                qualityLevel=0.01,         # 特征点质量阈值（相对最大值）
                minDistance=1,             # 特征点之间的最小距离（像素）
                blockSize=3,               # 计算特征值的窗口大小
                useHarrisDetector=False,   # 是否使用Harris角点检测（False表示使用Shi-Tomasi）
                k=0.04                     # Harris检测器的自由参数（useHarrisDetector=True时使用）
            )
            # 可选：打开文件记录GMC结果（当前已注释）
            # self.gmc_file = open('GMC_results.txt', 'w')

        # 从文件读取预计算的变换矩阵
        elif self.method == 'file' or self.method == 'files':
            seqName = verbose[0]  # 序列名称
            ablation = verbose[1]  # 是否为ablation实验
            
            # 根据是否为ablation实验选择不同的文件路径
            if ablation:
                filePath = r'tracker/GMC_files/MOT17_ablation'
            else:
                filePath = r'tracker/GMC_files/MOTChallenge'

            # 处理序列名称中的后缀（如-FRCNN、-DPM、-SDP等）
            if '-FRCNN' in seqName:
                seqName = seqName[:-6]  # 移除"-FRCNN"后缀
            elif '-DPM' in seqName:
                seqName = seqName[:-4]  # 移除"-DPM"后缀
            elif '-SDP' in seqName:
                seqName = seqName[:-4]  # 移除"-SDP"后缀

            # 打开GMC文件
            self.gmcFile = open(filePath + "/GMC-" + seqName + ".txt", 'r')

            if self.gmcFile is None:
                raise ValueError("Error: Unable to open GMC file in directory:" + filePath)
        
        # 不进行运动补偿
        elif self.method == 'none' or self.method == 'None':
            self.method = 'none'
        else:
            raise ValueError("Error: Unknown CMC method:" + method)

        # 初始化用于存储前一帧信息的变量
        self.prevFrame = None           # 前一帧灰度图像
        self.prevKeyPoints = None       # 前一帧的关键点
        self.prevDescriptors = None     # 前一帧的特征描述符（用于ORB/SIFT方法）

        # 标记是否已完成第一帧的初始化
        self.initializedFirstFrame = False

    def apply(self, raw_frame, detections=None):
        """
        应用全局运动补偿
        
        根据初始化时选择的方法估计相机运动并返回变换矩阵。
        该方法会根据不同的运动估计方法调用相应的具体实现。
        
        Args:
            raw_frame (np.ndarray): 当前帧BGR图像，形状为 (height, width, 3)
            detections (list, optional): 检测结果列表，格式为 [x1, y1, x2, y2, ...]
                用于某些方法（如ORB/SIFT）中排除检测区域，避免在移动物体上提取特征点
            
        Returns:
            H (np.ndarray): 2x3仿射变换矩阵，用于补偿相机运动
                矩阵形式：
                [a  b  tx]  其中 a,b,c,d 表示旋转和缩放，tx,ty 表示平移
                [c  d  ty]
                如果是第一帧或方法为'none'，返回单位矩阵（表示无变换）
        
        方法选择：
            - 'sparseOptFlow': 使用稀疏光流法（默认，速度最快）
            - 'orb'/'sift': 使用特征点匹配法（精度较高）
            - 'ecc': 使用增强相关系数法（基于整幅图像）
            - 'file': 从文件读取预计算的变换矩阵
            - 'none': 不进行运动补偿
        """
        # 根据初始化时选择的方法调用相应的实现
        if self.method == 'orb' or self.method == 'sift':
            # ORB/SIFT特征匹配方法
            return self.applyFeaures(raw_frame, detections)
        elif self.method == 'ecc':
            # ECC增强相关系数方法
            return self.applyEcc(raw_frame, detections)
        elif self.method == 'sparseOptFlow':
            # 稀疏光流法（默认方法）
            return self.applySparseOptFlow(raw_frame, detections)
        elif self.method == 'file':
            # 从文件读取预计算的变换矩阵
            return self.applyFile(raw_frame, detections)
        elif self.method == 'none':
            # 不进行运动补偿，返回单位矩阵
            return np.eye(2, 3)
        else:
            # 默认情况：返回单位矩阵（不进行补偿）
            return np.eye(2, 3)

    def applyEcc(self, raw_frame, detections=None):
        """
        使用增强相关系数（Enhanced Correlation Coefficient, ECC）算法估计相机运动
        
        ECC算法基于最大化两幅图像之间的相关性来估计变换矩阵，对光照变化有一定鲁棒性。
        该方法适用于处理整幅图像的全局运动，不需要特征点检测。
        
        Args:
            raw_frame (np.ndarray): 当前帧BGR图像，形状为 (height, width, 3)
            detections (list, optional): 检测结果列表，当前方法未使用此参数
        
        Returns:
            H (np.ndarray): 2x3仿射变换矩阵，用于补偿相机运动
                矩阵形式：
                [a  b  tx]  其中 a,b,c,d 表示旋转和缩放，tx,ty 表示平移
                [c  d  ty]
        """
        # 初始化：获取图像尺寸并转换为灰度图
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        # 初始化单位变换矩阵（表示无变换）
        H = np.eye(2, 3, dtype=np.float32)

        # 图像下采样以加速计算（TODO: 可考虑使用图像金字塔）
        if self.downscale > 1.0:
            # 先进行高斯模糊以减少下采样带来的混叠效应
            frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
            # 下采样图像
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))
            # 更新图像尺寸
            width = width // self.downscale
            height = height // self.downscale

        # 处理第一帧：只保存当前帧，不进行运动估计
        if not self.initializedFirstFrame:
            # 保存当前帧作为下一帧的参考帧
            self.prevFrame = frame.copy()

            # 标记已完成初始化
            self.initializedFirstFrame = True

            # 第一帧返回单位矩阵（无变换）
            return H

        # 运行ECC算法估计变换矩阵
        # cv2.findTransformECC通过最大化增强相关系数来找到最优变换
        # 参数说明：
        #   - self.prevFrame: 模板图像（前一帧）
        #   - frame: 输入图像（当前帧）
        #   - H: 初始变换矩阵（会更新为最终结果）
        #   - self.warp_mode: 变换模式（MOTION_EUCLIDEAN：旋转+平移）
        #   - self.criteria: 迭代终止条件
        #   - None: 输入掩码（未使用）
        #   1: gaussFiltSize，高斯滤波大小
        try:
            (cc, H) = cv2.findTransformECC(self.prevFrame, frame, H, self.warp_mode, self.criteria, None, 1)
            # cc: 增强相关系数值（越接近1表示匹配越好）
            # H: 估计得到的变换矩阵
        except:
            # 如果算法失败，输出警告并返回单位矩阵
            print('Warning: find transform failed. Set warp as identity')

        return H

    def applyFeaures(self, raw_frame, detections=None):
        """
        使用特征点匹配方法（ORB或SIFT）估计相机运动
        
        该方法通过检测和匹配特征点来估计帧间的相机运动：
        1. 检测特征点（使用FAST/SIFT检测器）
        2. 提取特征描述符（使用ORB/SIFT提取器）
        3. 匹配特征点（使用KNN匹配器）
        4. 过滤误匹配（基于描述符距离和空间距离）
        5. 估计仿射变换矩阵（使用RANSAC算法）
        
        Args:
            raw_frame (np.ndarray): 当前帧BGR图像，形状为 (height, width, 3)
            detections (list, optional): 检测结果列表，格式为 [x1, y1, x2, y2, ...]
                用于在特征点检测时排除检测区域（避免在移动物体上提取特征点）
        
        Returns:
            H (np.ndarray): 2x3仿射变换矩阵，用于补偿相机运动
        """
        # 初始化：获取图像尺寸并转换为灰度图
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        # 初始化单位变换矩阵
        H = np.eye(2, 3)

        # 图像下采样以加速计算（TODO: 可考虑使用图像金字塔）
        if self.downscale > 1.0:
            # 注意：ORB/SIFT方法通常不需要预处理模糊
            # frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))
            width = width // self.downscale
            height = height // self.downscale

        # 创建特征点检测掩码
        # 目的是在图像中心区域（排除边缘）和检测区域（移动物体）外检测特征点
        mask = np.zeros_like(frame)
        # 设置图像中心98%区域为有效区域（排除边缘2%）
        # 原注释：mask[int(0.05 * height): int(0.95 * height), int(0.05 * width): int(0.95 * width)] = 255
        mask[int(0.02 * height): int(0.98 * height), int(0.02 * width): int(0.98 * width)] = 255
        
        # 排除检测区域：在移动物体（球员、球等）上不应该提取特征点
        # 因为这些点会随物体移动，不能反映相机运动
        if detections is not None:
            for det in detections:
                # det格式：[x1, y1, x2, y2, ...]，需要根据下采样调整坐标
                tlbr = (det[:4] / self.downscale).astype(np.int_)
                # 将检测框区域设为0（无效区域）
                mask[tlbr[1]:tlbr[3], tlbr[0]:tlbr[2]] = 0

        # 在掩码区域内检测特征点
        keypoints = self.detector.detect(frame, mask)

        # 计算特征描述符（用于后续匹配）
        keypoints, descriptors = self.extractor.compute(frame, keypoints)

        # 处理第一帧：只保存当前帧信息，不进行匹配
        if not self.initializedFirstFrame:
            # 保存当前帧数据作为下一帧的参考
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)

            # 标记已完成初始化
            self.initializedFirstFrame = True

            # 第一帧返回单位矩阵（无变换）
            return H

        # 使用KNN匹配器匹配特征描述符（K=2用于Lowe's ratio test）
        # knnMatch返回每个查询描述符的K个最近邻匹配
        knnMatches = self.matcher.knnMatch(self.prevDescriptors, descriptors, 2)

        # 基于描述符距离和空间距离过滤匹配
        matches = []           # 经过过滤的匹配结果
        spatialDistances = []  # 每个匹配点的空间距离

        # 设置最大空间距离阈值：图像尺寸的25%
        # 如果特征点移动超过这个距离，可能是误匹配
        maxSpatialDistance = 0.25 * np.array([width, height])

        # 处理无匹配的情况
        if len(knnMatches) == 0:
            # 更新参考帧数据
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)

            # 无匹配时返回单位矩阵
            return H

        # 第一轮过滤：Lowe's ratio test + 空间距离过滤
        for m, n in knnMatches:
            # Lowe's ratio test：最佳匹配的距离应该明显小于次佳匹配
            # 阈值0.9：最佳匹配距离 < 0.9 * 次佳匹配距离
            if m.distance < 0.9 * n.distance:
                # 获取匹配点的坐标
                prevKeyPointLocation = self.prevKeyPoints[m.queryIdx].pt
                currKeyPointLocation = keypoints[m.trainIdx].pt

                # 计算空间距离（前一帧到当前帧的位移）
                spatialDistance = (prevKeyPointLocation[0] - currKeyPointLocation[0],
                                   prevKeyPointLocation[1] - currKeyPointLocation[1])

                # 空间距离过滤：只保留位移在合理范围内的匹配
                # 这样可以过滤掉因遮挡、误匹配等原因产生的大位移
                if (np.abs(spatialDistance[0]) < maxSpatialDistance[0]) and \
                        (np.abs(spatialDistance[1]) < maxSpatialDistance[1]):
                    spatialDistances.append(spatialDistance)
                    matches.append(m)

        # 第二轮过滤：基于空间距离的统计异常值检测
        # 计算空间距离的均值和标准差
        meanSpatialDistances = np.mean(spatialDistances, 0)
        stdSpatialDistances = np.std(spatialDistances, 0)

        # 使用2.5倍标准差规则过滤异常值（内点检测）
        # 保留在均值±2.5倍标准差范围内的匹配点
        inliesrs = (spatialDistances - meanSpatialDistances) < 2.5 * stdSpatialDistances

        # 提取经过所有过滤的优质匹配点
        goodMatches = []
        prevPoints = []   # 前一帧的匹配点坐标
        currPoints = []   # 当前帧的匹配点坐标
        for i in range(len(matches)):
            # 检查x和y方向都是内点
            if inliesrs[i, 0] and inliesrs[i, 1]:
                goodMatches.append(matches[i])
                prevPoints.append(self.prevKeyPoints[matches[i].queryIdx].pt)
                currPoints.append(keypoints[matches[i].trainIdx].pt)

        # 转换为numpy数组以便后续处理
        prevPoints = np.array(prevPoints)
        currPoints = np.array(currPoints)

        # 调试功能：可视化特征点匹配（默认关闭）
        if 0:
            # 将两帧图像水平拼接
            matches_img = np.hstack((self.prevFrame, frame))
            matches_img = cv2.cvtColor(matches_img, cv2.COLOR_GRAY2BGR)
            W = np.size(self.prevFrame, 1)  # 第一帧的宽度
            
            # 绘制匹配点和连线
            for m in goodMatches:
                prev_pt = np.array(self.prevKeyPoints[m.queryIdx].pt, dtype=np.int_)
                curr_pt = np.array(keypoints[m.trainIdx].pt, dtype=np.int_)
                curr_pt[0] += W  # 调整第二帧中点的x坐标
                
                # 随机颜色
                color = np.random.randint(0, 255, (3,))
                color = (int(color[0]), int(color[1]), int(color[2]))

                # 绘制连线和特征点
                matches_img = cv2.line(matches_img, prev_pt, curr_pt, tuple(color), 1, cv2.LINE_AA)
                matches_img = cv2.circle(matches_img, prev_pt, 2, tuple(color), -1)
                matches_img = cv2.circle(matches_img, curr_pt, 2, tuple(color), -1)

            # 显示图像
            plt.figure()
            plt.imshow(matches_img)
            plt.show()

        # 使用RANSAC算法估计仿射变换矩阵
        # 需要至少4个匹配点才能估计仿射变换（6个参数）
        if (np.size(prevPoints, 0) > 4) and (np.size(prevPoints, 0) == np.size(prevPoints, 0)):
            # cv2.estimateAffinePartial2D估计相似变换（旋转、平移、缩放）
            # RANSAC算法可以自动过滤异常值（内点/外点检测）
            H, inliesrs = cv2.estimateAffinePartial2D(prevPoints, currPoints, cv2.RANSAC)
            # H: 估计得到的2x3仿射变换矩阵
            # inliesrs: 标记哪些点是内点（符合估计的变换）

            # 如果进行了下采样，需要将平移分量恢复到原始尺度
            if self.downscale > 1.0:
                H[0, 2] *= self.downscale  # tx 平移分量
                H[1, 2] *= self.downscale  # ty 平移分量
        else:
            # 匹配点不足时输出警告
            print('Warning: not enough matching points')

        # 更新参考帧数据，为下一帧处理做准备
        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)
        self.prevDescriptors = copy.copy(descriptors)

        return H

    def applySparseOptFlow(self, raw_frame, detections=None):
        """
        使用稀疏光流法（Sparse Optical Flow）估计相机运动（默认方法）
        
        稀疏光流法是速度最快的运动估计方法，流程如下：
        1. 在当前帧检测角点特征（goodFeaturesToTrack）
        2. 使用金字塔LK光流跟踪这些特征点到下一帧（calcOpticalFlowPyrLK）
        3. 根据跟踪结果估计仿射变换矩阵（estimateAffinePartial2D）
        
        该方法速度快、实时性好，适合视频跟踪应用。
        
        Args:
            raw_frame (np.ndarray): 当前帧BGR图像，形状为 (height, width, 3)
            detections (list, optional): 检测结果列表，当前方法未使用此参数
        
        Returns:
            H (np.ndarray): 2x3仿射变换矩阵，用于补偿相机运动
        """
        # 记录开始时间（用于性能分析，可选）
        t0 = time.time()

        # 初始化：获取图像尺寸并转换为灰度图
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        # 初始化单位变换矩阵
        H = np.eye(2, 3)

        # 图像下采样以加速计算
        if self.downscale > 1.0:
            # 注意：稀疏光流法通常不需要预处理模糊
            # frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))

        # 在当前帧检测角点特征（Shi-Tomasi角点检测）
        # goodFeaturesToTrack使用特征值检测强角点，适合光流跟踪
        # mask=None: 在整个图像范围内检测特征点
        # **self.feature_params: 展开参数字典
        keypoints = cv2.goodFeaturesToTrack(frame, mask=None, **self.feature_params)

        # 处理第一帧：只保存当前帧信息，不进行光流跟踪
        if not self.initializedFirstFrame:
            # 保存当前帧数据作为下一帧的参考
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)

            # 标记已完成初始化
            self.initializedFirstFrame = True

            # 第一帧返回单位矩阵（无变换）
            return H

        # 使用金字塔Lucas-Kanade光流法跟踪特征点
        # calcOpticalFlowPyrLK使用图像金字塔进行多尺度光流跟踪，对大位移更鲁棒
        # 参数说明：
        #   - self.prevFrame: 前一帧灰度图像
        #   - frame: 当前帧灰度图像
        #   - self.prevKeyPoints: 前一帧的特征点（需要跟踪的点）
        #   - None: 下一帧特征点（输出，未使用）
        matchedKeypoints, status, err = cv2.calcOpticalFlowPyrLK(self.prevFrame, frame, self.prevKeyPoints, None)
        # matchedKeypoints: 当前帧中跟踪到的特征点位置
        # status: 跟踪状态数组，1表示成功跟踪，0表示跟踪失败
        # err: 跟踪误差（可选）

        # 只保留成功跟踪的特征点
        prevPoints = []   # 前一帧中成功跟踪的点
        currPoints = []   # 当前帧中对应的点

        for i in range(len(status)):
            if status[i]:  # 只保留跟踪成功的点
                prevPoints.append(self.prevKeyPoints[i])
                currPoints.append(matchedKeypoints[i])

        # 转换为numpy数组以便后续处理
        prevPoints = np.array(prevPoints)
        currPoints = np.array(currPoints)

        # 使用RANSAC算法估计仿射变换矩阵
        # 需要至少4个匹配点才能估计仿射变换（6个参数）
        if (np.size(prevPoints, 0) > 4) and (np.size(prevPoints, 0) == np.size(prevPoints, 0)):
            # cv2.estimateAffinePartial2D估计相似变换（旋转、平移、缩放）
            # RANSAC算法可以自动过滤异常值（内点/外点检测）
            H, inliesrs = cv2.estimateAffinePartial2D(prevPoints, currPoints, cv2.RANSAC)
            # H: 估计得到的2x3仿射变换矩阵
            # inliesrs: 标记哪些点是内点（符合估计的变换）

            # 如果进行了下采样，需要将平移分量恢复到原始尺度
            if self.downscale > 1.0:
                H[0, 2] *= self.downscale  # tx 平移分量
                H[1, 2] *= self.downscale  # ty 平移分量
        else:
            # 跟踪点不足时输出警告
            print('Warning: not enough matching points')

        # 更新参考帧数据，为下一帧处理做准备
        # 注意：使用当前帧检测的新特征点，而不是跟踪得到的点
        # 这样可以保证特征点始终分布均匀，避免特征点逐渐集中在某些区域
        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)

        # 记录结束时间（用于性能分析，可选）
        t1 = time.time()

        # 可选：将GMC结果写入文件（当前已注释）
        # 格式：处理时间(ms) + 变换矩阵的6个参数
        # gmc_line = str(1000 * (t1 - t0)) + "\t" + str(H[0, 0]) + "\t" + str(H[0, 1]) + "\t" + str(
        #     H[0, 2]) + "\t" + str(H[1, 0]) + "\t" + str(H[1, 1]) + "\t" + str(H[1, 2]) + "\n"
        # self.gmc_file.write(gmc_line)

        return H

    def applyFile(self, raw_frame, detections=None):
        """
        从文件读取预计算的变换矩阵
        
        该方法用于使用预先计算好的GMC变换矩阵，通常用于：
        - 离线处理时的结果重现
        - 使用其他工具计算的GMC结果
        - 性能基准测试
        
        文件格式：每行包含7个制表符分隔的值
        - tokens[0]: 处理时间（毫秒，可选）
        - tokens[1-6]: 变换矩阵的6个参数 [a, b, tx, c, d, ty]
        
        Args:
            raw_frame (np.ndarray): 当前帧BGR图像，当前方法不使用此参数
            detections (list, optional): 检测结果列表，当前方法不使用此参数
        
        Returns:
            H (np.ndarray): 2x3仿射变换矩阵，从文件中读取
        
        注意：此方法假设文件中的变换矩阵是按帧顺序排列的
        """
        # 从文件中读取一行（对应当前帧的变换矩阵）
        line = self.gmcFile.readline()
        
        # 按制表符分割行内容
        tokens = line.split("\t")
        
        # 初始化单位变换矩阵
        H = np.eye(2, 3, dtype=np.float_)
        
        # 从文件读取变换矩阵的6个参数
        # 变换矩阵格式：
        # [a   b   tx]   对应 tokens[1], tokens[2], tokens[3]
        # [c   d   ty]   对应 tokens[4], tokens[5], tokens[6]
        H[0, 0] = float(tokens[1])  # a: x方向缩放和旋转
        H[0, 1] = float(tokens[2])  # b: x方向旋转和缩放
        H[0, 2] = float(tokens[3])  # tx: x方向平移
        H[1, 0] = float(tokens[4])  # c: y方向旋转和缩放
        H[1, 1] = float(tokens[5])  # d: y方向缩放和旋转
        H[1, 2] = float(tokens[6])  # ty: y方向平移

        return H