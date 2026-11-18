import torch
import numpy as np
if not hasattr(np, "object"):
    np.object = object

from torchvision.ops import box_iou

def get_IoU(result, iou_threshold=0.15):
    """
    计算YOLOv8检测结果中的bbox遮挡情况。

    参数:
        result: YOLOv8推理返回的单个result对象 (results[0])
        iou_threshold: 判断遮挡的IoU阈值。
                       0.0 表示只要有像素重叠就算遮挡。
                       可以设置为 0.1 或更高以忽略微小的边缘接触。

    返回:
        list of tuples: [(idx1, idx2, iou_value), ...]
                        返回所有发生遮挡的两两组合及其IoU值。
    """
    # 1. 获取检测框坐标 (xyxy格式: xmin, ymin, xmax, ymax)
    # result.boxes.xyxy 是一个 Tensor，通常已经在 GPU 上
    boxes = result.boxes.xyxy

    # 如果检测框少于2个，不可能存在两两遮挡
    if boxes.shape[0] < 2:
        return []

    # 2. 高效计算 N x N 的 IoU 矩阵
    # torchvision.ops.box_iou 是高度优化的 C++ 实现
    # 结果 matrix[i][j] 代表第 i 个框和第 j 个框的 IoU
    iou_matrix = box_iou(boxes, boxes)

    # 3. 过滤有效遮挡对
    # 我们只需要矩阵的上三角部分 (Upper Triangle)，去除对角线(自己跟自己比)和下三角(重复比较)
    # diagonal=1 表示不包含主对角线
    triu_idx = torch.triu_indices(
        iou_matrix.shape[0],
        iou_matrix.shape[1],
        offset=1,
        device=iou_matrix.device  # <--- 关键修改：让索引和数据在同一个设备上
    )
    # 获取上三角的所有IoU值
    upper_tri_ious = iou_matrix[triu_idx[0], triu_idx[1]]

    # 4. 根据阈值筛选
    # 找到 IoU > threshold 的索引掩码
    mask = upper_tri_ious > iou_threshold

    # 如果没有遮挡，直接返回空
    if not mask.any():
        return []

    # 5. 提取结果
    # 使用掩码提取符合条件的框的索引
    idx1 = triu_idx[0][mask]
    idx2 = triu_idx[1][mask]
    values = upper_tri_ious[mask]

    # 转换为Python列表返回 (将Tensor转为cpu上的数值)
    # 格式: [(框1索引, 框2索引, IoU值), ...]
    occlusions = list(zip(idx1.tolist(), idx2.tolist(), values.tolist()))

    return occlusions


# =================使用示例=================
if __name__ == "__main__":
    from ultralytics import YOLO
    from frame_tool import load_frame
    import supervision as sv

    # 加载模型
    model = YOLO('../models/yolov8n.pt')

    # 推理图片
    frame= load_frame('../oimg/frame_back.jpg')
    results = model.predict(frame)
    result = results[0]  # 获取第一张图的结果

    # 计算遮挡
    # 阈值设为0表示只要有重叠就算，设为0.1过滤掉轻微擦边
    overlaps = get_IoU(result, iou_threshold=0.15)

    print(f"检测到 {len(result.boxes)} 个目标")
    if overlaps:
        print(f"发现 {len(overlaps)} 组遮挡:")
        for i, j, score in overlaps:
            # 获取类别名称
            cls1 = result.names[int(result.boxes.cls[i])]
            cls2 = result.names[int(result.boxes.cls[j])]
            print(f"  - 框{i}({cls1}) 与 框{j}({cls2}) 重叠, IoU: {score:.4f}")
    else:
        print("未发现遮挡")

    sv.plot_image(frame)