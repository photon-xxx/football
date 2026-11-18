import numpy as np
IDX = 0  # 示例索引值，可根据需要调整

def my_distort_point(x, y, k1, k2, k3, p1, p2):
    """
    正向畸变计算：已知理想坐标 (x, y)，计算畸变后的坐标 (x_corrected, y_corrected)
    对应图片中的公式直接实现。
    """
    # 1. 计算 r^2
    r2 = x**2 + y**2
    r4 = r2 * r2
    r6 = r2 * r4
    
    # 2. 计算径向畸变系数 (Radial)
    # (1 + k1*r^2 + k2*r^4 + k3*r^6)
    k_radial = 1 + k1 * r2 + k2 * r4 + k3 * r6
    
    # 3. 计算切向畸变项 (Tangential)
    # 注意图片中 x_corrected 对应 p2 项包含 (r^2 + 2x^2)
    # y_corrected 对应 p1 项包含 (r^2 + 2y^2)
    x_tangential = 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
    y_tangential = p1 * (r2 + 2 * y**2) + 2 * p2 * x * y
    
    # 4. 组合得到最终结果
    x_corrected = x * k_radial + x_tangential
    y_corrected = y * k_radial + y_tangential
    
    return x_corrected, y_corrected

def my_undistort_point(x_corr, y_corr, k1, k2, k3, p1, p2, iter_num=10):
    """
    反向去畸变：已知畸变坐标 (x_corr, y_corr)，反解理想坐标 (x, y)
    由于无法求解析解，这里使用【定点迭代法】近似求解。
    通常 5-10 次迭代即可达到极高的精度。
    """
    # 初始猜测：假设没有畸变，理想坐标等于畸变坐标
    x = x_corr
    y = y_corr
    
    for _ in range(iter_num):
        r2 = x**2 + y**2
        r4 = r2 * r2
        r6 = r2 * r4
        
        # 计算当前的径向因子
        k_radial = 1 + k1 * r2 + k2 * r4 + k3 * r6
        
        # 计算当前的切向畸变增量
        delta_x = 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
        delta_y = p1 * (r2 + 2 * y**2) + 2 * p2 * x * y
        
        # 更新 x, y
        # 根据公式：x_corr = x * k_radial + delta_x
        # 移项得：x = (x_corr - delta_x) / k_radial
        x = (x_corr - delta_x) / k_radial
        y = (y_corr - delta_y) / k_radial
        
    return x, y

# ================= 示例验证 =================
if __name__ == "__main__":
    # 假设的一组参数
    k1, k2, k3 = -0.2, 0.1, 0.05
    p1, p2 = 0.01, -0.01
    
    # 原始理想坐标
    x_real, y_real = 0.5, 0.5
    print(f"原始理想坐标: ({x_real}, {y_real})")
    
    # 1. 计算畸变 (Forward)
    x_dist, y_dist = my_distort_point(x_real, y_real, k1, k2, k3, p1, p2)
    print(f"畸变后坐标:   ({x_dist:.6f}, {y_dist:.6f})")
    
    # 2. 反解理想坐标 (Inverse)
    x_rec, y_rec = my_undistort_point(x_dist, y_dist, k1, k2, k3, p1, p2)
    print(f"反解出的坐标: ({x_rec:.6f}, {y_rec:.6f})")
    
    # 检查误差
    error = np.sqrt((x_rec - x_real)**2 + (y_rec - y_real)**2)
    print(f"反解误差:     {error:.10f}")