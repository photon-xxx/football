import numpy as np

def is_valid(points, pitch_to_pixel, pixel_to_pitch, w, h):
    """
    Checks the validity of a set of points based on coordinate transformations and geometric constraints.
    Parameters
    ----------
    points : array-like, shape (N, 2)
        List or array of points in pitch (field) coordinates.
    pitch_to_pixel : callable
        Function that converts pitch coordinates to pixel coordinates.
    pixel_to_pitch : callable
        Function that converts pixel coordinates back to pitch coordinates.
    w : int or float
        Width of the image (in pixels).
    h : int or float
        Height of the image (in pixels).
    Returns
    -------
    mask : np.ndarray, shape (N,)
        Boolean array indicating which points are valid according to the following criteria:
        - The pixel coordinates are within the image boundaries.
        - The round-trip transformation error (pitch -> pixel -> pitch) is less than 5 units.
        - The local transformation preserves orientation (Jacobian determinant is positive).
    """
    import os
    import yaml
    root_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(root_path, "config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    # 将点从球场坐标转换为像素坐标
    points = np.array(points, dtype=np.float32).reshape(-1, 2)  # 确保输入是正确的形状
    N = points.shape[0]
    points_x = points.copy()
    points_x[:, 0] += 0.1

    points_y = points.copy()
    points_y[:, 1] += 0.1

    points_tot = np.vstack([points, points_x, points_y])
    pixel_points = pitch_to_pixel(points_tot)
    
    # 将像素坐标转换回球场坐标
    pitch_points = pixel_to_pitch(pixel_points[:N, :])
    
    mask = np.zeros(points.shape[0], dtype=bool)
    for i, (pixel_pt, pixel_pt_x, pixel_pt_y,pitch_pt, pt) in enumerate(zip(pixel_points[:N, :],
                                                     pixel_points[N:2*N, :],
                                                     pixel_points[2 * N:, :],
                                                     pitch_points,
                                                     points)):
        if 0 <= pixel_pt[0] < w and 0 <= pixel_pt[1] < h \
            and (np.linalg.norm(pitch_pt - pt) < config_dict['reprojection_eps']) \
            and ((not config_dict['enable_orientation']) or 
                (pixel_pt_x[0] - pixel_pt[0]) * (pixel_pt_y[1] - pixel_pt[1]) \
              > (pixel_pt_x[1] - pixel_pt[1]) * (pixel_pt_y[0] - pixel_pt[0])):
            mask[i] = True

    return mask

def is_valid_img(points, pitch_to_pixel, pixel_to_pitch):
    import os
    import yaml
    root_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(root_path, "config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    points = np.array(points, dtype=np.float32).reshape(-1, 2)  # 确保输入是正确的形状
    N = points.shape[0]
    points_x = points.copy()
    points_x[:, 0] += 0.1

    points_y = points.copy()
    points_y[:, 1] += 0.1

    points_tot = np.vstack([points, points_x, points_y])
    pitch_points = pixel_to_pitch(points_tot)
    
    pixel_points = pitch_to_pixel(pitch_points[:N, :])
    
    mask = np.zeros(points.shape[0], dtype=bool)
    for i, (pitch_pt, pitch_pt_x, pitch_pt_y,pixel_pt, pt) in enumerate(zip(pitch_points[:N, :],
                                                     pitch_points[N:2*N, :],
                                                     pitch_points[2 * N:, :],
                                                     pixel_points,
                                                     points)):
        if (np.linalg.norm(pixel_pt - pt) < config_dict['reprojection_eps']) \
            and ((not config_dict['enable_orientation']) or 
                (pitch_pt_x[0] - pitch_pt[0]) * (pitch_pt_y[1] - pitch_pt[1]) \
              > (pitch_pt_x[1] - pitch_pt[1]) * (pitch_pt_y[0] - pitch_pt[0])):
            mask[i] = True

    return mask