#!/usr/bin/env python3
"""
ArUco标记误差分析工具
遍历每个二维码，计算该二维码相对于其他二维码的位姿误差
包括旋转误差和距离误差，并生成可视化结果
"""

import json
import numpy as np
import cv2
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from typing import Optional, Tuple, Dict, Any, List
import glob
from pathlib import Path

# 设置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False


def load_camera_params(json_file):
    """从JSON文件加载相机内参"""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        intrinsic = data['intrinsic']
        
        # 构建相机内参矩阵
        camera_matrix = np.array([
            [intrinsic['fx'], 0, intrinsic['ppx']],
            [0, intrinsic['fy'], intrinsic['ppy']],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # 畸变系数
        dist_coeffs = np.array([
            intrinsic['k1'],
            intrinsic['k2'],
            intrinsic['p1'],
            intrinsic['p2'],
            intrinsic['k3']
        ], dtype=np.float32)
        
        print(f"相机内参加载成功:")
        print(f"  焦距: fx={intrinsic['fx']}, fy={intrinsic['fy']}")
        print(f"  主点: cx={intrinsic['ppx']}, cy={intrinsic['ppy']}")
        print(f"  畸变系数: {dist_coeffs}")
        
        return camera_matrix, dist_coeffs
        
    except Exception as e:
        print(f"加载相机内参失败: {e}")
        return None, None


def get_aruco_dict(dict_name):
    """获取ArUco字典"""
    dict_mapping = {
        '4x4_50': cv2.aruco.DICT_4X4_50,
        '4x4_100': cv2.aruco.DICT_4X4_100,
        '4x4_250': cv2.aruco.DICT_4X4_250,
        '4x4_1000': cv2.aruco.DICT_4X4_1000,
        '5x5_50': cv2.aruco.DICT_5X5_50,
        '5x5_100': cv2.aruco.DICT_5X5_100,
        '5x5_250': cv2.aruco.DICT_5X5_250,
        '5x5_1000': cv2.aruco.DICT_5X5_1000,
        '6x6_50': cv2.aruco.DICT_6X6_50,
        '6x6_100': cv2.aruco.DICT_6X6_100,
        '6x6_250': cv2.aruco.DICT_6X6_250,
        '6x6_1000': cv2.aruco.DICT_6X6_1000,
        '7x7_50': cv2.aruco.DICT_7X7_50,
        '7x7_100': cv2.aruco.DICT_7X7_100,
        '7x7_250': cv2.aruco.DICT_7X7_250,
        '7x7_1000': cv2.aruco.DICT_7X7_1000,
        'aruco_original': cv2.aruco.DICT_ARUCO_ORIGINAL
    }
    
    if dict_name in dict_mapping:
        try:
            return cv2.aruco.getPredefinedDictionary(dict_mapping[dict_name])
        except AttributeError:
            try:
                return cv2.aruco.Dictionary_get(dict_mapping[dict_name])
            except AttributeError:
                print("错误: 无法获取ArUco字典，请检查OpenCV版本")
                return None
    else:
        print(f"不支持的字典类型: {dict_name}")
        return None


def rotation_matrix_to_euler_angles(R):
    """将旋转矩阵转换为欧拉角 (ZYX顺序)"""
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    
    singular = sy < 1e-6
    
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    
    # 转换为度
    euler_angles = np.array([x, y, z]) * 180 / np.pi
    return euler_angles


def calculate_marker_id_from_position(row: int, col: int, squares_x: int) -> Optional[int]:
    """
    根据行列位置计算标记ID
    
    Args:
        row: 行号
        col: 列号
        squares_x: 标定板X方向方格数
    
    Returns:
        marker_id: 标记ID，如果位置无效则返回None
    """
    # 检查位置是否有效
    if row < 0 or col < 0 or col >= squares_x:
        return None
    
    # 检查该位置是否有标记（根据棋盘格规律）
    if (row % 2 == 0 and col % 2 == 0) or (row % 2 == 1 and col % 2 == 1):
        # 计算该位置的标记ID
        if row % 2 == 0:  # 偶数行
            col_in_row = col // 2
        else:  # 奇数行
            col_in_row = col // 2
        
        # 计算该行之前的标记总数
        markers_before_this_row = 0
        for prev_row in range(row):
            if prev_row % 2 == 0:  # 偶数行
                markers_before_this_row += (squares_x + 1) // 2
            else:  # 奇数行
                markers_before_this_row += squares_x // 2
        
        marker_id = markers_before_this_row + col_in_row
        return marker_id
    else:
        return None


def calculate_ground_truth_distance(marker1_id: int, marker2_id: int, squares_x: int, squares_y: int, square_length: float) -> float:
    """
    计算两个标记之间的真实距离
    
    Args:
        marker1_id: 第一个标记ID
        marker2_id: 第二个标记ID
        squares_x: 标定板X方向方格数
        squares_y: 标定板Y方向方格数
        square_length: 方格边长
    
    Returns:
        distance: 真实距离（米）
    """
    # 计算标记1的位置
    current_id = marker1_id
    row1 = 0
    col_in_row1 = 0
    
    while True:
        if row1 % 2 == 0:  # 偶数行
            markers_in_this_row = (squares_x + 1) // 2
        else:  # 奇数行
            markers_in_this_row = squares_x // 2
        
        if current_id < markers_in_this_row:
            col_in_row1 = current_id
            break
        else:
            current_id -= markers_in_this_row
            row1 += 1
    
    if row1 % 2 == 0:  # 偶数行：标记在偶数列
        col1 = col_in_row1 * 2
    else:  # 奇数行：标记在奇数列
        col1 = col_in_row1 * 2 + 1
    
    # 计算标记2的位置
    current_id = marker2_id
    row2 = 0
    col_in_row2 = 0
    
    while True:
        if row2 % 2 == 0:  # 偶数行
            markers_in_this_row = (squares_x + 1) // 2
        else:  # 奇数行
            markers_in_this_row = squares_x // 2
        
        if current_id < markers_in_this_row:
            col_in_row2 = current_id
            break
        else:
            current_id -= markers_in_this_row
            row2 += 1
    
    if row2 % 2 == 0:  # 偶数行：标记在偶数列
        col2 = col_in_row2 * 2
    else:  # 奇数行：标记在奇数列
        col2 = col_in_row2 * 2 + 1
    
    # 计算真实距离
    x1 = (col1 + 0.5) * square_length
    y1 = (row1 + 0.5) * square_length
    x2 = (col2 + 0.5) * square_length
    y2 = (row2 + 0.5) * square_length
    
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance


def get_row_col_from_marker_id(marker_id: int, squares_x: int) -> Tuple[int, int]:
    """
    根据标记ID反推其在棋盘中的行(row)与列(col)
    偶数行ID在偶数列，奇数行ID在奇数列
    """
    current_id = marker_id
    row = 0
    col_in_row = 0
    while True:
        if row % 2 == 0:
            markers_in_this_row = (squares_x + 1) // 2
        else:
            markers_in_this_row = squares_x // 2
        if current_id < markers_in_this_row:
            col_in_row = current_id
            break
        current_id -= markers_in_this_row
        row += 1
    if row % 2 == 0:
        col = col_in_row * 2
    else:
        col = col_in_row * 2 + 1
    return row, col


def get_marker_world_position(marker_id: int, squares_x: int, square_length: float) -> np.ndarray:
    """
    返回标记在世界坐标系中的中心点坐标 (x, y, z=0)
    """
    row, col = get_row_col_from_marker_id(marker_id, squares_x)
    x = (col + 0.5) * square_length
    y = (row + 0.5) * square_length
    return np.array([x, y, 0.0], dtype=np.float32)


def detect_aruco_markers(image_path, camera_matrix, dist_coeffs, dict_name='5x5_1000', marker_size=0.1):
    """
    检测图片中的所有ArUco标记位姿
    
    Args:
        image_path: 图片路径
        camera_matrix: 相机内参矩阵
        dist_coeffs: 畸变系数
        dict_name: ArUco字典名称
        marker_size: 标记边长（米）
    
    Returns:
        markers_info: 检测到的标记信息字典
    """
    # 获取ArUco字典
    aruco_dict = get_aruco_dict(dict_name)
    if aruco_dict is None:
        return {}
    
    # 创建ArUco检测器参数
    try:
        aruco_params = cv2.aruco.DetectorParameters()
    except AttributeError:
        try:
            aruco_params = cv2.aruco.DetectorParameters_create()
        except AttributeError:
            print("错误: 无法创建ArUco检测器参数，请检查OpenCV版本")
            return {}
    
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图片: {image_path}")
        return {}
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(f"图片尺寸: {image.shape}")
    
    # 检测ArUco标记
    try:
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        corners, ids, rejected = detector.detectMarkers(gray)
    except AttributeError:
        try:
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray, aruco_dict, parameters=aruco_params
            )
        except AttributeError:
            print("错误: 无法检测ArUco标记，请检查OpenCV版本")
            return {}
    
    if ids is None:
        print("未检测到任何ArUco标记")
        return {}
    
    print(f"检测到 {len(ids)} 个ArUco标记")
    print(f"检测到的ID: {ids.flatten()}")
    
    # 处理所有检测到的标记
    markers_info = {}
    
    for i, marker_id in enumerate(ids.flatten()):
        print(f"\n处理标记ID: {marker_id}")
        
        # 获取当前角点
        current_corners = corners[i]
        
        # 使用ArUco内置函数直接获取位姿
        try:
            try:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    [current_corners], marker_size, camera_matrix, dist_coeffs
                )
                rvec = rvecs[0][0]  # 旋转向量 (3x1)
                tvec = tvecs[0][0]  # 平移向量 (3x1)
                print("使用 estimatePoseSingleMarkers 获取位姿")
            except AttributeError:
                # 如果estimatePoseSingleMarkers不可用，使用solvePnP
                marker_points = np.array([
                    [-marker_size/2, -marker_size/2, 0],  # corner[0]: 左上角
                    [marker_size/2, -marker_size/2, 0],   # corner[1]: 右上角
                    [marker_size/2, marker_size/2, 0],    # corner[2]: 右下角
                    [-marker_size/2, marker_size/2, 0]    # corner[3]: 左下角
                ], dtype=np.float32)
                
                # 使用solvePnP估计位姿
                success, rvec, tvec = cv2.solvePnP(
                    marker_points, 
                    current_corners.astype(np.float32), 
                    camera_matrix, 
                    dist_coeffs
                )
                
                if not success:
                    print(f"位姿估计失败: solvePnP未收敛 (ID: {marker_id})")
                    continue
                
                print("使用 solvePnP 获取位姿")
                
        except Exception as e:
            print(f"位姿估计失败: {e} (ID: {marker_id})")
            continue
        
        # 将旋转向量转换为旋转矩阵
        R, _ = cv2.Rodrigues(rvec)
        
        # 计算欧拉角
        euler_angles = rotation_matrix_to_euler_angles(R)
        
        # 计算距离
        distance = np.linalg.norm(tvec)
        
        # 计算标记中心点
        center = np.mean(current_corners, axis=0)
        
        # 计算标记的面积
        area = cv2.contourArea(current_corners.astype(np.int32))
        
        markers_info[marker_id] = {
            'id': marker_id,
            'corners': current_corners,
            'center': center,
            'area': area,
            'rvec': rvec,
            'tvec': tvec,
            'R': R,
            'euler_angles': euler_angles,
            'distance': distance,
            'marker_size': marker_size,
            'dict_name': dict_name
        }
    
    return markers_info


def calculate_relative_pose(marker1_info, marker2_info):
    """
    计算两个标记之间的相对位姿
    
    Args:
        marker1_info: 第一个标记的信息（参考标记）
        marker2_info: 第二个标记的信息（目标标记）
    
    Returns:
        relative_pose: 相对位姿信息
    """
    if marker1_info is None or marker2_info is None:
        return None
    
    # 获取两个标记的旋转矩阵和平移向量
    R1 = marker1_info['R']  # 参考标记的旋转矩阵
    t1 = marker1_info['tvec'].flatten()  # 参考标记的平移向量
    
    R2 = marker2_info['R']  # 目标标记的旋转矩阵
    t2 = marker2_info['tvec'].flatten()  # 目标标记的平移向量
    
    # 计算相对旋转矩阵 (R_rel = R2 * R1^T)
    R1_inv = R1.T  # R1的逆矩阵
    R_rel = R2 @ R1_inv
    
    # 计算相对平移向量 (t_rel = t2 - R_rel * t1)
    t_rel = t2 - R_rel @ t1
    
    # 计算相对欧拉角
    euler_rel = rotation_matrix_to_euler_angles(R_rel)
    
    # 计算相对距离
    distance_rel = np.linalg.norm(t_rel)
    
    # 直接计算两个标记在相机坐标系下的距离
    distance_in_camera_frame = np.linalg.norm(t2 - t1)
    
    relative_pose = {
        'reference_marker_id': marker1_info['id'],
        'target_marker_id': marker2_info['id'],
        'R_relative': R_rel,
        't_relative': t_rel,
        'euler_relative': euler_rel,
        'distance_relative': distance_rel,
        'distance_in_camera_frame': distance_in_camera_frame
    }
    
    return relative_pose


def analyze_marker_errors(markers_info: Dict, squares_x: int, squares_y: int, square_length: float) -> Dict:
    """
    分析每个标记相对于其他标记的误差
    
    Args:
        markers_info: 检测到的标记信息
        squares_x: 标定板X方向方格数
        squares_y: 标定板Y方向方格数
        square_length: 方格边长
    
    Returns:
        error_analysis: 误差分析结果
    """
    error_analysis = {}
    marker_ids = list(markers_info.keys())
    
    print(f"\n=== 开始误差分析 ===")
    print(f"检测到的标记ID: {marker_ids}")
    
    for ref_marker_id in marker_ids:
        print(f"\n分析标记 {ref_marker_id} 的误差...")
        
        ref_marker_info = markers_info[ref_marker_id]
        rotation_errors = []
        distance_errors = []
        target_markers = []
        
        for target_marker_id in marker_ids:
            if target_marker_id == ref_marker_id:
                continue
            
            target_marker_info = markers_info[target_marker_id]
            
            # 计算相对位姿
            relative_pose = calculate_relative_pose(ref_marker_info, target_marker_info)
            if relative_pose is None:
                continue
            
            # 计算旋转误差（真值应该是[0, 0, 0]度）
            euler_error = np.abs(relative_pose['euler_relative'])
            rotation_error = np.linalg.norm(euler_error)  # 欧拉角误差的范数
            
            # 计算距离误差
            measured_distance = relative_pose['distance_in_camera_frame']
            ground_truth_distance = calculate_ground_truth_distance(
                ref_marker_id, target_marker_id, squares_x, squares_y, square_length
            )
            distance_error = abs(measured_distance - ground_truth_distance)
            
            rotation_errors.append(rotation_error)
            distance_errors.append(distance_error)
            target_markers.append(target_marker_id)
            
            print(f"  相对于标记 {target_marker_id}:")
            print(f"    旋转误差: {rotation_error:.4f} 度")
            print(f"    距离误差: {distance_error:.4f} 米")
            print(f"    测量距离: {measured_distance:.4f} 米")
            print(f"    真实距离: {ground_truth_distance:.4f} 米")
        
        # 计算统计信息
        if rotation_errors:
            error_analysis[ref_marker_id] = {
                'rotation_errors': rotation_errors,
                'distance_errors': distance_errors,
                'target_markers': target_markers,
                'mean_rotation_error': np.mean(rotation_errors),
                'std_rotation_error': np.std(rotation_errors),
                'max_rotation_error': np.max(rotation_errors),
                'mean_distance_error': np.mean(distance_errors),
                'std_distance_error': np.std(distance_errors),
                'max_distance_error': np.max(distance_errors)
            }
            
            print(f"  标记 {ref_marker_id} 统计信息:")
            print(f"    平均旋转误差: {np.mean(rotation_errors):.4f} ± {np.std(rotation_errors):.4f} 度")
            print(f"    最大旋转误差: {np.max(rotation_errors):.4f} 度")
            print(f"    平均距离误差: {np.mean(distance_errors):.4f} ± {np.std(distance_errors):.4f} 米")
            print(f"    最大距离误差: {np.max(distance_errors):.4f} 米")
    
    return error_analysis


def visualize_marker_errors(image_path: str, markers_info: Dict, error_analysis: Dict, 
                           ref_marker_id: int, output_dir: str, squares_x: int, square_length: float, bar_metric: str = 'rotation'):
    """
    可视化单个标记的误差分析结果（3D版）
    使用标定板世界坐标将ID放置在平面z=0上，并用线段显示与参考标记的误差
    """
    if ref_marker_id not in error_analysis:
        print(f"标记 {ref_marker_id} 没有误差分析数据")
        return
    
    # 创建3D图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制所有标记在世界坐标系的中心点
    error_data = error_analysis[ref_marker_id]
    target_markers = error_data['target_markers']
    rotation_errors = error_data['rotation_errors']
    distance_errors = error_data['distance_errors']

    # 从 markers_info 推断棋盘参数（需要主程序传入，故在外层管理）
    # 这里不做推断，留由主程序在标题中展示

    # 绘制所有检测到的标记（世界坐标）
    world_points = {}
    for marker_id in markers_info.keys():
        pos = get_marker_world_position(marker_id, squares_x, square_length)
        world_points[marker_id] = pos

    # 散点图：所有标记
    all_xyz = np.stack(list(world_points.values()), axis=0)
    ax.scatter(all_xyz[:, 0], all_xyz[:, 1], all_xyz[:, 2], c='gray', s=30, label='All Markers')

    # 高亮参考标记
    ref_pos = world_points.get(ref_marker_id, None)
    if ref_pos is not None:
        ax.scatter([ref_pos[0]], [ref_pos[1]], [ref_pos[2]], c='red', s=80, label=f'Ref {ref_marker_id}')
        ax.text(ref_pos[0], ref_pos[1], ref_pos[2], f"{ref_marker_id}", color='red')

    # 用柱状高度表示误差，不画连线
    # bar_metric: 'distance' 使用距离误差作高度；'rotation' 使用旋转误差作高度
    for i, target_id in enumerate(target_markers):
        tpos = world_points.get(target_id, None)
        if tpos is None:
            continue
        height = distance_errors[i] if bar_metric == 'distance' else rotation_errors[i]
        color = 'blue'
        ax.bar3d(tpos[0]-0.5*square_length, tpos[1]-0.5*square_length, 0.0,
                 0.6*square_length, 0.6*square_length, height,
                 shade=True, color=color, alpha=0.8)
        # 顶部标高文字（带单位）
        label_txt = f"{height:.3f}m" if bar_metric == 'distance' else f"{height:.1f}°"
        ax.text(tpos[0], tpos[1], height, label_txt, color=color, fontsize=8)
        # 标注目标ID
        ax.text(tpos[0], tpos[1], 0.0, f"{target_id}", color='black', fontsize=9)

    # 轴设置
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    z_label = 'Z (m)' if bar_metric == 'distance' else 'Z (deg)'
    ax.set_zlabel(z_label)
    ax.set_title(f'Marker {ref_marker_id} Error Analysis (3D)')

    # 设置平面范围（根据所有点自动扩展）
    min_x, max_x = np.min(all_xyz[:,0]), np.max(all_xyz[:,0])
    min_y, max_y = np.min(all_xyz[:,1]), np.max(all_xyz[:,1])
    pad_x = max(1e-6, 0.5 * (max_x - min_x))
    pad_y = max(1e-6, 0.5 * (max_y - min_y))
    ax.set_xlim(min_x - pad_x, max_x + pad_x)
    ax.set_ylim(min_y - pad_y, max_y + pad_y)

    # 设置Z轴范围，避免柱状超出画面
    vals = np.array(distance_errors if bar_metric == 'distance' else rotation_errors)
    vmax = float(np.max(vals)) if len(vals) > 0 else 1.0
    ax.set_zlim(0.0, vmax * 1.2 + 1e-6)

    ax.legend(loc='upper right')
    fig.tight_layout()

    # 输出保存
    output_path = Path(output_dir) / f'marker_{ref_marker_id}_errors_3d.png'
    fig.savefig(str(output_path), dpi=200, bbox_inches='tight')
    plt.close(fig)

    # 保持原先JSON统计输出
    stats = {
        'ref_marker_id': int(ref_marker_id),
        'target_markers': [int(t) for t in target_markers],
        'rotation_errors_deg': [float(v) for v in rotation_errors],
        'distance_errors_m': [float(v) for v in distance_errors],
        'mean_rotation_error_deg': float(error_data['mean_rotation_error']),
        'std_rotation_error_deg': float(error_data['std_rotation_error']),
        'max_rotation_error_deg': float(error_data['max_rotation_error']),
        'mean_distance_error_m': float(error_data['mean_distance_error']),
        'std_distance_error_m': float(error_data['std_distance_error']),
        'max_distance_error_m': float(error_data['max_distance_error'])
    }
    stats_path = Path(output_dir) / f'marker_{ref_marker_id}_stats.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description='ArUco标记误差分析工具')
    parser.add_argument('--image', required=True, help='输入图片路径')
    parser.add_argument('--camera-params', default='head_intrinsic_params.json', help='相机内参JSON文件路径')
    parser.add_argument('--dict', default='5x5_1000',
                        choices=['4x4_50','4x4_100','4x4_250','4x4_1000','5x5_50','5x5_100','5x5_250','5x5_1000',
                                 '6x6_50','6x6_100','6x6_250','6x6_1000','7x7_50','7x7_100','7x7_250','7x7_1000',
                                 'aruco_original'], help='ArUco字典类型')
    parser.add_argument('--marker-size', type=float, default=0.1, help='ArUco标记边长(米)')
    parser.add_argument('--squares-x', type=int, default=7, help='标定板X方向方格数')
    parser.add_argument('--squares-y', type=int, default=10, help='标定板Y方向方格数')
    parser.add_argument('--square-length', type=float, default=0.03, help='标定板方格边长(米)')
    parser.add_argument('--output-dir', default=None, help='输出目录，默认在output下根据图片名创建')
    parser.add_argument('--bar-metric', default='rotation', choices=['distance','rotation'], help='3D柱状高度表示的误差类型')
    args = parser.parse_args()

    # 检查输入
    if not os.path.exists(args.image):
        print(f'错误: 图片文件不存在: {args.image}')
        return
    if not os.path.exists(args.camera_params):
        print(f'错误: 相机内参文件不存在: {args.camera_params}')
        return

    # 输出目录
    default_dir = Path('output') / f"aruco_error_analysis_{Path(args.image).stem}"
    out_dir = Path(args.output_dir) if args.output_dir else default_dir
    ensure_dir(str(out_dir))
    print(f'输出目录: {out_dir}')

    # 加载相机参数
    camera_matrix, dist_coeffs = load_camera_params(args.camera_params)
    if camera_matrix is None:
        return

    # 检测所有标记
    markers_info = detect_aruco_markers(args.image, camera_matrix, dist_coeffs, args.dict, args.marker_size)
    if len(markers_info) < 2:
        print(f'错误: 只检测到 {len(markers_info)} 个标记，至少需要2个进行误差分析')
        return

    # 误差分析
    error_analysis = analyze_marker_errors(markers_info, args.squares_x, args.squares_y, args.square_length)

    # 可视化度量选择：'distance' 或 'rotation'
    bar_metric = args.bar_metric
    
    # 为每个标记生成3D图像与统计
    for ref_marker_id in markers_info.keys():
        visualize_marker_errors(args.image, markers_info, error_analysis, ref_marker_id, str(out_dir), args.squares_x, args.square_length, bar_metric=bar_metric)

    # 保存整体结果
    summary_path = out_dir / 'summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(error_analysis, f, indent=2, ensure_ascii=False, default=lambda o: float(o) if isinstance(o, (np.floating,)) else o)


if __name__ == '__main__':
    main()
