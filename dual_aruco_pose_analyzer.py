#!/usr/bin/env python3
"""
双ArUco标记位姿关系分析脚本
识别两个给定ID的ArUco二维码，计算它们之间的位置和姿态关系
"""

import json
import numpy as np
import cv2
import argparse
import os
from typing import Optional, Tuple, Dict, Any, List


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


def detect_aruco_markers(image_path, camera_matrix, dist_coeffs, target_ids, 
                        dict_name='5x5_1000', marker_size=0.1):
    """
    检测指定ID的ArUco标记位姿
    
    Args:
        image_path: 图片路径
        camera_matrix: 相机内参矩阵
        dist_coeffs: 畸变系数
        target_ids: 目标ArUco ID列表
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
    
    # 查找目标ID
    markers_info = {}
    target_ids_set = set(target_ids)
    
    for i, marker_id in enumerate(ids.flatten()):
        if marker_id in target_ids_set:
            print(f"\n找到目标ID: {marker_id}")
            
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
    
    # 计算在参考标记坐标系下的位置
    # 将目标标记的位置转换到参考标记的坐标系
    t_in_ref_frame = R1_inv @ (t2 - t1)
    
    relative_pose = {
        'reference_marker_id': marker1_info['id'],
        'target_marker_id': marker2_info['id'],
        'R_relative': R_rel,
        't_relative': t_rel,
        'euler_relative': euler_rel,
        'distance_relative': distance_rel,
        't_in_reference_frame': t_in_ref_frame,
        'R_in_reference_frame': R_rel,  # 在参考坐标系下的旋转矩阵
        'euler_in_reference_frame': euler_rel  # 在参考坐标系下的欧拉角
    }
    
    return relative_pose


def print_marker_info(marker_info, marker_name="标记"):
    """打印单个标记的信息"""
    if marker_info is None:
        print(f"{marker_name}信息: 无")
        return
    
    print(f"\n=== {marker_name} ID {marker_info['id']} 位姿信息 ===")
    print("=" * 50)
    
    # 基本信息
    print(f"标记ID: {marker_info['id']}")
    print(f"标记尺寸: {marker_info['marker_size']} 米")
    
    # 图像坐标
    center = marker_info['center']
    if len(center.shape) > 1:
        center = np.mean(center, axis=0)
    print(f"中心点: ({float(center[0])}, {float(center[1])})")
    
    # 标记几何信息
    area = marker_info['area']
    print(f"标记面积: {float(area)} 像素²")
    
    # 旋转信息
    rvec = marker_info['rvec'].flatten()
    R = marker_info['R']
    tvec = marker_info['tvec'].flatten()

    print(f"\n旋转向量 (弧度): [{rvec[0]}, {rvec[1]}, {rvec[2]}]")
    print(f"旋转矩阵R:")
    print(f"{R}")
    
    # 欧拉角
    euler = marker_info['euler_angles'].flatten()
    print(f"平移向量t: [{tvec[0]}, {tvec[1]}, {tvec[2]}]")
    print(f"欧拉角 (roll, pitch, yaw): [{euler[0]}, {euler[1]}, {euler[2]}]")
    
    # 距离
    distance = marker_info['distance']
    print(f"距离: {float(distance)} 米")
    
    print("=" * 50)


def print_relative_pose_info(relative_pose):
    """打印相对位姿信息"""
    if relative_pose is None:
        print("相对位姿信息: 无")
        return
    
    print(f"\n=== 相对位姿分析 ===")
    print("=" * 50)
    print(f"参考标记ID: {relative_pose['reference_marker_id']}")
    print(f"目标标记ID: {relative_pose['target_marker_id']}")
    
    # 相对旋转信息
    R_rel = relative_pose['R_relative']
    euler_rel = relative_pose['euler_relative']
    print(f"\n相对旋转矩阵:")
    print(f"{R_rel}")
    print(f"相对欧拉角 (roll, pitch, yaw): [{euler_rel[0]}, {euler_rel[1]}, {euler_rel[2]}]")
    
    # 相对平移信息
    t_rel = relative_pose['t_relative']
    distance_rel = relative_pose['distance_relative']
    print(f"\n相对平移向量: [{t_rel[0]}, {t_rel[1]}, {t_rel[2]}]")
    print(f"相对距离: {distance_rel} 米")
    
    # 在参考坐标系下的位置
    t_in_ref = relative_pose['t_in_reference_frame']
    euler_in_ref = relative_pose['euler_in_reference_frame']
    print(f"\n在参考标记坐标系下的位置:")
    print(f"  位置向量: [{t_in_ref[0]}, {t_in_ref[1]}, {t_in_ref[2]}]")
    print(f"  欧拉角: [{euler_in_ref[0]}, {euler_in_ref[1]}, {euler_in_ref[2]}]")
    
    # 坐标系说明
    print(f"\n坐标系说明:")
    print(f"  - 参考标记坐标系: 以标记{relative_pose['reference_marker_id']}为中心")
    print(f"  - X轴: 指向标记的右方向")
    print(f"  - Y轴: 指向标记的下方向")
    print(f"  - Z轴: 垂直于标记平面，指向观察者")
    print(f"  - 相对位姿: 从参考标记到目标标记的变换")
    
    print("=" * 50)


def visualize_dual_markers(image_path, markers_info, relative_pose, output_path=None):
    """可视化双标记检测结果"""
    if not markers_info:
        return
    
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print("无法读取图片进行可视化")
        return
    
    # 绘制检测到的标记
    try:
        corners_list = []
        ids_list = []
        
        for marker_id, marker_info in markers_info.items():
            corners_list.append(marker_info['corners'])
            ids_list.append([marker_id])
        
        # 绘制检测到的标记
        cv2.aruco.drawDetectedMarkers(image, corners_list, np.array(ids_list))
        print("使用ArUco内置函数绘制标记")
    except (AttributeError, cv2.error):
        # 如果内置函数不可用，使用手动绘制
        for marker_id, marker_info in markers_info.items():
            corners = marker_info['corners'].astype(np.int32)
            cv2.polylines(image, [corners], True, (0, 255, 0), 2)
            
            # 绘制标记ID
            center = marker_info['center']
            if len(center.shape) > 1:
                center = np.mean(center, axis=0)
            center = center.astype(np.int32)
            cv2.putText(image, f"ID: {marker_id}", 
                       (int(center[0]) - 20, int(center[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print("使用手动绘制标记")
    
    # 绘制中心点和位姿信息
    for marker_id, marker_info in markers_info.items():
        center = marker_info['center']
        if len(center.shape) > 1:
            center = np.mean(center, axis=0)
        center = center.astype(np.int32)
        
        # 绘制中心点
        cv2.circle(image, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)
        
        # 绘制位姿信息
        tvec = marker_info['tvec']
        pose_text = f"Z: {float(tvec[2])}m"
        cv2.putText(image, pose_text, 
                   (int(center[0]) - 20, int(center[1]) + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    # 绘制相对位姿信息
    if relative_pose:
        # 在图像上显示相对距离
        info_text = f"相对距离: {relative_pose['distance_relative']}m"
        cv2.putText(image, info_text, 
                   (10, image.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 绘制连接线
        if len(markers_info) == 2:
            marker_ids = list(markers_info.keys())
            center1 = markers_info[marker_ids[0]]['center']
            center2 = markers_info[marker_ids[1]]['center']
            
            if len(center1.shape) > 1:
                center1 = np.mean(center1, axis=0)
            if len(center2.shape) > 1:
                center2 = np.mean(center2, axis=0)
            
            center1 = center1.astype(np.int32)
            center2 = center2.astype(np.int32)
            
            # 绘制连接线
            cv2.line(image, tuple(center1), tuple(center2), (255, 255, 0), 2)
            
            # 在连接线中点显示相对距离
            mid_point = ((center1[0] + center2[0]) // 2, (center1[1] + center2[1]) // 2)
            cv2.putText(image, f"{relative_pose['distance_relative']:.3f}m", 
                       mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    # 绘制坐标系
    camera_matrix, dist_coeffs = load_camera_params('head_intrinsic_params.json')
    if camera_matrix is not None and dist_coeffs is not None:
        for marker_id, marker_info in markers_info.items():
            try:
                cv2.aruco.drawAxis(image, camera_matrix, dist_coeffs, 
                                  marker_info['rvec'], marker_info['tvec'], 0.1)
            except AttributeError:
                pass  # 如果内置函数不可用，跳过坐标系绘制
    
    # 保存或显示结果
    if output_path:
        cv2.imwrite(output_path, image)
        print(f"可视化结果已保存到: {output_path}")
    else:
        # 显示图片
        cv2.imshow('Dual ArUco Detection', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='双ArUco标记位姿关系分析工具')
    parser.add_argument('--image', required=True, help='输入图片路径')
    parser.add_argument('--id1', type=int, required=True, help='第一个ArUco ID（参考标记）')
    parser.add_argument('--id2', type=int, required=True, help='第二个ArUco ID（目标标记）')
    parser.add_argument('--camera-params', default='head_intrinsic_params.json',
                       help='相机内参JSON文件路径')
    parser.add_argument('--dict', default='5x5_1000',
                       choices=['4x4_50', '4x4_100', '4x4_250', '4x4_1000',
                               '5x5_50', '5x5_100', '5x5_250', '5x5_1000',
                               '6x6_50', '6x6_100', '6x6_250', '6x6_1000',
                               '7x7_50', '7x7_100', '7x7_250', '7x7_1000',
                               'aruco_original'],
                       help='ArUco字典类型')
    parser.add_argument('--marker-size', type=float, default=0.1,
                       help='ArUco标记边长，单位：米')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='可视化检测结果')
    parser.add_argument('--output', help='输出图片路径')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.image):
        print(f"错误: 图片文件不存在: {args.image}")
        return
    
    if not os.path.exists(args.camera_params):
        print(f"错误: 相机内参文件不存在: {args.camera_params}")
        return
    
    print("=== 双ArUco标记位姿关系分析工具 ===")
    print(f"图片: {args.image}")
    print(f"参考标记ID: {args.id1}")
    print(f"目标标记ID: {args.id2}")
    print(f"字典类型: {args.dict}")
    print(f"标记尺寸: {args.marker_size} 米")
    print("=" * 50)
    
    # 加载相机内参
    camera_matrix, dist_coeffs = load_camera_params(args.camera_params)
    if camera_matrix is None:
        return
    
    # 检测ArUco标记
    target_ids = [args.id1, args.id2]
    markers_info = detect_aruco_markers(
        args.image, camera_matrix, dist_coeffs, 
        target_ids, args.dict, args.marker_size
    )
    
    # 检查是否检测到两个标记
    if len(markers_info) < 2:
        print(f"错误: 只检测到 {len(markers_info)} 个目标标记，需要检测到 2 个")
        print(f"检测到的标记ID: {list(markers_info.keys())}")
        return
    
    # 打印标记信息
    print_marker_info(markers_info.get(args.id1), f"参考标记 (ID:{args.id1})")
    print_marker_info(markers_info.get(args.id2), f"目标标记 (ID:{args.id2})")
    
    # 计算相对位姿
    relative_pose = calculate_relative_pose(
        markers_info[args.id1], markers_info[args.id2]
    )
    
    # 打印相对位姿信息
    print_relative_pose_info(relative_pose)
    
    # 可视化（如果需要）
    if args.visualize or args.output:
        visualize_dual_markers(args.image, markers_info, relative_pose, args.output)


if __name__ == '__main__':
    main() 