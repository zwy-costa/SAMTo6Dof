#!/usr/bin/env python3
"""
ArUco标记6D位姿检测脚本
专门检测指定ID的ArUco标记并计算其6D位姿
"""

from errno import EMULTIHOP
import json
import numpy as np
import cv2
import argparse
import os


def load_camera_params(json_file):
    """
    从JSON文件加载相机内参
    
    Args:
        json_file: 相机内参JSON文件路径
    
    Returns:
        camera_matrix: 3x3相机内参矩阵
        dist_coeffs: 畸变系数
    """
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
    """
    获取ArUco字典
    
    Args:
        dict_name: 字典名称，如 '5x5_1000'
    
    Returns:
        aruco_dict: ArUco字典对象
    """
    # 预定义的ArUco字典映射
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
            # 尝试新版本的API
            return cv2.aruco.getPredefinedDictionary(dict_mapping[dict_name])
        except AttributeError:
            try:
                # 尝试旧版本的API
                return cv2.aruco.Dictionary_get(dict_mapping[dict_name])
            except AttributeError:
                print("错误: 无法获取ArUco字典，请检查OpenCV版本")
                return None
    else:
        print(f"不支持的字典类型: {dict_name}")
        print(f"支持的字典类型: {list(dict_mapping.keys())}")
        return None


def detect_aruco_pose(image_path, camera_matrix, dist_coeffs, target_id, 
                     dict_name='5x5_1000', marker_size=0.1):
    """
    检测指定ID的ArUco标记位姿
    使用ArUco内置函数直接获取中心位置和姿态
    
    Args:
        image_path: 图片路径
        camera_matrix: 相机内参矩阵
        dist_coeffs: 畸变系数
        target_id: 目标ArUco ID
        dict_name: ArUco字典名称
        marker_size: 标记边长（米）
    
    Returns:
        pose_info: 位姿信息字典
    """
    # 获取ArUco字典
    aruco_dict = get_aruco_dict(dict_name)
    if aruco_dict is None:
        return None
    
    # 创建ArUco检测器参数
    try:
        # 尝试新版本的API
        aruco_params = cv2.aruco.DetectorParameters()
    except AttributeError:
        try:
            # 尝试旧版本的API
            aruco_params = cv2.aruco.DetectorParameters_create()
        except AttributeError:
            print("错误: 无法创建ArUco检测器参数，请检查OpenCV版本")
            return None
    
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图片: {image_path}")
        return None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(f"图片尺寸: {image.shape}")
    
    # 检测ArUco标记
    try:
        # 尝试新版本的API
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        corners, ids, rejected = detector.detectMarkers(gray)
    except AttributeError:
        try:
            # 尝试旧版本的API
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray, aruco_dict, parameters=aruco_params
            )
        except AttributeError:
            print("错误: 无法检测ArUco标记，请检查OpenCV版本")
            return None
    
    if ids is None:
        print("未检测到任何ArUco标记")
        return None
    
    print(f"检测到 {len(ids)} 个ArUco标记")
    print(f"检测到的ID: {ids.flatten()}")
    
    # 查找目标ID
    target_found = False
    pose_info = None
    
    for i, marker_id in enumerate(ids.flatten()):
        if marker_id == target_id:
            target_found = True
            print(f"\n找到目标ID: {target_id}")
            
            # 使用ArUco内置函数直接获取位姿
            try:
                # 使用estimatePoseSingleMarkers（如果可用）
                try:
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        [corners[i]], marker_size, camera_matrix, dist_coeffs
                    )
                    rvec = rvecs[0][0]  # 旋转向量 (3x1)
                    tvec = tvecs[0][0]  # 平移向量 (3x1)
                    print("使用 estimatePoseSingleMarkers 获取位姿")
                except AttributeError:
                    # 如果estimatePoseSingleMarkers不可用，使用solvePnP
                    # 定义ArUco标记的3D点（假设标记在Z=0平面上）
                    # marker_points = np.array([
                    #     [-marker_size/2, marker_size/2, 0],
                    #     [marker_size/2, marker_size/2, 0],
                    #     [marker_size/2, -marker_size/2, 0],
                    #     [-marker_size/2, -marker_size/2, 0]
                    # ], dtype=np.float32)
                    marker_points = np.array([
                        [-marker_size/2, -marker_size/2, 0],
                        [marker_size/2, -marker_size/2, 0],
                        [marker_size/2, marker_size/2, 0],
                        [-marker_size/2, marker_size/2, 0]
                    ], dtype=np.float32)
                    
                    # 使用solvePnP估计位姿
                    success, rvec, tvec = cv2.solvePnP(
                        marker_points, 
                        corners[i].astype(np.float32), 
                        camera_matrix, 
                        dist_coeffs
                    )
                    
                    if not success:
                        print("位姿估计失败: solvePnP未收敛")
                        return None
                    
                    print("使用 solvePnP 获取位姿")
                    
            except Exception as e:
                print(f"位姿估计失败: {e}")
                return None
            
            # 将旋转向量转换为旋转矩阵
            R, _ = cv2.Rodrigues(rvec)
            
            # 计算欧拉角
            euler_angles = rotation_matrix_to_euler_angles(R)
            
            # 计算距离
            distance = np.linalg.norm(tvec)
            
            # 计算标记中心点（四个角点的平均值）
            center = np.mean(corners[i], axis=0)
            
            # 计算标记的面积
            area = cv2.contourArea(corners[i].astype(np.int32))
            
            pose_info = {
                'id': target_id,
                'corners': corners[i],
                'center': center,
                'area': area,
                'rvec': rvec,
                'tvec': tvec,
                'R': R,
                'euler_angles': euler_angles,
                'distance': distance,
                'marker_size': marker_size,
                'dict_name': dict_name,
                'detection_time': 'N/A',  # 可以添加实际检测时间
                'confidence': 'N/A'  # 可以添加置信度信息
            }
            
            break
    
    if not target_found:
        print(f"未找到目标ID: {target_id}")
        return None
    
    return pose_info


def rotation_matrix_to_euler_angles(R):
    """
    将旋转矩阵转换为欧拉角 (ZYX顺序)
    
    Args:
        R: 3x3旋转矩阵
    
    Returns:
        euler_angles: [rx, ry, rz] 欧拉角（度）
    """
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


def print_pose_info(pose_info):
    """
    打印位姿信息
    
    Args:
        pose_info: 位姿信息字典
    """
    if pose_info is None:
        print("无位姿信息")
        return
    
    print(f"\n=== ArUco ID {pose_info['id']} 位姿信息 ===")
    print("=" * 50)
    
    # 基本信息
    print(f"标记ID: {pose_info['id']}")
    print(f"标记尺寸: {pose_info['marker_size']} 米")
    
    # 图像坐标
    center = pose_info['center']
    # 确保center是正确格式的中心点
    if len(center.shape) > 1:
        # 如果center是多个点的数组，计算平均值
        center = np.mean(center, axis=0)
    print(f"中心点: ({float(center[0])}, {float(center[1])})")
    
    # 标记几何信息
    area = pose_info['area']
    print(f"标记面积: {float(area)} 像素²")
    
    # 旋转信息
    rvec = pose_info['rvec'].flatten()
    R = pose_info['R']
    tvec = pose_info['tvec'].flatten()

    print(f"\n旋转向量 (弧度): [{rvec[0]}, {rvec[1]}, {rvec[2]}]")
    print(f"旋转矩阵R:")
    print(f"{R}")
    # print(f"  [{R[0,0]}, {R[0,1]}, {R[0,2]}]")
    # print(f"  [{R[1,0]}, {R[1,1]}, {R[1,2]}]")
    # print(f"  [{R[2,0]}, {R[2,1]}, {R[2,2]}]")
    
    # 欧拉角
    euler = pose_info['euler_angles'].flatten()
    print(f"平移向量t: [{tvec[0]}, {tvec[1]}, {tvec[2]}]")
    # print(f"欧拉角 (roll, pitch, yaw): {euler[0]}, {euler[1]}, {euler[2]}")
    print(f"欧拉角 (roll, pitch, yaw): [{euler[0]}, {euler[1]}, {euler[2]}]")
    
    # 平移信息
    # print(f"位置: X={tvec[0]}m, Y={tvec[1]}m, Z={tvec[2]}m")
    
    # 距离
    distance = pose_info['distance']
    print(f"距离: {float(distance)} 米")
    
    # 坐标系说明
    print(f"\n坐标系说明:")
    print(f"  - 原点: ArUco标记的中心点")
    print(f"  - X轴: 红色，指向标记的右方向")
    print(f"  - Y轴: 绿色，指向标记的下方向") 
    print(f"  - Z轴: 蓝色，指向标记的前方向（垂直于标记平面）")
    print(f"  - 旋转向量: 相对于相机坐标系的旋转")
    print(f"  - 平移向量: 标记中心相对于相机的位置")
    
    # 检测质量信息
    print(f"\n检测质量信息:")
    print(f"  - 标记尺寸: {pose_info['marker_size']} 米")
    print(f"  - 图像面积: {float(pose_info['area'])} 像素²")
    print(f"  - 检测距离: {float(pose_info['distance'])} 米")
    
    # 相机坐标系说明
    print(f"\n相机坐标系说明:")
    print(f"  - 原点: 相机光心")
    print(f"  - X轴: 向右为正")
    print(f"  - Y轴: 向下为正")
    print(f"  - Z轴: 向前为正（指向被摄物体）")
    
    # 检测统计信息
    print(f"\n检测统计信息:")
    print(f"  - 检测时间: {pose_info.get('detection_time', 'N/A')}")
    print(f"  - 置信度: {pose_info.get('confidence', 'N/A')}")
    print(f"  - 标记族: {pose_info.get('dict_name', 'N/A')}")
    
    # 几何信息
    print(f"\n几何信息:")
    corners = pose_info['corners']
    print(f"  - 角点数量: {len(corners)}")
    print(f"  - 角点坐标:")
    for i, corner in enumerate(corners):
        for j in range(len(corner)):
            print(f"    角点{i+1}: ({float(corner[j][0])}, {float(corner[j][1])})")
    
    # 总结信息
    print(f"\n=== 检测总结 ===")
    print(f"✓ 成功检测到ArUco标记 ID: {pose_info['id']}")
    print(f"✓ 标记中心位置: ({float(center[0])}, {float(center[1])}) 像素")
    print(f"✓ 3D位置: X={tvec[0]}m, Y={tvec[1]}m, Z={tvec[2]}m")
    print(f"✓ 检测距离: {float(distance)} 米")
    print(f"✓ 标记尺寸: {pose_info['marker_size']} 米")
    print("=" * 50)


def draw_coordinate_system(image, pose_info, camera_matrix, dist_coeffs):
    """
    绘制ArUco标记的坐标系
    
    Args:
        image: 图像
        pose_info: 位姿信息
        camera_matrix: 相机内参矩阵
        dist_coeffs: 畸变系数
    """
    if camera_matrix is None or dist_coeffs is None:
        print("警告: 无法绘制坐标系，缺少相机参数")
        return
    
    try:
        # 尝试使用ArUco内置的坐标轴绘制函数
        cv2.aruco.drawAxis(image, camera_matrix, dist_coeffs, 
                          pose_info['rvec'], pose_info['tvec'], 0.1)
        print("使用ArUco内置函数绘制坐标系")
    except AttributeError:
        # 如果内置函数不可用，手动绘制坐标系
        print("使用手动绘制坐标系")
        draw_manual_coordinate_system(image, pose_info, camera_matrix, dist_coeffs)


def draw_manual_coordinate_system(image, pose_info, camera_matrix, dist_coeffs):
    """
    手动绘制坐标系
    
    Args:
        image: 图像
        pose_info: 位姿信息
        camera_matrix: 相机内参矩阵
        dist_coeffs: 畸变系数
    """
    # 获取旋转矩阵和平移向量
    rvec = pose_info['rvec'].flatten()
    tvec = pose_info['tvec'].flatten()
    
    # 定义坐标系的原点和三个轴的方向（以米为单位）
    axis_length = 0.1  # 坐标轴长度
    origin = np.array([0, 0, 0], dtype=np.float32)
    x_axis = np.array([axis_length, 0, 0], dtype=np.float32)  # X轴（红色）
    y_axis = np.array([0, axis_length, 0], dtype=np.float32)  # Y轴（绿色）
    z_axis = np.array([0, 0, axis_length], dtype=np.float32)  # Z轴（蓝色）
    
    # 将旋转向量转换为旋转矩阵
    R, _ = cv2.Rodrigues(rvec)
    
    # 变换坐标系点
    origin_transformed = R @ origin + tvec
    x_axis_transformed = R @ x_axis + tvec
    y_axis_transformed = R @ y_axis + tvec
    z_axis_transformed = R @ z_axis + tvec
    
    # 投影到图像平面
    points_3d = np.array([origin_transformed, x_axis_transformed, y_axis_transformed, z_axis_transformed], dtype=np.float32)
    points_2d, _ = cv2.projectPoints(points_3d, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)
    points_2d = points_2d.reshape(-1, 2).astype(np.int32)
    
    # 绘制坐标轴
    origin_2d = tuple(points_2d[0])
    x_axis_2d = tuple(points_2d[1])
    y_axis_2d = tuple(points_2d[2])
    z_axis_2d = tuple(points_2d[3])
    
    # 绘制X轴（红色）
    cv2.line(image, origin_2d, x_axis_2d, (0, 0, 255), 2)
    cv2.putText(image, 'X', x_axis_2d, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # 绘制Y轴（绿色）
    cv2.line(image, origin_2d, y_axis_2d, (0, 255, 0), 2)
    cv2.putText(image, 'Y', y_axis_2d, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # 绘制Z轴（蓝色）
    cv2.line(image, origin_2d, z_axis_2d, (255, 0, 0), 2)
    cv2.putText(image, 'Z', z_axis_2d, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # 绘制原点
    cv2.circle(image, origin_2d, 3, (255, 255, 255), -1)
    
    # 添加坐标系说明
    cv2.putText(image, 'ArUco Coordinate System', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def visualize_detection(image_path, pose_info, output_path=None):
    """
    可视化检测结果
    使用ArUco内置函数进行可视化，并绘制坐标系
    
    Args:
        image_path: 原始图片路径
        pose_info: 位姿信息
        output_path: 输出图片路径
    """
    if pose_info is None:
        return
    
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print("无法读取图片进行可视化")
        return
    
    # 使用ArUco内置函数绘制标记
    try:
        # 尝试使用ArUco内置的绘制函数
        corners = [pose_info['corners']]
        ids = np.array([[pose_info['id']]])
        
        # 绘制检测到的标记
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        print("使用ArUco内置函数绘制标记")
    except AttributeError:
        # 如果内置函数不可用，使用手动绘制
        corners = pose_info['corners'].astype(np.int32)
        cv2.polylines(image, [corners], True, (0, 255, 0), 2)
        
        # 绘制标记ID
        center = pose_info['center']
        # 确保center是正确格式的中心点
        if len(center.shape) > 1:
            # 如果center是多个点的数组，计算平均值
            center = np.mean(center, axis=0)
        center = center.astype(np.int32)
        cv2.putText(image, f"ID: {pose_info['id']}", 
                   (int(center[0]) - 20, int(center[1]) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print("使用手动绘制标记")
    
    # 绘制中心点
    center = pose_info['center']
    # 确保center是正确格式的中心点
    if len(center.shape) > 1:
        # 如果center是多个点的数组，计算平均值
        center = np.mean(center, axis=0)
    center = center.astype(np.int32)
    cv2.circle(image, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)
    
    # 绘制位姿信息
    tvec = pose_info['tvec']
    pose_text = f"Z: {float(tvec[2])}m"
    cv2.putText(image, pose_text, 
               (int(center[0]) - 20, int(center[1]) + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    # 绘制更多信息
    info_text = f"ID: {pose_info['id']} | Distance: {float(pose_info['distance'])}m"
    cv2.putText(image, info_text, 
               (10, image.shape[0] - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 绘制位姿详细信息
    tvec = pose_info['tvec'].flatten()
    pose_detail = f"X:{tvec[0]}m Y:{tvec[1]}m Z:{tvec[2]}m"
    cv2.putText(image, pose_detail, 
               (10, image.shape[0] - 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # 绘制坐标系（ArUco标记的坐标系）
    camera_matrix, dist_coeffs = load_camera_params('head_intrinsic_params.json')
    draw_coordinate_system(image, pose_info, camera_matrix, dist_coeffs)
    
    # 保存或显示结果
    if output_path:
        cv2.imwrite(output_path, image)
        print(f"可视化结果已保存到: {output_path}")
    else:
        # 显示图片
        cv2.imshow('ArUco Detection', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='ArUco标记6D位姿检测工具')
    parser.add_argument('--image', required=True, help='输入图片路径')
    parser.add_argument('--id', type=int, required=True, help='目标ArUco ID')
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
    
    print("=== ArUco标记6D位姿检测工具 ===")
    print(f"图片: {args.image}")
    print(f"目标ID: {args.id}")
    print(f"字典类型: {args.dict}")
    print(f"标记尺寸: {args.marker_size} 米")
    print("=" * 40)
    
    # 加载相机内参
    camera_matrix, dist_coeffs = load_camera_params(args.camera_params)
    if camera_matrix is None:
        return
    
    # 检测位姿
    pose_info = detect_aruco_pose(
        args.image, camera_matrix, dist_coeffs, 
        args.id, args.dict, args.marker_size
    )
    
    # 打印位姿信息
    print_pose_info(pose_info)
    
    # 可视化（如果需要）
    if args.visualize or args.output:
        visualize_detection(args.image, pose_info, args.output)


if __name__ == '__main__':
    main() 