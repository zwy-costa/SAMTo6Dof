#!/usr/bin/env python3
"""
ArUco标记6D位姿检测类
专门检测指定ID的ArUco标记并计算其6D位姿
"""

import json
import numpy as np
import cv2
import os
from typing import Optional, Tuple, Dict, Any


class ArUcoPoseEstimator:
    def __init__(self, intrinsic_path="./head_intrinsic_params.json"):
        """初始化ArUco位姿估计器"""
        self.K, self.dist_coeffs = self._load_camera_intrinsics(intrinsic_path)
        
    def _load_camera_intrinsics(self, intrinsic_path):
        """加载相机内参"""
        try:
            with open(intrinsic_path, 'r') as f:
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
    
    def _get_aruco_dict(self, dict_name):
        """获取ArUco字典"""
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
    
    def _rotation_matrix_to_euler_angles(self, R):
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
    
    def _get_aruco_corner_order(self):
        """获取ArUco标记的标准角点顺序"""
        corner_order = {
            'description': 'ArUco标记的标准角点顺序（从左上角开始，顺时针）',
            'corners': [
                'corner[0]: 左上角 (Top-Left)',
                'corner[1]: 右上角 (Top-Right)', 
                'corner[2]: 右下角 (Bottom-Right)',
                'corner[3]: 左下角 (Bottom-Left)'
            ],
            'direction': 'ArUco标记内部有方向性图案，可以确定标记的朝向',
            'coordinate_system': {
                'origin': '标记中心',
                'x_axis': '从左上角指向右上角',
                'y_axis': '从左上角指向左下角', 
                'z_axis': '垂直于标记平面，指向观察者'
            }
        }
        return corner_order
    
    def _validate_aruco_orientation(self, corners):
        """验证ArUco标记的方向是否正确"""
        if len(corners) != 4:
            return False, "角点数量不正确"
        
        # 检查是否为正方形（允许一定的透视变形）
        edges = []
        for i in range(4):
            pt1 = corners[i]
            pt2 = corners[(i+1) % 4]
            edge_length = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
            edges.append(edge_length)
        
        # 检查边长比例
        max_edge = max(edges)
        min_edge = min(edges)
        aspect_ratio = max_edge / min_edge if min_edge > 0 else float('inf')
        
        if aspect_ratio > 1.5:  # 允许50%的变形
            return False, f"标记变形过大，长宽比: {aspect_ratio}"
        
        # 检查是否为顺时针顺序
        # 计算有向面积
        area = 0
        for i in range(4):
            pt1 = corners[i]
            pt2 = corners[(i+1) % 4]
            area += (pt2[0] - pt1[0]) * (pt2[1] + pt1[1])
        
        if area < 0:
            return False, "角点顺序不是顺时针"
        
        return True, "方向验证通过"
    
    def _analyze_aruco_direction_features(self, corners):
        """分析ArUco标记的方向特征"""
        if len(corners) != 4:
            return None
        
        # 计算边长
        edges = []
        for i in range(4):
            pt1 = corners[i]
            pt2 = corners[(i+1) % 4]
            edge_length = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
            edges.append(edge_length)
        
        # 计算角度
        angles = []
        for i in range(4):
            pt1 = corners[i]
            pt2 = corners[(i+1) % 4]
            pt3 = corners[(i+2) % 4]
            
            vec1 = pt2 - pt1
            vec2 = pt3 - pt2
            
            dot_product = np.dot(vec1, vec2)
            norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
            if norms > 0:
                cos_angle = dot_product / norms
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle) * 180 / np.pi
                angles.append(angle)
        
        # 计算中心点
        center = np.mean(corners, axis=0)
        
        # 计算面积
        area = cv2.contourArea(corners.astype(np.int32))
        
        features = {
            'edges': edges,
            'angles': angles,
            'center': center,
            'area': area,
            'aspect_ratio': max(edges) / min(edges) if min(edges) > 0 else float('inf'),
            'is_square': abs(max(angles) - min(angles)) < 10 if len(angles) == 4 else False
        }
        
        return features
    
    def _visualize_aruco_internal_direction(self, image, corners, pose_info, output_path=None):
        """可视化ArUco标记的内部方向"""
        vis_img = image.copy()
        
        # 确保corners是正确的格式
        if len(corners.shape) == 3:
            # 如果是嵌套数组格式，取第一个元素
            corners_for_draw = corners[0]
        else:
            corners_for_draw = corners
            
        # 绘制标记边界
        corners_int = corners_for_draw.astype(np.int32)
        cv2.polylines(vis_img, [corners_int], True, (0, 255, 0), 2)
        
        # 绘制角点并标注顺序
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # 蓝、绿、红、青
        labels = ['TL', 'TR', 'BR', 'BL']  # Top-Left, Top-Right, Bottom-Right, Bottom-Left
        
        # 确保corners是正确的格式
        if len(corners.shape) == 3:
            # 如果是嵌套数组格式，取第一个元素
            corners_for_draw = corners[0]
        else:
            corners_for_draw = corners
            
        for i, (corner, color, label) in enumerate(zip(corners_for_draw, colors, labels)):
            x, y = int(corner[0]), int(corner[1])
            cv2.circle(vis_img, (x, y), 8, color, -1)
            cv2.circle(vis_img, (x, y), 8, (255, 255, 255), 2)
            cv2.putText(vis_img, f"{i}:{label}", (x+10, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 绘制中心点
        center = pose_info['center']
        if len(center.shape) > 1:
            center = np.mean(center, axis=0)
        center_x, center_y = int(center[0]), int(center[1])
        cv2.circle(vis_img, (center_x, center_y), 5, (255, 255, 255), -1)
        cv2.putText(vis_img, "Center", (center_x+10, center_y+10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 绘制方向箭头（从左上角到右上角）
        tl = corners_for_draw[0].astype(np.int32)
        tr = corners_for_draw[1].astype(np.int32)
        cv2.arrowedLine(vis_img, tuple(tl), tuple(tr), (0, 255, 255), 3, tipLength=0.2)
        cv2.putText(vis_img, "X-axis", (tr[0]+5, tr[1]-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # 绘制方向箭头（从左上角到左下角）
        bl = corners_for_draw[3].astype(np.int32)
        cv2.arrowedLine(vis_img, tuple(tl), tuple(bl), (255, 0, 255), 3, tipLength=0.2)
        cv2.putText(vis_img, "Y-axis", (bl[0]+5, bl[1]+15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # 添加图例
        legend_y = 30
        cv2.putText(vis_img, "ArUco Internal Direction", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        legend_y += 25
        cv2.putText(vis_img, "0:TL 1:TR 2:BR 3:BL", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        legend_y += 20
        cv2.putText(vis_img, f"ID: {pose_info['id']}", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        legend_y += 20
        cv2.putText(vis_img, f"Distance: {pose_info['distance']}m", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if output_path:
            cv2.imwrite(output_path, vis_img)
            print(f"ArUco内部方向可视化已保存到: {output_path}")
        
        return vis_img
    
    def detect_aruco_pose(self, image_path, target_id, dict_name='5x5_1000', 
                         marker_size=0.1, output_dir="./output/", base_name="aruco_result"):
        """
        检测指定ID的ArUco标记位姿
        
        Args:
            image_path: 图片路径
            target_id: 目标ArUco ID
            dict_name: ArUco字典名称
            marker_size: 标记边长（米）
            output_dir: 输出目录
            base_name: 输出文件基础名称
            
        Returns:
            pose_info: 位姿信息字典，如果检测失败返回None
        """
        # 获取ArUco字典
        aruco_dict = self._get_aruco_dict(dict_name)
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
                
                # 获取当前角点
                current_corners = corners[i]
                
                # 确保center是正确格式的中心点
                center = np.mean(current_corners, axis=0)
                if len(center.shape) > 1:
                    center = np.mean(center, axis=0)
                
                # 获取ArUco角点顺序信息
                corner_order_info = self._get_aruco_corner_order()
                print(f"ArUco角点顺序: {corner_order_info['description']}")
                
                # 验证ArUco方向
                is_valid, validation_msg = self._validate_aruco_orientation(current_corners)
                print(f"方向验证: {validation_msg}")
                
                # 分析方向特征
                direction_features = self._analyze_aruco_direction_features(current_corners)
                if direction_features:
                    print(f"方向特征分析:")
                    print(f"  边长: {direction_features['edges']}")
                    print(f"  角度: {direction_features['angles']}")
                    print(f"  长宽比: {direction_features['aspect_ratio']}")
                    print(f"  是否为正方形: {direction_features['is_square']}")
                
                # 使用ArUco内置函数直接获取位姿
                try:
                    # 使用estimatePoseSingleMarkers（如果可用）
                    try:
                        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                            [current_corners], marker_size, self.K, self.dist_coeffs
                        )
                        rvec = rvecs[0][0]  # 旋转向量 (3x1)
                        tvec = tvecs[0][0]  # 平移向量 (3x1)
                        print("使用 estimatePoseSingleMarkers 获取位姿")
                    except AttributeError:
                        # 如果estimatePoseSingleMarkers不可用，使用solvePnP
                        # 定义ArUco标记的3D点（按照ArUco标准角点顺序）
                        # ArUco角点顺序：左上角[0] -> 右上角[1] -> 右下角[2] -> 左下角[3]
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
                            self.K, 
                            self.dist_coeffs
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
                euler_angles = self._rotation_matrix_to_euler_angles(R)
                
                # 计算距离
                distance = np.linalg.norm(tvec)
                
                # 计算标记的面积
                area = cv2.contourArea(current_corners.astype(np.int32))
                
                pose_info = {
                    'id': target_id,
                    'corners': current_corners,
                    'center': center,
                    'area': area,
                    'rvec': rvec,
                    'tvec': tvec,
                    'R': R,
                    'euler_angles': euler_angles,
                    'distance': distance,
                    'marker_size': marker_size,
                    'dict_name': dict_name,
                    'detection_time': 'N/A',
                    'confidence': 'N/A',
                    'corner_order_info': corner_order_info,
                    'direction_features': direction_features,
                    'validation_result': {'is_valid': is_valid, 'message': validation_msg}
                }
                
                # 可视化ArUco内部方向
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    direction_vis_path = os.path.join(output_dir, base_name + "_internal_direction.jpg")
                    self._visualize_aruco_internal_direction(image, current_corners, pose_info, direction_vis_path)
                
                break
        
        if not target_found:
            print(f"未找到目标ID: {target_id}")
            return None
        
        return pose_info
    
    def print_pose_info(self, pose_info):
        """打印位姿信息"""
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
        
        # 欧拉角
        euler = pose_info['euler_angles'].flatten()
        print(f"平移向量t: [{tvec[0]}, {tvec[1]}, {tvec[2]}]")
        print(f"欧拉角 (roll, pitch, yaw): [{euler[0]}, {euler[1]}, {euler[2]}]")
        
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
        
        # 方向验证信息
        if 'validation_result' in pose_info:
            validation = pose_info['validation_result']
            print(f"\n方向验证:")
            print(f"  - 验证结果: {'通过' if validation['is_valid'] else '失败'}")
            print(f"  - 验证信息: {validation['message']}")
        
        # 方向特征信息
        if 'direction_features' in pose_info and pose_info['direction_features']:
            features = pose_info['direction_features']
            print(f"\n方向特征:")
            print(f"  - 边长: {features['edges']}")
            print(f"  - 角度: {features['angles']}")
            print(f"  - 长宽比: {features['aspect_ratio']}")
            print(f"  - 是否为正方形: {features['is_square']}")
        
        # 总结信息
        print(f"\n=== 检测总结 ===")
        print(f"✓ 成功检测到ArUco标记 ID: {pose_info['id']}")
        print(f"✓ 标记中心位置: ({float(center[0])}, {float(center[1])}) 像素")
        print(f"✓ 3D位置: X={tvec[0]}m, Y={tvec[1]}m, Z={tvec[2]}m")
        print(f"✓ 检测距离: {float(distance)} 米")
        print(f"✓ 标记尺寸: {pose_info['marker_size']} 米")
        print("=" * 50)
    
    def _draw_coordinate_system(self, image, pose_info):
        """绘制ArUco标记的坐标系"""
        if self.K is None or self.dist_coeffs is None:
            print("警告: 无法绘制坐标系，缺少相机参数")
            return
        
        try:
            # 尝试使用ArUco内置的坐标轴绘制函数
            cv2.aruco.drawAxis(image, self.K, self.dist_coeffs, 
                              pose_info['rvec'], pose_info['tvec'], 0.1)
            print("使用ArUco内置函数绘制坐标系")
        except AttributeError:
            # 如果内置函数不可用，手动绘制坐标系
            print("使用手动绘制坐标系")
            self._draw_manual_coordinate_system(image, pose_info)
    
    def _draw_manual_coordinate_system(self, image, pose_info):
        """手动绘制坐标系"""
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
        points_2d, _ = cv2.projectPoints(points_3d, np.zeros(3), np.zeros(3), self.K, self.dist_coeffs)
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
    
    def visualize_detection(self, image_path, pose_info, output_path=None):
        """可视化检测结果"""
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
            
            # 检查corners格式
            corners_for_draw = corners[0]
            if corners_for_draw.shape == (4, 2):
                # 绘制检测到的标记
                cv2.aruco.drawDetectedMarkers(image, corners, ids)
                print("使用ArUco内置函数绘制标记")
            else:
                raise AttributeError("角点格式不正确")
        except (AttributeError, cv2.error):
            # 如果内置函数不可用或格式不正确，使用手动绘制
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
        self._draw_coordinate_system(image, pose_info)
        
        # 保存或显示结果
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"可视化结果已保存到: {output_path}")
        else:
            # 显示图片
            cv2.imshow('ArUco Detection', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    def estimate_pose(self, image_path, target_id, dict_name='5x5_1000', 
                     marker_size=0.1, output_dir="./output/", base_name="aruco_result",
                     visualize=True, save_visualization=True):
        """
        估计ArUco标记的6D位姿
        
        Args:
            image_path: 图片路径
            target_id: 目标ArUco ID
            dict_name: ArUco字典名称
            marker_size: 标记边长（米）
            output_dir: 输出目录
            base_name: 输出文件基础名称
            visualize: 是否可视化结果
            save_visualization: 是否保存可视化结果
            
        Returns:
            pose_info: 位姿信息字典，如果检测失败返回None
        """
        print("=== ArUco标记6D位姿检测工具 ===")
        print(f"图片: {image_path}")
        print(f"目标ID: {target_id}")
        print(f"字典类型: {dict_name}")
        print(f"标记尺寸: {marker_size} 米")
        print("=" * 40)
        
        # 检查输入文件
        if not os.path.exists(image_path):
            print(f"错误: 图片文件不存在: {image_path}")
            return None
        
        # 检测位姿
        pose_info = self.detect_aruco_pose(
            image_path, target_id, dict_name, marker_size, output_dir, base_name
        )
        
        if pose_info is None:
            print("位姿检测失败")
            return None
        
        # 打印位姿信息
        self.print_pose_info(pose_info)
        
        # 可视化（如果需要）
        if visualize or save_visualization:
            output_path = None
            if save_visualization and output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, base_name + "_visualization.jpg")
            
            self.visualize_detection(image_path, pose_info, output_path)
        
        return pose_info


# 使用示例
if __name__ == "__main__":
    # 创建ArUco位姿估计器
    estimator = ArUcoPoseEstimator()
    
    # 示例1: 检测指定ID的ArUco标记
    image_path = "./code_imgs_Aruco/0814/2-5/rgb/rgb_0000.png"
    target_id = 24  # 目标ArUco ID
    
    # 估计位姿
    pose_info = estimator.estimate_pose(
        image_path=image_path,
        target_id=target_id,
        dict_name='5x5_1000',
        marker_size=0.05,
        output_dir="./output/",
        base_name="aruco_detection",
        visualize=True,
        save_visualization=True
    )
    
    if pose_info is not None:
        print(f"\n=== 检测成功 ===")
        print(f"ArUco ID: {pose_info['id']}")
        print(f"3D位置: X={pose_info['tvec'][0]}m, Y={pose_info['tvec'][1]}m, Z={pose_info['tvec'][2]}m")
        print(f"检测距离: {pose_info['distance']}m")
        print(f"欧拉角: roll={pose_info['euler_angles'][0]}°, pitch={pose_info['euler_angles'][1]}°, yaw={pose_info['euler_angles'][2]}°")
    else:
        print("检测失败")
    
    # 示例2: 批量检测多个ArUco标记
    print("\n" + "="*50)
    print("示例2: 批量检测多个ArUco标记")
    print("="*50)
    
    # 假设要检测多个ID
    target_ids = [24]  # 要检测的ArUco ID列表
    
    for target_id in target_ids:
        print(f"\n正在检测ArUco ID: {target_id}")
        pose_info = estimator.estimate_pose(
            image_path=image_path,
            target_id=target_id,
            dict_name='5x5_1000',
            marker_size=0.05,
            output_dir="./output/",
            base_name=f"aruco_id_{target_id}",
            visualize=False,  # 批量检测时不显示图像
            save_visualization=True
        )
        
        if pose_info is not None:
            print(f"✓ ArUco ID {target_id} 检测成功")
        else:
            print(f"✗ ArUco ID {target_id} 检测失败")
    
    print("\n=== 检测完成 ===")