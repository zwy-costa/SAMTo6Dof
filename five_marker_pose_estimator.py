#!/usr/bin/env python3
"""
基于五个ArUco标记的位姿估计类
使用中心标记及其四个邻居标记的所有角点，估计中心标记相对于其中心的位姿
"""

import json
import numpy as np
import cv2
import os
from typing import Optional, Tuple, Dict, Any, List


class FiveMarkerPoseEstimator:
    def __init__(self, intrinsic_path="./head_intrinsic_params.json"):
        """初始化五标记位姿估计器"""
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
    
    def _get_marker_grid_coords(self, marker_id: int) -> Tuple[int, int]:
        """获取标记在网格中的坐标位置（仅支持 {4,5,8,11,12}，布局与原映射一致）"""
        mapping = {
            4: (1, 1),
            5: (1, 3),
            8: (2, 2),  # 中心
            11: (3, 1),
            12: (3, 3),
        }
        if marker_id in mapping:
            return mapping[marker_id]
        raise KeyError(f"不支持的标记ID: {marker_id}，仅支持 {list(mapping.keys())}")
    
    def _build_board_marker_corners_3d(self, marker_id: int, square_length_mm: float, marker_length_mm: float) -> np.ndarray:
        """以ID=8为坐标系原点(0,0)，构建给定ID标记的3D角点（单位mm）"""
        # 获取给定ID与中心ID=8的网格坐标
        row, col = self._get_marker_grid_coords(marker_id)
        row_c, col_c = self._get_marker_grid_coords(8)
        # 将网格坐标转换为以中心为原点的相对偏移（单位：一个square）
        d_row = float(row - row_c)
        d_col = float(col - col_c)
        # 该标记中心在本地坐标中的位置（mm）。注意列对应X，行对应Y。
        cx = d_col * float(square_length_mm)
        cy = d_row * float(square_length_mm)
        h = float(marker_length_mm) / 2.0
        # 角点顺序与cv2.aruco一致：TL, TR, BR, BL
        corners = np.array([
            [cx - h, cy - h, 0.0],  # TL
            [cx + h, cy - h, 0.0],  # TR
            [cx + h, cy + h, 0.0],  # BR
            [cx - h, cy + h, 0.0],  # BL
        ], dtype=np.float32)
        return corners
    
    def _neighbor_ids_for_center(self, center_id: int, ids: np.ndarray) -> List[int]:
        """根据中心id与检测到的ids，选出其周围的4个邻居(左上、右上、左下、右下)与自身，共5个id"""
        row_c, col_c = self._get_marker_grid_coords(center_id)
        candidates = [(row_c - 1, col_c - 1), (row_c - 1, col_c + 1), 
                     (row_c + 1, col_c - 1), (row_c + 1, col_c + 1)]
        
        # 构建已检测id的(row,col)索引
        rc_of_id: Dict[int, Tuple[int, int]] = {}
        if ids is not None and len(ids) > 0:
            for mid in ids.flatten().tolist():
                rc_of_id[int(mid)] = self._get_marker_grid_coords(int(mid))
        
        sel = [center_id]
        for rc in candidates:
            for mid, rc_val in rc_of_id.items():
                if rc_val == rc:
                    sel.append(mid)
                    break
        return sel
    
    def _compute_board_translated_pose_for_marker(self, marker_id: int,
                                                 board_rvec: np.ndarray,
                                                 board_tvec: np.ndarray,
                                                 square_length_mm: float) -> Tuple[np.ndarray, np.ndarray]:
        """计算标记相对于标定板中心的位姿"""
        row, col = self._get_marker_grid_coords(marker_id)
        center_3d = np.array([(col + 0.5) * square_length_mm, (row + 0.5) * square_length_mm, 0.0], dtype=np.float32)
        R_board, _ = cv2.Rodrigues(board_rvec)
        board_tvec_reshaped = board_tvec.reshape(3, 1)
        offset = (R_board @ center_3d.reshape(3, 1))
        new_tvec = board_tvec_reshaped + offset
        new_rvec = board_rvec.copy()
        return new_rvec.reshape(3, 1), new_tvec.reshape(3, 1)
    
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
    
    def estimate_marker_pose_from_five_markers(self, center_id: int,
                                             image: np.ndarray,
                                             corners: List[np.ndarray],
                                             ids: np.ndarray,
                                             square_length_mm: float,
                                             marker_length_mm: float) -> Tuple[bool, np.ndarray, np.ndarray, List[int]]:
        """
        使用ID集合{4,5,8,11,12}内已检测到的所有角点，
        在以ID=8为原点的本地坐标系下直接构建3D角点并求解PnP，返回(center_rvec, center_tvec)。
        注意：这里的center_id参数将被视为输出信息用途，坐标系固定以8为中心。
        """
        if ids is None or len(ids) == 0:
            return False, None, None, []
        id_list = ids.flatten().tolist()
        # 仅使用已知的五个ID，且实际被检测到的那些
        known_ids = [4, 5, 8, 11, 12]
        selected_ids = [mid for mid in known_ids if mid in id_list]
        if len(selected_ids) == 0:
            return False, None, None, []
        object_points: List[np.ndarray] = []
        image_points: List[np.ndarray] = []
        for mid in selected_ids:
            idx = id_list.index(mid)
            c2d = corners[idx].reshape(4, 2).astype(np.float32)
            c3d = self._build_board_marker_corners_3d(mid, square_length_mm, marker_length_mm)
            object_points.append(c3d)
            image_points.append(c2d)
        obj = np.concatenate(object_points, axis=0)
        img = np.concatenate(image_points, axis=0)
        if obj.shape[0] < 4:
            return False, None, None, selected_ids
        
        # ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        #     obj, img, self.K, self.dist_coeffs,
        #     flags=cv2.SOLVEPNP_ITERATIVE,
        #     iterationsCount=200,
        #     reprojectionError=2.5,
        #     confidence=0.99
        # )

        

        if not ok:
            return False, None, None, selected_ids
        return True, rvec, tvec, selected_ids
    
    def detect_markers(self, image: np.ndarray, dict_name: str = 'DICT_4X4_50') -> Tuple[List[np.ndarray], np.ndarray]:
        """检测ArUco标记"""
        try:
            d = getattr(cv2.aruco, dict_name)
            aruco_dict = cv2.aruco.getPredefinedDictionary(d)
            detector = cv2.aruco.ArucoDetector(aruco_dict)
            corners, ids, _ = detector.detectMarkers(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            return corners, ids
        except AttributeError:
            print("错误: 无法检测ArUco标记，请检查OpenCV版本")
            return [], None
    
    def estimate_pose(self, image_path: str, center_id: int, 
                     square_length_mm: float = 30.0, marker_length_mm: float = 22.0,
                     dict_name: str = 'DICT_4X4_50', output_dir: str = "./output/",
                     base_name: str = "five_marker_result", visualize: bool = True) -> Optional[Dict[str, Any]]:
        """
        估计指定中心标记的位姿
        
        Args:
            image_path: 图片路径
            center_id: 中心标记ID
            square_length_mm: 方格边长（毫米）
            marker_length_mm: 标记边长（毫米）
            dict_name: ArUco字典名称
            output_dir: 输出目录
            base_name: 输出文件基础名称
            visualize: 是否可视化结果
            
        Returns:
            pose_info: 位姿信息字典，如果检测失败返回None
        """
        print("=== 五标记ArUco位姿估计工具 ===")
        print(f"图片: {image_path}")
        print(f"中心标记ID: {center_id}")
        print(f"方格边长: {square_length_mm} mm")
        print(f"标记边长: {marker_length_mm} mm")
        print(f"字典类型: {dict_name}")
        print("=" * 40)
        
        # 检查输入文件
        if not os.path.exists(image_path):
            print(f"错误: 图片文件不存在: {image_path}")
            return None
        
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图片: {image_path}")
            return None
        
        print(f"图片尺寸: {image.shape}")
        
        # 检测ArUco标记
        corners, ids = self.detect_markers(image, dict_name)
        if ids is None or len(ids) == 0:
            print("未检测到任何ArUco标记")
            return None
        
        print(f"检测到 {len(ids)} 个ArUco标记")
        print(f"检测到的ID: {ids.flatten().tolist()}")
        
        # 检查中心标记是否存在
        if center_id not in ids.flatten():
            print(f"错误: 未找到中心标记ID {center_id}")
            return None
        
        # 使用五标记方法（以中心为原点的本地坐标系）估计位姿
        success, rvec, tvec, used_ids = self.estimate_marker_pose_from_five_markers(
            center_id, image, corners, ids, square_length_mm, marker_length_mm
        )
        
        if not success:
            print("五标记位姿估计失败")
            return None
        
        print(f"五标记位姿估计成功")
        print(f"使用的标记ID: {used_ids}")
        
        # 将旋转向量转换为旋转矩阵
        R, _ = cv2.Rodrigues(rvec)
        
        # 计算欧拉角
        euler_angles = self._rotation_matrix_to_euler_angles(R)
        
        # 计算距离
        distance = np.linalg.norm(tvec)
        
        # 获取中心标记的角点
        id_list = ids.flatten().tolist()
        center_idx = id_list.index(center_id)
        center_corners = corners[center_idx].reshape(4, 2)
        center_2d = np.mean(center_corners, axis=0)
        
        # 计算标记的面积
        area = cv2.contourArea(center_corners.astype(np.int32))
        
        pose_info = {
            'center_id': center_id,
            'corners': center_corners,
            'center_2d': center_2d,
            'area': area,
            'rvec': rvec,
            'tvec': tvec,
            'R': R,
            'euler_angles': euler_angles,
            'distance': distance,
            'square_length_mm': square_length_mm,
            'marker_length_mm': marker_length_mm,
            'dict_name': dict_name,
            'used_ids': used_ids,
            'total_detected': len(ids),
            'detection_time': 'N/A',
            'confidence': 'N/A'
        }
        
        # 打印位姿信息
        self.print_pose_info(pose_info)
        
        # 可视化（如果需要）
        if visualize:
            output_path = None
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, base_name + "_visualization.jpg")
            
            self.visualize_detection(image_path, pose_info, output_path)
        
        return pose_info
    
    def print_pose_info(self, pose_info):
        """打印位姿信息"""
        if pose_info is None:
            print("无位姿信息")
            return
        
        print(f"\n=== 中心标记 {pose_info['center_id']} 位姿信息 ===")
        print("=" * 50)
        
        # 基本信息
        print(f"中心标记ID: {pose_info['center_id']}")
        print(f"方格边长: {pose_info['square_length_mm']} mm")
        print(f"标记边长: {pose_info['marker_length_mm']} mm")
        print(f"使用的标记ID: {pose_info['used_ids']}")
        print(f"总检测标记数: {pose_info['total_detected']}")
        
        # 图像坐标
        center_2d = pose_info['center_2d']
        print(f"中心点: ({float(center_2d[0])}, {float(center_2d[1])})")
        
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
        print(f"距离: {float(distance)} mm")
        
        # 坐标系说明
        print(f"\n坐标系说明:")
        print(f"  - 原点: 中心标记的中心点")
        print(f"  - X轴: 红色，指向标记的右方向")
        print(f"  - Y轴: 绿色，指向标记的下方向") 
        print(f"  - Z轴: 蓝色，指向标记的前方向（垂直于标记平面）")
        print(f"  - 旋转向量: 相对于相机坐标系的旋转")
        print(f"  - 平移向量: 标记中心相对于相机的位置")
        
        # 检测质量信息
        print(f"\n检测质量信息:")
        print(f"  - 方格边长: {pose_info['square_length_mm']} mm")
        print(f"  - 标记边长: {pose_info['marker_length_mm']} mm")
        print(f"  - 图像面积: {float(pose_info['area'])} 像素²")
        print(f"  - 检测距离: {float(pose_info['distance'])} mm")
        
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
            print(f"    角点{i+1}: ({float(corner[0])}, {float(corner[1])})")
        
        # 总结信息
        print(f"\n=== 检测总结 ===")
        print(f"✓ 成功检测到中心标记 ID: {pose_info['center_id']}")
        print(f"✓ 标记中心位置: ({float(center_2d[0])}, {float(center_2d[1])}) 像素")
        print(f"✓ 3D位置: X={tvec[0]}mm, Y={tvec[1]}mm, Z={tvec[2]}mm")
        print(f"✓ 检测距离: {float(distance)} mm")
        print(f"✓ 使用标记数: {len(pose_info['used_ids'])}")
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
        
        # 定义坐标系的原点和三个轴的方向（以毫米为单位）
        axis_length = 100  # 坐标轴长度（毫米）
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
        cv2.putText(image, 'Five-Marker Coordinate System', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
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
            ids = np.array([[pose_info['center_id']]])
            
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
            center_2d = pose_info['center_2d']
            center = center_2d.astype(np.int32)
            cv2.putText(image, f"ID: {pose_info['center_id']}", 
                       (int(center[0]) - 20, int(center[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print("使用手动绘制标记")
        
        # 绘制中心点
        center_2d = pose_info['center_2d']
        center = center_2d.astype(np.int32)
        cv2.circle(image, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)
        
        # 绘制位姿信息
        tvec = pose_info['tvec']
        pose_text = f"Z: {float(tvec[2])}mm"
        cv2.putText(image, pose_text, 
                   (int(center[0]) - 20, int(center[1]) + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # 绘制更多信息
        info_text = f"ID: {pose_info['center_id']} | Distance: {float(pose_info['distance'])}mm"
        cv2.putText(image, info_text, 
                   (10, image.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 绘制位姿详细信息
        tvec = pose_info['tvec'].flatten()
        pose_detail = f"X:{tvec[0]}mm Y:{tvec[1]}mm Z:{tvec[2]}mm"
        cv2.putText(image, pose_detail, 
                   (10, image.shape[0] - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 绘制使用的标记信息
        used_info = f"Used IDs: {pose_info['used_ids']}"
        cv2.putText(image, used_info, 
                   (10, image.shape[0] - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 绘制坐标系（ArUco标记的坐标系）
        self._draw_coordinate_system(image, pose_info)
        
        # 保存或显示结果
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"可视化结果已保存到: {output_path}")
        else:
            # 显示图片
            cv2.imshow('Five-Marker ArUco Detection', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


# 使用示例
if __name__ == "__main__":
    # 创建五标记位姿估计器
    estimator = FiveMarkerPoseEstimator()
    
    # 示例: 检测指定ID的ArUco标记
    image_path = "./input/calibrate_img/6/rgb/rgb_0000.png"
    center_id = 8  # 本地坐标系以ID=8为中心
    
    # 估计位姿
    pose_info = estimator.estimate_pose(
        image_path=image_path,
        center_id=center_id,
        square_length_mm=30.0,
        marker_length_mm=22.0,
        dict_name='DICT_4X4_1000',
        output_dir="./output/",
        base_name="five_marker_detection",
        visualize=True
    )
    
    if pose_info is not None:
        print(f"\n=== 检测成功 ===")
        print(f"中心标记ID: {pose_info['center_id']}")
        print(f"3D位置: X={pose_info['tvec'][0]}mm, Y={pose_info['tvec'][1]}mm, Z={pose_info['tvec'][2]}mm")
        print(f"检测距离: {pose_info['distance']}mm")
        print(f"欧拉角: roll={pose_info['euler_angles'][0]}°, pitch={pose_info['euler_angles'][1]}°, yaw={pose_info['euler_angles'][2]}°")
        print(f"使用的标记ID: {pose_info['used_ids']}")
    else:
        print("检测失败")
    
    print("\n=== 检测完成 ===") 