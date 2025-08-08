#!/usr/bin/env python3
"""
立方体姿态估计类
计算立方体上立面与相机坐标系之间的变换关系
"""

import numpy as np
import cv2
import json
import ast
import os
from fastsam import FastSAM, FastSAMPrompt
from scipy.spatial import ConvexHull
from itertools import combinations


class CubePoseEstimator:
    def __init__(self, intrinsic_path="./head_intrinsic_params.json"):
        """初始化相机内参"""
        self.K, self.dist_coeffs = self._load_camera_intrinsics(intrinsic_path)
        
    def _load_camera_intrinsics(self, intrinsic_path):
        """加载相机内参"""
        with open(intrinsic_path, 'r') as f:
            intrinsic_data = json.load(f)
        
        intrinsic = intrinsic_data['intrinsic']
        
        # 构建相机内参矩阵K
        K = np.array([
            [intrinsic['fx'], 0, intrinsic['ppx']],
            [0, intrinsic['fy'], intrinsic['ppy']],
            [0, 0, 1]
        ])
        
        # 畸变系数
        dist_coeffs = np.array([
            intrinsic['k1'], intrinsic['k2'], intrinsic['k3'],
            intrinsic['p1'], intrinsic['p2']
        ])
        
        return K, dist_coeffs
    
    def simple_segmentation_visualization(self, rgb_img, annotations, output_dir, base_name):
        """简单的分割结果可视化，显示所有分割对象并标注ID"""
        # 创建可视化图像
        vis_img = rgb_img.copy()
        
        # 为每个分割对象添加颜色和ID标注
        for i, annotation in enumerate(annotations):
            # 检查annotation的格式
            if isinstance(annotation, dict):
                mask = annotation['segmentation']
            else:
                # 如果annotation不是字典，直接使用它作为mask
                mask = annotation
            
            mask_bool = self._safe_mask_to_bool(mask)
            
            # 生成随机颜色
            color = np.random.randint(0, 255, 3).tolist()
            
            # 在掩码区域应用颜色
            vis_img[mask_bool] = vis_img[mask_bool] * 0.7 + np.array(color) * 0.3
            
            # 计算质心用于标注ID
            y_coords, x_coords = np.where(mask_bool)
            if len(y_coords) > 0:
                centroid_y, centroid_x = int(np.mean(y_coords)), int(np.mean(x_coords))
                
                # 标注ID
                cv2.putText(vis_img, str(i+1), (centroid_x-10, centroid_y+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.circle(vis_img, (centroid_x, centroid_y), 5, (255, 255, 255), -1)
        
        # 保存结果
        output_path = os.path.join(output_dir, base_name + "_all_segmentations.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        
        return output_path
    
    def visualize_box_prompt_result(self, rgb_img, box_prompt, selected_corners, output_path):
        """可视化box_prompt和选中的角点"""
        vis_img = rgb_img.copy()
        
        try:
            # 解析box_prompt
            if isinstance(box_prompt, str):
                box_prompt = ast.literal_eval(box_prompt)
            
            # 绘制box_prompt区域（仅当box_prompt不为None时）
            if box_prompt is not None and isinstance(box_prompt, list) and len(box_prompt) == 4:  # [x, y, x2, y2] 格式
                x, y, x2, y2 = box_prompt
                
                # 绘制box_prompt区域（半透明蓝色）
                overlay = vis_img.copy()
                cv2.rectangle(overlay, (int(x), int(y)), (int(x2), int(y2)), (255, 0, 0), -1)
                cv2.addWeighted(overlay, 0.3, vis_img, 0.7, 0, vis_img)
                
                # 绘制box_prompt边界（蓝色）
                cv2.rectangle(vis_img, (int(x), int(y)), (int(x2), int(y2)), (255, 0, 0), 2)
                
                # 绘制左上角点和右下角点
                cv2.circle(vis_img, (int(x), int(y)), 8, (0, 255, 255), -1)  # 左上角点（黄色）
                cv2.circle(vis_img, (int(x), int(y)), 8, (255, 255, 255), 2)  # 白色边框
                cv2.putText(vis_img, "TL", (int(x)+10, int(y)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                cv2.circle(vis_img, (int(x2), int(y2)), 8, (0, 255, 255), -1)  # 右下角点（黄色）
                cv2.circle(vis_img, (int(x2), int(y2)), 8, (255, 255, 255), 2)  # 白色边框
                cv2.putText(vis_img, "BR", (int(x2)+10, int(y2)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # 标注box
                cv2.putText(vis_img, "Box Prompt", (int(x), int(y)-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # 绘制选中的角点
            if selected_corners is not None:
                for i, (x, y) in enumerate(selected_corners):
                    # 绘制角点（绿色圆圈）
                    cv2.circle(vis_img, (int(x), int(y)), 12, (0, 255, 0), -1)
                    cv2.circle(vis_img, (int(x), int(y)), 12, (255, 255, 255), 2)
                    
                    # 标注角点编号
                    cv2.putText(vis_img, str(i+1), (int(x)+15, int(y)-15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # 绘制角点之间的连线（绿色）
                for i in range(4):
                    pt1 = selected_corners[i]
                    pt2 = selected_corners[(i+1) % 4]
                    cv2.line(vis_img, (int(pt1[0]), int(pt1[1])), 
                            (int(pt2[0]), int(pt2[1])), (0, 255, 0), 3)
                
                # 根据是否有box_prompt添加不同的标题
                if box_prompt is not None:
                    cv2.putText(vis_img, "Selected Corners in Box Prompt", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                else:
                    cv2.putText(vis_img, "Selected Corners", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                # 根据是否有box_prompt添加不同的标题
                if box_prompt is not None:
                    cv2.putText(vis_img, "No Corners Found in Box Prompt", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                else:
                    cv2.putText(vis_img, "No Corners Found", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        except Exception as e:
            print(f"可视化box_prompt时出错: {e}")
        
        # 保存结果
        cv2.imwrite(output_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        
        return vis_img
    
    def _get_corner_depths(self, corners_2d, depth_img):
        """根据2D角点和深度图获取3D相机坐标，以中心点深度为参考进行估计"""
        if len(corners_2d) != 4:
            print(f"错误: 需要4个角点") 
            return None
        
        corners_3d_camera = []
        depth_validity = []  # 记录每个角点的深度是否有效
        
        def is_valid_depth(depth):
            """检查深度值是否有效"""
            return depth > 0 and not np.isnan(depth) and not np.isinf(depth)
        
        def get_center_depth(corners_2d, depth_img, search_radius=10):
            """获取中心点周围的深度值作为参考"""
            # 计算四个角点的中心
            center_x = np.mean([corner[0] for corner in corners_2d])
            center_y = np.mean([corner[1] for corner in corners_2d])
            center_x, center_y = int(center_x), int(center_y)
            
            print(f"  计算中心点: ({center_x}, {center_y})")
            
            # 收集中心点周围的深度值
            center_depths = []
            for dy in range(-search_radius, search_radius + 1):
                for dx in range(-search_radius, search_radius + 1):
                    nx, ny = center_x + dx, center_y + dy
                    if (0 <= nx < depth_img.shape[1] and 0 <= ny < depth_img.shape[0]):
                        neighbor_depth = depth_img[ny, nx]
                        if is_valid_depth(neighbor_depth):
                            center_depths.append(neighbor_depth)
            
            if center_depths:
                center_depth_mean = np.mean(center_depths)
                center_depth_std = np.std(center_depths)
                print(f"  中心点深度: 均值={center_depth_mean:.3f}, 标准差={center_depth_std:.3f} (基于{len(center_depths)}个像素)")
                return center_depth_mean, center_depth_std, center_depths
            else:
                print(f"  警告: 中心点周围无法获取有效深度")
                return None, None, []
        
        def estimate_depth_from_neighbors(x, y, depth_img, center_depth_mean, center_depth_std, tolerance, search_radius=5):
            """从周围像素估计深度值，以中心点深度为参考，先筛选异常值"""
            valid_depths = []
            filtered_depths = []
            
            # 第一步：收集周围像素的深度值
            for dy in range(-search_radius, search_radius + 1):
                for dx in range(-search_radius, search_radius + 1):
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < depth_img.shape[1] and 0 <= ny < depth_img.shape[0]):
                        neighbor_depth = depth_img[ny, nx]
                        if is_valid_depth(neighbor_depth):
                            valid_depths.append(neighbor_depth)
            
            if valid_depths:
                # 第二步：根据中心点深度筛选异常值
                if center_depth_mean is not None and center_depth_std is not None:
                    # 使用传入的tolerance参数
                    for depth in valid_depths:
                        if abs(depth - center_depth_mean) <= tolerance:
                            filtered_depths.append(depth)
                    
                    print(f"    收集到 {len(valid_depths)} 个有效深度值")
                    print(f"    根据中心点深度筛选后剩余 {len(filtered_depths)} 个深度值")
                    
                    if filtered_depths:
                        # 使用筛选后的深度值计算估计值
                        estimated_depth = np.median(filtered_depths)
                        print(f"    筛选后深度范围: {min(filtered_depths):.3f} ~ {max(filtered_depths):.3f}")
                        print(f"    筛选后估计深度: {estimated_depth:.3f}")
                    else:
                        # 如果筛选后没有深度值，使用中心点深度
                        print(f"   收集到的深度范围: {min(valid_depths):.3f} ~ {max(valid_depths):.3f}")            
                        print(f" ×××××筛选后无有效深度值，使用中心点深度: {center_depth_mean:.3f}")
                        estimated_depth = center_depth_mean
                else:
                    # 如果没有中心点参考，使用原始方法
                    estimated_depth = np.median(valid_depths)
                    print(f"    无中心点参考，使用原始估计深度: {estimated_depth:.3f}")
                
                return estimated_depth, len(filtered_depths)
            else:
                # 如果周围没有有效深度，使用中心点深度
                if center_depth_mean is not None:
                    print(f"    周围无有效深度，使用中心点深度: {center_depth_mean:.3f}")
                    return center_depth_mean, 0
                else:
                    return None, 0
        
        print(f"获取角点深度信息...")
        
        # 首先获取中心点深度作为参考
        center_depth_mean, center_depth_std, center_depths = get_center_depth(corners_2d, depth_img)
        if center_depth_std is None:
            print(f"错误: 无法获取中心点深度")
            return None, None, None, None
            
        if center_depth_std >= 0.002:
            tolerance = 15 * center_depth_std
        else:
            tolerance = 30 * center_depth_std
        tolerance = min(tolerance, 0.05)
        print(f"  角点深度与中心点深度的允许差异阈值: {tolerance:.3f} *****")


        
        for i, (x, y) in enumerate(corners_2d):
            # 确保坐标在图像范围内
            x, y = int(x), int(y)
            if 0 <= x < depth_img.shape[1] and 0 <= y < depth_img.shape[0]:
                # 获取直接深度值
                depth = depth_img[y, x]
                
                if is_valid_depth(depth):
                    # 深度有效，检查是否与中心点深度相近
                    if center_depth_mean is not None:
                        # 根据center_depth_std动态设置tolerance
                        tolerance_center = 12 * center_depth_std  # 使用更严格的标准 (其他地方是15)
                        if abs(depth - center_depth_mean) > tolerance_center:
                            print(f"  角点 {i+1}: 直接深度 {depth:.3f} 与中心点深度差异较大 (差异: {abs(depth - center_depth_mean):.3f})，尝试重新估计...")
                            estimated_depth, neighbor_count = estimate_depth_from_neighbors(x, y, depth_img, center_depth_mean, center_depth_std, tolerance)
                            if estimated_depth is not None:
                                depth = estimated_depth
                                depth_validity.append(False)  # 标记为估计值
                                print(f"  角点 {i+1}: 重新估计深度 {depth:.3f}")
                            else:
                                depth_validity.append(True)  # 保持原始值
                                print(f"  角点 {i+1}: 保持原始深度 {depth:.3f}")
                        else:
                            depth_validity.append(True)
                            print(f"  角点 {i+1}: 直接获取深度 {depth:.3f} (与中心点深度相近，差异: {abs(depth - center_depth_mean):.3f})")
                    else:
                        depth_validity.append(True)
                        print(f"  角点 {i+1}: 直接获取深度 {depth:.3f}")
                else:
                    # 深度无效，从周围像素估计
                    print(f"  角点 {i+1}: 直接深度无效 ({depth:.3f})，尝试从周围像素估计...")
                    estimated_depth, neighbor_count = estimate_depth_from_neighbors(x, y, depth_img, center_depth_mean, center_depth_std, tolerance)
                    
                    if estimated_depth is not None:
                        depth = estimated_depth
                        depth_validity.append(False)  # 标记为估计值
                        print(f"  角点 {i+1}: 从 {neighbor_count} 个邻居像素估计深度 {depth:.3f}")
                    else:
                        print(f"  角点 {i+1}: 无法估计深度，跳过")
                        return None, None, None, None
                
                # 将像素坐标转换为归一化坐标
                pixel_coord = np.array([[x, y]], dtype=np.float32)
                normalized_coord = cv2.undistortPoints(pixel_coord, self.K, self.dist_coeffs)[0][0]
                
                # 根据深度和归一化坐标计算3D相机坐标
                # 归一化坐标是Z=1平面上的点，所以3D坐标为 (X*Z, Y*Z, Z)
                X_camera = normalized_coord[0] * depth
                Y_camera = normalized_coord[1] * depth
                Z_camera = depth
                
                corners_3d_camera.append([X_camera, Y_camera, Z_camera])
                status = "直接" if depth_validity[-1] else "估计"
                print(f"  角点 {i+1}: 像素({x},{y}) -> {status}深度{depth:.3f} -> 相机坐标({X_camera:.3f}, {Y_camera:.3f}, {Z_camera:.3f})")
            else:
                print(f"  角点 {i+1}: 坐标({x},{y})超出图像范围")
                return None
        
        # 验证所有角点深度的一致性
        if center_depth_mean is not None:
            print(f"\n深度一致性验证:")
            all_depths = [corners_3d_camera[i][2] for i in range(4)]
            depth_std = np.std(all_depths)
            depth_mean = np.mean(all_depths)
            
            print(f"  所有角点深度: {[f'{d:.3f}' for d in all_depths]}")
            print(f"  深度均值: {depth_mean:.3f}, 标准差: {depth_std:.3f}")
            print(f"  中心点参考深度: {center_depth_mean:.3f}")
            print(f"  中心点深度标准差: {center_depth_std:.3f}")
            
            # 检查每个角点深度与中心点深度的差异
            # 根据center_depth_std动态设置tolerance
            # if center_depth_std >= 0.02:
            #     tolerance = 15 * center_depth_std
            # else:
            #     tolerance = 30 * center_depth_std
            consistent_count = 0
            for i, depth in enumerate(all_depths):
                diff = abs(depth - center_depth_mean)
                if diff > tolerance:
                    print(f"  警告: 角点 {i+1} 深度 {depth:.3f} 与中心点深度差异较大 ({diff:.3f} > {tolerance:.3f})")
                else:
                    print(f"  角点 {i+1} 深度 {depth:.3f} 与中心点深度一致 (差异: {diff:.3f} <= {tolerance:.3f})")
                    consistent_count += 1
            
            print(f"  深度一致性统计: {consistent_count}/4 个角点与中心点深度一致")
            
            # 检查角点间深度的一致性
            max_depth_diff = max(all_depths) - min(all_depths)
            print(f"  角点间最大深度差异: {max_depth_diff:.3f}")
            if max_depth_diff > 20 * center_depth_std:
                print(f"  警告: 角点间深度差异较大，超过20倍中心点深度标准差({20 * center_depth_std:.3f})，可能影响3D变换精度")
            else:
                print(f"  角点间深度差异在合理范围内，20倍中心点深度标准差为{20 * center_depth_std:.3f}")
        
        return np.array(corners_3d_camera, dtype=np.float32), depth_validity, center_depth_mean, center_depth_std
    
    def _solve_rigid_transform_svd(self, points_camera, points_object):
        """使用SVD方法求解刚体变换"""
        if len(points_camera) != len(points_object) or len(points_camera) < 3:
            return None, None
        
        # 计算质心
        centroid_camera = np.mean(points_camera, axis=0)
        centroid_object = np.mean(points_object, axis=0)
        
        # 去质心
        points_camera_centered = points_camera - centroid_camera
        points_object_centered = points_object - centroid_object
        
        # 计算协方差矩阵
        H = points_object_centered.T @ points_camera_centered
        
        # SVD分解
        U, S, Vt = np.linalg.svd(H)
        
        # 计算旋转矩阵
        R = Vt.T @ U.T
        
        # 确保是右手坐标系
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # 计算平移向量
        t = centroid_camera - R @ centroid_object
        
        return R, t
    
    def _rotation_matrix_to_euler_angles(self, R):
        """将旋转矩阵转换为欧拉角（ZYX顺序）"""
        if R.shape != (3, 3):
            raise ValueError("旋转矩阵必须是3x3矩阵")
        
        # 提取欧拉角（ZYX顺序）
        sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
        # sy = np.sqrt(R[2,1]**2 + R[2,2]**2)
        
        if sy > 1e-6:  # 非奇异情况
            roll = np.arctan2(R[2,1], R[2,2])      # 绕X轴旋转（Roll）
            pitch = np.arctan2(-R[2,0], sy)        # 绕Y轴旋转（Pitch）
            yaw = np.arctan2(R[1,0], R[0,0])       # 绕Z轴旋转（Yaw）
        else:  # 奇异情况（万向锁）
            roll = np.arctan2(-R[1,2], R[1,1])
            pitch = np.arctan2(-R[2,0], sy)
            yaw = 0
        
        # 转换为度数
        roll_deg = np.degrees(roll)
        pitch_deg = np.degrees(pitch)
        yaw_deg = np.degrees(yaw)
        
        return roll_deg, pitch_deg, yaw_deg
    
    def _safe_mask_to_numpy(self, mask):
        """安全地将mask转换为numpy数组"""
        if isinstance(mask, np.ndarray):
            return mask
        elif hasattr(mask, 'cpu'):
            return mask.cpu().numpy()
        elif hasattr(mask, 'numpy'):
            return mask.numpy()
        else:
            return np.array(mask)
    
    def _safe_mask_to_bool(self, mask):
        """安全地将mask转换为布尔数组"""
        mask_np = self._safe_mask_to_numpy(mask)
        # 确保是numpy数组而不是tensor
        if hasattr(mask_np, 'cpu'):
            mask_np = mask_np.cpu().numpy()
        return mask_np.astype(bool)
    
    def _sort_corners_clockwise_from_top_left(self, corners):
        """
        将角点按左上角为1号角点，顺时针顺序排列
        角点顺序：左上(1) -> 右上(2) -> 右下(3) -> 左下(4)
        """
        if len(corners) != 4:
            return corners
        
        # 计算中心点
        center = np.mean(corners, axis=0)
        
        # 计算每个角点相对于中心的角度
        angles = []
        for corner in corners:
            dx = corner[0] - center[0]
            dy = corner[1] - center[1]
            angle = np.arctan2(dy, dx)
            angles.append(angle)
        
        # 按角度排序，确保顺时针顺序
        sorted_indices = np.argsort(angles)
        corners_sorted = corners[sorted_indices]
        
        # 找到左上角点：选择距离原点(0,0)最近的角点
        top_left_idx = 0
        min_distance = float('inf')
        
        for i, corner in enumerate(corners_sorted):
            distance = np.sqrt(corner[0]**2 + corner[1]**2)  # 距离原点(0,0)的距离
            if distance < min_distance:
                min_distance = distance
                top_left_idx = i
        
        # 重新排列角点，使左上角为第一个
        corners_reordered = np.roll(corners_sorted, -top_left_idx, axis=0)
        
        # 检查边长关系，确保12是短边，14是长边
        # 计算边长：12(左上到右上), 14(左上到左下), 23(右上到右下), 34(右下到左下)
        edge_12 = np.sqrt((corners_reordered[1][0] - corners_reordered[0][0])**2 + 
                         (corners_reordered[1][1] - corners_reordered[0][1])**2)  # 12边
        edge_14 = np.sqrt((corners_reordered[3][0] - corners_reordered[0][0])**2 + 
                         (corners_reordered[3][1] - corners_reordered[0][1])**2)  # 14边
        edge_23 = np.sqrt((corners_reordered[2][0] - corners_reordered[1][0])**2 + 
                         (corners_reordered[2][1] - corners_reordered[1][1])**2)  # 23边
        edge_34 = np.sqrt((corners_reordered[3][0] - corners_reordered[2][0])**2 + 
                         (corners_reordered[3][1] - corners_reordered[2][1])**2)  # 34边
        
        # 判断是否需要调整顺序
        # 期望：12和34是短边，14和23是长边
        short_edges = [edge_12, edge_34]
        long_edges = [edge_14, edge_23]
        
        avg_short = np.mean(short_edges)
        avg_long = np.mean(long_edges)
        
        # 如果12边比14边长，说明需要调整顺序
        if edge_12 > edge_14:
            print(f"边长检查：12边({edge_12:.3f}) > 14边({edge_14:.3f})，需要调整顺序")
            # 将1,2,3,4顺序改成4,1,2,3
            corners_reordered = np.roll(corners_reordered, -1, axis=0)
            print(f"调整后角点: {corners_reordered}")
        else:
            print(f"边长检查：12边({edge_12:.3f}) <= 14边({edge_14:.3f})，顺序正确")
        
        # 添加调试信息
        print(f"原始角点: {corners}")
        print(f"角度: {[f'{a:.3f}' for a in angles]}")
        print(f"排序后角点: {corners_sorted}")
        distances = [np.sqrt(corner[0]**2 + corner[1]**2) for corner in corners_sorted]
        print(f"距离原点: {[f'{d:.3f}' for d in distances]}")
        print(f"左上角索引: {top_left_idx}, 左上角: {corners_sorted[top_left_idx]}, 距离: {min_distance:.3f}")
        print(f"边长: 12={edge_12:.3f}, 14={edge_14:.3f}, 23={edge_23:.3f}, 34={edge_34:.3f}")
        print(f"最终角点: {corners_reordered}")
        
        return corners_reordered
    
    def _calculate_angles_and_edges(self, points):
        """计算4个点的角度和边长"""
        if len(points) != 4:
            return None, None
        
        # 计算边长
        edges = []
        for i in range(4):
            pt1 = points[i]
            pt2 = points[(i+1) % 4]
            edge_length = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
            edges.append(edge_length)
        
        # 计算角度
        angles = []
        for i in range(4):
            pt1 = points[i]
            pt2 = points[(i+1) % 4]
            pt3 = points[(i+2) % 4]
            
            vec1 = pt2 - pt1
            vec2 = pt3 - pt2
            
            dot_product = np.dot(vec1, vec2)
            norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
            if norms > 0:
                cos_angle = dot_product / norms
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle) * 180 / np.pi
                angles.append(angle)
        
        return angles, edges
    
    def _optimize_parallelogram_fit(self, points, mask_np):
        """优化平行四边形拟合，使其更符合立方体上立面的特征"""
        if len(points) < 4:
            return None
        
        # 方法1: 使用凸包找到边界点，然后选择最佳的4个点
        try:
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            
            # 如果凸包点数等于4，直接使用
            if len(hull_points) == 4:
                return hull_points
            elif len(hull_points) > 4:
                # 计算掩码面积
                mask_area = np.sum(mask_np)
                
                # 遍历所有点，选择最接近平行四边形的4个点
                best_points = self._select_best_parallelogram_points(hull_points, mask_area)
                
                if best_points is not None:
                    # 计算拟合质量
                    bbox_array = np.array(best_points)
                    bbox_area = cv2.contourArea(bbox_array)
                    fit_quality = mask_area / bbox_area if bbox_area > 0 else 0
                    if abs(fit_quality - 1) < 0.5:
                        return best_points
                    else:
                        return None
                else:
                    return None
        except:
            pass
        
        return None
    
    def _select_best_parallelogram_points(self, points, bbox_area=None):
        """从多个点中选择最接近平行四边形的4个点"""
        if len(points) <= 4:
            return points[:4]
        
        # 计算所有可能的4点组合
        best_points = None
        best_score = float('inf')
        
        for combo in combinations(points, 4):
            combo = np.array(combo)
            
            # 计算平行四边形特征评分
            score = self._calculate_parallelogram_score(combo, bbox_area)
            if score < best_score:
                best_score = score
                best_points = combo
        
        return best_points if best_points is not None else points[:4]
    
    def _calculate_parallelogram_score(self, points, bbox_area=None):
        """计算4个点构成平行四边形的质量评分（分数越低越好）"""
        if len(points) != 4:
            return float('inf')
        
        # 使用公共函数计算角度和边长
        angles, edges = self._calculate_angles_and_edges(points)
        if angles is None or edges is None:
            return float('inf')
        
        # 计算评分指标
        
        # 1. 对角度数一致性（平行四边形对角度数相等）
        angle_pairs = [(angles[0], angles[2]), (angles[1], angles[3])]
        angle_consistency_score = sum(abs(pair[0] - pair[1]) for pair in angle_pairs)
        
        # 2. 对边长度一致性（平行四边形对边长度相等）
        edge_pairs = [(edges[0], edges[2]), (edges[1], edges[3])]
        edge_consistency_score = 0
        for pair in edge_pairs:
            if max(pair[0], pair[1]) > 0:
                edge_consistency_score += abs(pair[0] - pair[1]) / max(pair[0], pair[1])
        
        # 3. 角度合理性（应该在60-120度范围内，适应透视）
        angle_reasonableness = 0
        for angle in angles:
            if angle < 60 or angle > 120:
                angle_reasonableness += min(abs(angle - 60), abs(angle - 120))
        
        # 4. 面积合理性（避免过于细长的形状）
        area = cv2.contourArea(points.astype(np.int32))
        perimeter = sum(edges)
        if perimeter > 0:
            compactness = 4 * np.pi * area / (perimeter * perimeter)
            compactness_score = abs(compactness - 0.785)  # 理想正方形的紧凑度
        else:
            compactness_score = 1.0
        
        # 5. 重叠性指标（与原始掩码的重叠程度）
        overlap_score = 0
        if bbox_area is not None and bbox_area > 0:
            # 计算重叠比例，理想情况下应该接近1
            overlap_ratio = area / bbox_area
            # 如果重叠比例偏离1太多，增加惩罚
            overlap_score = abs(overlap_ratio - 1.0)
        else:
            overlap_score = 0.5  # 默认惩罚值
        
        # 综合评分（权重可调整）
        total_score = (
            angle_consistency_score * 2.0 +      # 对角度数一致性最重要
            edge_consistency_score * 1.5 +       # 对边长度一致性次之
            angle_reasonableness * 0.5 +         # 角度合理性
            compactness_score * 1.0 +            # 紧凑度
            overlap_score * 1000.0                  # 重叠性指标
            # overlap_score * 100.0                  # 重叠性指标
        )
        
        return total_score
    
    def _filter_cube_segments(self, annotations, depth_img, min_area=4500, max_area=50000):
        """基于几何特征和深度信息筛选立方体上立面分割"""
        filtered_annotations = []
        
        for i, annotation in enumerate(annotations):
            # 检查annotation的格式
            if isinstance(annotation, dict):
                mask = annotation['segmentation']
            else:
                # 如果annotation不是字典，直接使用它作为mask
                mask = annotation
            
            # 确保mask是numpy数组
            mask_np = self._safe_mask_to_numpy(mask)
            
            # 计算面积
            area = np.sum(mask_np)
            if area < min_area or area > max_area:
                print(f"××× 分割 {i+1} 面积 {area:.3f} 不在 {min_area:.3f} 和 {max_area:.3f} 之间")
                continue
            
            # 计算边界框
            y_coords, x_coords = np.where(mask_np)
            if len(y_coords) == 0:
                continue
            
            # 使用优化的平行四边形拟合
            points = np.column_stack((x_coords, y_coords))
            if len(points) < 4:
                continue
                
            try:
                # 使用优化的平行四边形拟合
                bbox_points = self._optimize_parallelogram_fit(points, mask_np)
                if bbox_points is None:
                    continue
                
                # 使用公共函数计算角度和边长
                angles_deg, edges = self._calculate_angles_and_edges(bbox_points)
                if angles_deg is None or edges is None:
                    print(f"××× 分割 {i+1} 无法找到平行四边形")
                    continue
                
                # 检查平行四边形角度特征
                angle_pairs = [(angles_deg[0], angles_deg[2]), (angles_deg[1], angles_deg[3])]
                angle_consistency = True
                for pair in angle_pairs:
                    if abs(pair[0] - pair[1]) > 15:
                        angle_consistency = False
                        break
                
                if not angle_consistency:
                    print(f"××× 分割 {i+1} 对角角度不一致，角度差值 {abs(pair[0] - pair[1])} 大于 15 度")
                    continue
                
                # 计算角度偏差
                angle_deviation = np.mean([abs(angle - 90) for angle in angles_deg])
                if angle_deviation > 45:
                    print(f"××× 分割 {i+1} 角度偏差 {angle_deviation} 大于 45 度")
                    continue
                
                # 计算深度统计信息
                mask_bool = self._safe_mask_to_bool(mask)
                depth_values = depth_img[mask_bool]
                if len(depth_values) == 0:
                    print(f"××× 分割 {i+1} 所有点 深度值为0")
                    continue
                    
                depth_mean = np.mean(depth_values)
                depth_std = np.std(depth_values)
                
                # 检查深度变化
                if depth_std > 0.15:
                    print(f"××× 分割 {i+1} 深度标准差 {depth_std} 大于 0.15")
                    continue
                
                # 检查四个角点深度信息与中心点深度信息的差异
                # 使用四个角点计算出的中心点
                center_x = np.mean([point[0] for point in bbox_points])
                center_y = np.mean([point[1] for point in bbox_points])
                center_x, center_y = int(center_x), int(center_y)
                
                # 获取中心点周围一定范围内的有效深度值均值
                search_radius = 5  # 搜索半径
                center_depths = []
                
                for dy in range(-search_radius, search_radius + 1):
                    for dx in range(-search_radius, search_radius + 1):
                        nx, ny = center_x + dx, center_y + dy
                        if (0 <= nx < depth_img.shape[1] and 0 <= ny < depth_img.shape[0]):
                            neighbor_depth = depth_img[ny, nx]
                            if neighbor_depth > 0 and not np.isnan(neighbor_depth) and not np.isinf(neighbor_depth):
                                center_depths.append(neighbor_depth)
                
                if len(center_depths) == 0:
                    print(f"××× 分割 {i+1} 中心点周围无法获取有效深度值")
                    continue
                
                center_depth = np.mean(center_depths)
                center_depth_std = np.std(center_depths)
                
                corner_depth_tolerance = 0.05  # 角点深度与中心点深度的允许差异阈值
                print(f" 分割 {i+1} 中心点深度均值: {center_depth:.3f}, 标准差: {center_depth_std:.3f}, 角点深度与中心点深度的允许差异阈值: {corner_depth_tolerance:.5f} (基于{len(center_depths)}个有效像素)")
                # if center_depth_std >= 0.02:
                #     corner_depth_tolerance = 15 * center_depth_std
                # else:
                #     corner_depth_tolerance = 30 * center_depth_std
                
                # 获取四个角点周围一定范围内的深度值并检查差异
                corner_search_radius = 5  # 角点搜索半径
                corner_depth_valid = True
                
                for j, point in enumerate(bbox_points):
                    x, y = int(point[0]), int(point[1])
                    corner_depths = []
                    
                    # 获取角点周围一定范围内的有效深度值
                    for dy in range(-corner_search_radius, corner_search_radius + 1):
                        for dx in range(-corner_search_radius, corner_search_radius + 1):
                            nx, ny = x + dx, y + dy
                            if (0 <= nx < depth_img.shape[1] and 0 <= ny < depth_img.shape[0]):
                                neighbor_depth = depth_img[ny, nx]
                                if neighbor_depth > 0 and not np.isnan(neighbor_depth) and not np.isinf(neighbor_depth):
                                    corner_depths.append(neighbor_depth)
                    
                    if len(corner_depths) == 0:
                        print(f"××× 分割 {i+1} 角点 {j+1} 周围无法获取有效深度值")
                        corner_depth_valid = False
                        break
                    
                    # 根据中心点深度筛选异常值
                    filtered_corner_depths = []
                    for depth in corner_depths:
                        depth_diff = abs(depth - center_depth)
                        if depth_diff <= corner_depth_tolerance:
                            filtered_corner_depths.append(depth)
                    
                    # 如果筛选后没有有效深度值，说明角点附近所有值都与中心点差异很大
                    if len(filtered_corner_depths) == 0:
                        print(f"××× 分割 {i+1} 角点 {j+1} 附近所有深度值都与中心点差异过大 (收集到{len(corner_depths)}个深度值)")
                        corner_depth_valid = False
                        break
                    
                    # 计算筛选后的深度均值
                    corner_depth_mean = np.mean(filtered_corner_depths)
                    print(f" 分割 {i+1} 角点 {j+1} 深度均值: {corner_depth_mean:.3f} (筛选后{len(filtered_corner_depths)}个有效值)")
                
                if not corner_depth_valid:
                    continue
                
                # 计算凸包
                hull = ConvexHull(points)
                hull_area = hull.volume
                solidity = area / hull_area
                
                # 检查实心度
                if solidity < 0.94:
                    print(f"××× 分割 {i+1} 实心度 {solidity} 小于 0.94")
                    continue
                
                # 检查平行四边形的对边长度一致性
                edge_consistency = True
                if len(edges) == 4:
                    edge_pairs = [(edges[0], edges[2]), (edges[1], edges[3])]
                    for pair in edge_pairs:
                        if abs(pair[0] - pair[1]) / max(pair[0], pair[1]) > 0.25:
                            edge_consistency = False
                            break
                
                if not edge_consistency:
                    print(f"××× 分割 {i+1} 对边长度不一致，对边长度差值 {abs(pair[0] - pair[1])} 大于 25%")
                    continue
                
                # 添加到筛选结果
                if isinstance(annotation, dict):
                    annotation['bbox'] = bbox_points.tolist()
                    annotation['area'] = area
                    annotation['angle'] = angle_deviation
                    annotation['depth_mean'] = depth_mean
                    annotation['depth_std'] = depth_std
                    annotation['solidity'] = solidity
                    annotation['edges'] = edges
                    filtered_annotations.append(annotation)
                else:
                    # 如果annotation不是字典，创建一个新的字典
                    new_annotation = {
                        'segmentation': annotation,
                        'bbox': bbox_points.tolist(),
                        'area': area,
                        'angle': angle_deviation,
                        'depth_mean': depth_mean,
                        'depth_std': depth_std,
                        'solidity': solidity,
                        'edges': edges
                    }
                    filtered_annotations.append(new_annotation)
                
            except:
                continue
        
        return filtered_annotations
    
    def _find_cube_corners(self, mask, depth_img):
        """找到立方体上立面的四个角点"""
        # 确保mask是numpy数组
        mask_np = self._safe_mask_to_numpy(mask)
        
        # 获取掩码边界
        y_coords, x_coords = np.where(mask_np)
        if len(y_coords) < 4:
            return None
        
        points = np.column_stack((x_coords, y_coords))
        
        # 使用优化的平行四边形拟合
        bbox_points = self._optimize_parallelogram_fit(points, mask_np)
        if bbox_points is None:
            return None
        
        # 按左上角为1号角点，顺时针顺序排列角点
        bbox_points = self._sort_corners_clockwise_from_top_left(bbox_points)
        
        corners = bbox_points.tolist()
        
        # 验证角点
        valid_corners = []
        for point in corners:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < depth_img.shape[1] and 0 <= y < depth_img.shape[0]:
                valid_corners.append((x, y))
        
        if len(valid_corners) != 4:
            return None
        
        return valid_corners
    
    def _check_corners_in_box_prompt(self, corners, box_prompt):
        """检查角点是否在box_prompt范围内"""
        if box_prompt is None:
            return True
        
        x1, y1, x2, y2 = box_prompt
        for corner in corners:
            x, y = corner
            if not (x1 <= x <= x2 and y1 <= y <= y2):
                return False
        return True
    
    def estimate_pose(self, rgb_img, depth_img, box_prompt=None, cube_dimensions=(0.117, 0.06), output_dir="./output/", base_name="result"):
        """
        估计立方体姿态
        
        Args:
            rgb_img: RGB图像（方法会自动检测并转换BGR格式）
            depth_img: 深度图像
            box_prompt: 边界框提示 [x, y, x2, y2]，如果为None则使用整个图像
            cube_dimensions: 立方体尺寸 (width, height) in meters
            output_dir: 输出目录路径
            base_name: 输出文件的基础名称
            
        Returns:
            center_position: 上立面中心在相机坐标系中的位置 [x, y, z]
            euler_angles: 上立面坐标系在相机坐标系中的旋转 [roll, pitch, yaw] in degrees
        """
        # 确保输入图像是RGB格式
        if len(rgb_img.shape) == 3 and rgb_img.shape[2] == 3:
            # 检查是否是BGR格式（OpenCV默认格式）
            # 通过检查第一个像素的B和R通道来判断
            if rgb_img[0, 0, 0] > rgb_img[0, 0, 2]:  # B > R，可能是BGR格式
                print("检测到BGR格式图像，正在转换为RGB格式...")
                rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        
        # 初始化FastSAM
        model = FastSAM('FastSAM-x.pt')
        
        # 运行分割
        everything_results = model(rgb_img, device='cpu', retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
        
        # 创建提示
        prompt_process = FastSAMPrompt(rgb_img, everything_results, device='cpu')
        
        
        # 使用everything模式
        ann = prompt_process.everything_prompt()
        
        if ann is None or len(ann) == 0:
            return None, None
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 简单可视化所有分割结果
        print(f"\n检测到 {len(ann)} 个分割对象，正在生成可视化...")
        simple_vis_path = self.simple_segmentation_visualization(rgb_img, ann, output_dir, base_name)
        print(f"所有分割结果可视化已保存到: {simple_vis_path}")
        
        # 筛选立方体分割
        filtered_annotations = self._filter_cube_segments(ann, depth_img)
        
        if filtered_annotations is None or len(filtered_annotations) == 0:
            print(f"错误: 没有筛选带到任何有效的疑似上立面")
            return None, None
        
        # 处理角点选择逻辑
        valid_corners = []
        
        if box_prompt is not None:
            # 在box_prompt范围内查找符合条件的角点
            for annotation in filtered_annotations:
                # 检查annotation的格式
                if isinstance(annotation, dict):
                    mask = annotation['segmentation']
                else:
                    # 如果annotation不是字典，直接使用它作为mask
                    mask = annotation
                
                corners_2d = self._find_cube_corners(mask, depth_img)

                
                if corners_2d is not None:
                    # 检查角点是否在box_prompt范围内
                    if self._check_corners_in_box_prompt(corners_2d, box_prompt):
                        valid_corners.append(corners_2d)
                    else:
                        print(f"角点 {corners_2d} 不在box_prompt范围内")
            
            # 检查是否找到多个符合条件的角点集合
            if len(valid_corners) > 1:
                print(f"错误: 检测到多个边框，找到 {len(valid_corners)} 个符合条件的角点集合")
                return None, None
            elif len(valid_corners) == 1:
                selected_corners = valid_corners[0]
            else:
                print(f"\n错误: 无法在box_prompt范围内找到有效的角点集合")
                print(f"请检查以下可能的问题:")
                print(f"1. box_prompt参数是否正确: {box_prompt}")
                print(f"2. 图像中是否存在目标物体")
                print(f"3. 分割算法是否正确检测到目标")
                print(f"4. 角点检测算法是否正常工作")
                return None, None
        else:
            # 没有提供box_prompt，检查是否只有一组角点
            for annotation in filtered_annotations:
                # 检查annotation的格式
                if isinstance(annotation, dict):
                    mask = annotation['segmentation']
                else:
                    # 如果annotation不是字典，直接使用它作为mask
                    mask = annotation
                
                corners_2d = self._find_cube_corners(mask, depth_img)
                
                if corners_2d is not None:
                    valid_corners.append(corners_2d)
            
            # 检查是否找到多个角点集合
            if len(valid_corners) == 0:
                print(f"\n错误: 没有检测到任何有效的角点集合")
                print(f"请检查以下可能的问题:")
                print(f"1. 图像中是否存在目标物体")
                print(f"2. 分割算法是否正确检测到目标")
                print(f"3. 角点检测算法是否正常工作")
                return None, None
            elif len(valid_corners) > 1:
                print(f"\n错误: 检测到多个边框，找到 {len(valid_corners)} 个角点集合")
                print(f"请使用box_prompt参数缩小检测的边框范围，或者确保图像中只有一个目标物体")
                return None, None
            else:
                # 只有一组角点，使用它
                selected_corners = valid_corners[0]
                print(f"=== 未使用box_prompt,自动选择角点: {selected_corners}")
        
        # 可视化选中的角点（无论是否有box_prompt）
        if box_prompt is not None:
            box_prompt_vis_path = os.path.join(output_dir, base_name + "_box_prompt_result.jpg")
        else:
            box_prompt_vis_path = os.path.join(output_dir, base_name + "_selected_corners.jpg")
        
        box_prompt_vis = self.visualize_box_prompt_result(rgb_img, box_prompt, selected_corners, box_prompt_vis_path)
        print(f"角点可视化已保存到: {box_prompt_vis_path}")
        
        # 获取3D相机坐标
        result = self._get_corner_depths(selected_corners, depth_img)
        if result is None:
            return None, None
        
        corners_3d_camera, depth_validity, center_depth_mean, center_depth_std = result
        
        # 构建立方体物体坐标系
        # width_mm, height_mm = cube_dimensions
        height_mm, width_mm = cube_dimensions
        half_width = width_mm / 2.0
        half_height = height_mm / 2.0
        
        # 假设角点顺序为：左上、右上、右下、左下（顺时针）
        corners_3d_object = np.array([
            [-half_width, -half_height, 0],    # 左上
            [half_width, -half_height, 0],     # 右上
            [half_width, half_height, 0],      # 右下
            [-half_width, half_height, 0]      # 左下
        ], dtype=np.float32)
        
        # 求解刚体变换
        R, t = self._solve_rigid_transform_svd(corners_3d_camera, corners_3d_object)
        if R is None or t is None:
            return None, None
        print(f"旋转矩阵R:")
        print(f"{R}")
        
        # 计算上立面中心在相机坐标系中的位置
        center_position = t  # 因为物体坐标系原点就是上立面中心
        
        # 计算欧拉角
        euler_angles = self._rotation_matrix_to_euler_angles(R)
        
        return center_position, euler_angles


# 使用示例
if __name__ == "__main__":
    # 创建估计器
    estimator = CubePoseEstimator()
    
    # 加载图像
    # rgb_img = cv2.imread("./robot_img2/1/rgb/rgb_0000.jpg")
    # depth_img = cv2.imread("./robot_img2/1/depth/depth_0000.png", cv2.IMREAD_UNCHANGED)
    # rgb_img = cv2.imread("./code_imgs/4/rgb/rgb_0000.jpg")
    # depth_img = cv2.imread("./code_imgs/4/depth/depth_0000.png", cv2.IMREAD_UNCHANGED)
    rgb_img = cv2.imread("./code_imgs_Aruco/2/rgb/rgb_0000.jpg")
    depth_img = cv2.imread("./code_imgs_Aruco/2/depth/depth_0000.png", cv2.IMREAD_UNCHANGED)
    depth_img = depth_img.astype(np.float32) / 1000.0
    
    # 估计姿态
    # center_pos, euler_angles = estimator.estimate_pose(rgb_img, depth_img, box_prompt=[500, 200, 1100, 700], output_dir="./output/", base_name="rgb_0000")
    center_pos, euler_angles = estimator.estimate_pose(rgb_img, depth_img, output_dir="./output/", base_name="rgb_0000")
    if center_pos is not None:
        print(f"中心位置: {center_pos}")
        print(f"欧拉角: {euler_angles}")
    else:
        print("姿态估计失败") 