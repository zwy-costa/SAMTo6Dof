import argparse
from fastsam import FastSAM, FastSAMPrompt 
import ast
import torch
from PIL import Image
from utils.tools import convert_box_xywh_to_xyxy
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免显示问题
from scipy.spatial import ConvexHull
from scipy.ndimage import gaussian_filter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="./weights/FastSAM-x.pt", help="model"
    )
    parser.add_argument(
        "--img_path", type=str, default="./robot_img/rgb/rgb_0000.jpg", help="path to RGB image file"
    )
    parser.add_argument(
        "--depth_path", type=str, default="./robot_img/depth/depth_0000.png", help="path to depth image file"
    )
    parser.add_argument("--imgsz", type=int, default=1024, help="image size")
    parser.add_argument(
        "--iou",
        type=float,
        default=0.9,
        help="iou threshold for filtering the annotations",
    )
    parser.add_argument(
        "--conf", type=float, default=0.4, help="object confidence threshold"
    )
    parser.add_argument(
        "--output", type=str, default="./output/", help="image save path"
    )
    parser.add_argument(
        "--box_prompt", type=str, default="[[0,0,0,0]]", help="[[x,y,w,h],[x2,y2,w2,h2]] support multiple boxes"
    )
    parser.add_argument(
        "--better_quality",
        type=str,
        default=False,
        help="better quality using morphologyEx",
    )
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    parser.add_argument(
        "--device", type=str, default=device, help="cuda:[0,1,2,3,4] or cpu"
    )
    parser.add_argument(
        "--retina",
        type=bool,
        default=True,
        help="draw high-resolution segmentation masks",
    )
    parser.add_argument(
        "--withContours", type=bool, default=False, help="draw the edges of the masks"
    )
    return parser.parse_args()


def load_depth_image(depth_path):
    """加载深度图像"""
    depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    if depth_img is None:
        raise ValueError(f"无法加载深度图像: {depth_path}")
    
    # 深度图像通常是16位，转换为浮点数
    if depth_img.dtype == np.uint16:
        depth_img = depth_img.astype(np.float32) / 65535.0
    elif depth_img.dtype == np.uint8:
        depth_img = depth_img.astype(np.float32) / 255.0
    
    return depth_img


def safe_mask_to_numpy(mask):
    """安全地将mask转换为numpy数组"""
    if hasattr(mask, 'cpu'):
        return mask.cpu().numpy()
    else:
        return mask


def safe_mask_to_bool(mask):
    """安全地将mask转换为布尔数组，用于索引"""
    mask_np = safe_mask_to_numpy(mask)
    return mask_np.astype(bool)


def check_corners_in_box_prompt(corners, box_prompt):
    """检查角点是否都在指定的box_prompt范围内"""
    if corners is None or len(corners) != 4:
        return False
    
    try:
        # 解析box_prompt
        if isinstance(box_prompt, str):
            box_prompt = ast.literal_eval(box_prompt)
        
        # 支持多个box的情况，检查是否在任何一个box内
        if isinstance(box_prompt, list):
            for box in box_prompt:
                if len(box) == 4:  # [x, y, w, h] 格式
                    x, y, w, h = box
                    x2, y2 = x + w, y + h
                    
                    # 检查所有角点是否都在这个box内
                    all_in_box = True
                    for corner in corners:
                        cx, cy = corner
                        if not (x <= cx <= x2 and y <= cy <= y2):
                            all_in_box = False
                            break
                    
                    if all_in_box:
                        return True
        
        return False
        
    except:
        return False


def find_corners_in_box_prompt(corners_list, box_prompt):
    """在box_prompt范围内找到完整的四个角点"""
    if box_prompt is None or box_prompt == "[[0,0,0,0]]":
        return None
    
    print(f"\n=== 在box_prompt范围内查找角点 ===")
    print(f"box_prompt: {box_prompt}")
    
    # 检查每个检测到的角点集合
    for i, corners in enumerate(corners_list):
        if corners is not None and len(corners) == 4:
            if check_corners_in_box_prompt(corners, box_prompt):
                print(f"找到符合条件的角点集合 {i+1}: {corners}")
                return corners
    
    print("未找到完全在box_prompt范围内的角点集合")
    return None


def optimize_parallelogram_fit(points, mask_np):
    """优化平行四边形拟合，使其更符合立方体上立面的特征（透视变形下的平行四边形）"""
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
            # 遍历所有点，选择最接近平行四边形的4个点
            best_points = select_best_parallelogram_points(hull_points)
            # 计算拟合质量
            bbox_array = np.array(best_points)
            mask_area = np.sum(mask_np)
            bbox_area = cv2.contourArea(bbox_array)
            fit_quality = mask_area / bbox_area if bbox_area > 0 else 0
            if abs(fit_quality - 1) < 0.5:
                return best_points
            else:
                return None
    except:
        pass
    
    # # 方法2: 使用轮廓检测找到角点
    # try:
    #     # 创建掩码图像
    #     mask_img = np.zeros((mask_np.shape[0], mask_np.shape[1]), dtype=np.uint8)
    #     mask_img[mask_np] = 255
        
    #     # 找到轮廓
    #     contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     if contours:
    #         # 使用最大的轮廓
    #         largest_contour = max(contours, key=cv2.contourArea)
            
    #         # 使用Douglas-Peucker算法简化轮廓
    #         epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    #         approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
    #         # 如果简化后有4个点，直接使用
    #         if len(approx) == 4:
    #             return approx.reshape(-1, 2)
    #         elif len(approx) > 4:
    #             # 选择最接近平行四边形的4个点
    #             approx_points = approx.reshape(-1, 2)
    #             best_points = select_best_parallelogram_points(approx_points)
    #             return best_points
    # except:
    #     pass
    
    # # 方法3: 使用最小外接矩形作为备选方案
    # try:
    #     rect = cv2.minAreaRect(points)
    #     initial_box = cv2.boxPoints(rect)
    #     initial_box = np.int0(initial_box)
    #     return initial_box
    # except:
    #     pass
    
    return None


def select_best_parallelogram_points(points):
    """从多个点中选择最接近平行四边形的4个点"""
    if len(points) <= 4:
        return points[:4]
    
    # 计算所有可能的4点组合
    from itertools import combinations
    best_points = None
    best_score = float('inf')
    
    for combo in combinations(points, 4):
        combo = np.array(combo)
        
        # 计算平行四边形特征评分
        score = calculate_parallelogram_score(combo)
        if score < best_score:
            best_score = score
            best_points = combo
    
    return best_points if best_points is not None else points[:4]


def calculate_parallelogram_score(points):
    """计算4个点构成平行四边形的质量评分（分数越低越好）"""
    if len(points) != 4:
        return float('inf')
    
    # 使用公共函数计算角度和边长
    angles, edges = calculate_angles_and_edges(points)
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
    
    # 综合评分（权重可调整）
    total_score = (
        angle_consistency_score * 2.0 +      # 对角度数一致性最重要
        edge_consistency_score * 1.5 +       # 对边长度一致性次之
        angle_reasonableness * 0.5 +         # 角度合理性
        compactness_score * 1.0              # 紧凑度
    )
    
    return total_score


def calculate_angles_and_edges(points):
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


def filter_cube_segments(annotations, depth_img, min_area=1000, max_area=50000):
    """基于几何特征和深度信息筛选立方体上立面分割（平行四边形特征）"""
    filtered_annotations = []
    
    for i, annotation in enumerate(annotations):
        mask = annotation['segmentation']
        
        # 确保mask是numpy数组
        mask_np = safe_mask_to_numpy(mask)
        
        # 计算面积
        area = np.sum(mask_np)
        if area < min_area or area > max_area:
            continue
        
        # 计算边界框
        y_coords, x_coords = np.where(mask_np)
        if len(y_coords) == 0:
            continue
        
        # 使用优化的平行四边形拟合，采用多种方法确保准确性
        points = np.column_stack((x_coords, y_coords))
        if len(points) < 4:
            continue
            
        try:
            # 使用优化的平行四边形拟合，找到最符合平行四边形的四个点。采用多种方法确保准确性
            bbox_points = optimize_parallelogram_fit(points, mask_np)
            if bbox_points is None:
                continue
            
            # 按顺时针顺序排列角点
            center = np.mean(bbox_points, axis=0)
            angles = np.arctan2(bbox_points[:, 1] - center[1], bbox_points[:, 0] - center[0])
            sorted_indices = np.argsort(angles)
            bbox_points = bbox_points[sorted_indices]
            
            # 使用公共函数计算角度和边长
            angles_deg, edges = calculate_angles_and_edges(bbox_points)
            if angles_deg is None or edges is None:
                continue
            
            # 计算宽高比（使用对边的平均值）
            width = (edges[0] + edges[2]) / 2  # 对边平均
            height = (edges[1] + edges[3]) / 2  # 对边平均
            
            # 检查宽高比（立方体上立面应该有合理的宽高比）
            aspect_ratio = width / height
            # if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            #     continue
                
            # 检查平行四边形角度特征（对角度数相等，适应透视变形）
            # 检查对角度数是否相近（允许15度误差）
            angle_pairs = [(angles_deg[0], angles_deg[2]), (angles_deg[1], angles_deg[3])]
            angle_consistency = True
            for pair in angle_pairs:
                if abs(pair[0] - pair[1]) > 15:
                    angle_consistency = False
                    break
            
            if not angle_consistency:
                continue
            
            # 计算角度偏差（理想情况下应该在60-120度范围，适应透视）
            angle_deviation = np.mean([abs(angle - 90) for angle in angles_deg])
            if angle_deviation > 45:  # 允许45度的角度偏差，适应透视变形
                continue
            
            # 计算深度统计信息
            depth_values = depth_img[safe_mask_to_bool(mask)]
            if len(depth_values) == 0:
                continue
                
            depth_mean = np.mean(depth_values)
            depth_std = np.std(depth_values)
            
            # 检查深度变化（立方体上立面应该有较小的深度变化，但允许透视变形）
            if depth_std > 0.15:  # 允许更大的深度变化，适应透视效果
                continue
            
            # 计算凸包
            points = np.column_stack((x_coords, y_coords))
            if len(points) < 4:
                continue
                
            hull = ConvexHull(points)
            hull_area = hull.volume  # 对于2D，volume实际上是面积
            solidity = area / hull_area
            
            # 立方体上立面应该有较高的实心度（接近平行四边形）
            if solidity < 0.6:  # 降低要求，适应透视变形
                continue
            
            # 检查平行四边形的对边长度一致性
            edge_consistency = True
            if len(edges) == 4:
                # 检查对边长度是否相近（允许25%误差，适应透视变形）
                edge_pairs = [(edges[0], edges[2]), (edges[1], edges[3])]
                for pair in edge_pairs:
                    if abs(pair[0] - pair[1]) / max(pair[0], pair[1]) > 0.25:
                        edge_consistency = False
                        break
            
            if not edge_consistency:
                continue
                
        except:
            continue
        
        # 评估拟合质量，传入预计算的角度和边长
        quality_metrics = evaluate_fitting_quality(mask_np, bbox_points, angles_deg, edges)
        
        # 添加到筛选结果
        annotation['bbox'] = bbox_points.tolist()  # 使用拟合的平行四边形顶点
        annotation['area'] = area
        # annotation['aspect_ratio'] = aspect_ratio
        annotation['angle'] = angle_deviation  # 角度偏差
        annotation['depth_mean'] = depth_mean
        annotation['depth_std'] = depth_std
        annotation['solidity'] = solidity
        annotation['edges'] = edges  # 保存边长信息
        annotation['quality_metrics'] = quality_metrics  # 保存质量评估结果
        filtered_annotations.append(annotation)
    
    return filtered_annotations


def find_cube_corners(mask, depth_img, bbox_points=None, visualize=True):
    """找到立方体上立面的四个角点"""
    # 确保mask是numpy数组
    mask_np = safe_mask_to_numpy(mask)
    
    # 如果提供了边界框点，直接使用
    # if bbox_points is not None:
    #     corners = bbox_points.tolist()
    if 1 == 2:
        print("1 == 2")
    else:
        # 获取掩码边界
        y_coords, x_coords = np.where(mask_np)
        if len(y_coords) < 4:
            return None, None
        
        points = np.column_stack((x_coords, y_coords))
        
        # 使用优化的平行四边形拟合
        bbox_points = optimize_parallelogram_fit(points, mask_np)
        # bbox_points = find_corners_by_corner_detection(points, mask_np)
        if bbox_points is None:
            return None, None
        
        corners = bbox_points.tolist()
    
    # 验证角点
    valid_corners = []
    for point in corners:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < depth_img.shape[1] and 0 <= y < depth_img.shape[0]:
            valid_corners.append((x, y))
    
    if len(valid_corners) != 4:
        return None, None
    
    # 可视化角点
    if visualize:
        # 创建可视化图像
        vis_img = np.zeros((depth_img.shape[0], depth_img.shape[1], 3), dtype=np.uint8)
        vis_img[safe_mask_to_bool(mask)] = [255, 255, 255]  # 白色掩码
        
        # 绘制角点
        for i, (x, y) in enumerate(valid_corners):
            cv2.circle(vis_img, (x, y), 8, (0, 255, 0), -1)  # 绿色角点
            cv2.putText(vis_img, str(i+1), (x+10, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 绘制边界
        for i in range(4):
            pt1 = valid_corners[i]
            pt2 = valid_corners[(i+1) % 4]
            cv2.line(vis_img, pt1, pt2, (0, 0, 255), 2)  # 红色边界
        
        return valid_corners, vis_img
    
    return valid_corners, None


def visualize_fitting_process(rgb_img, mask_np, bbox_points, output_dir, base_name, object_id, angles=None, edges=None):
    """可视化平行四边形拟合过程"""
    # 创建可视化图像
    vis_img = rgb_img.copy()
    
    # 绘制掩码
    mask_bool = mask_np.astype(bool)
    vis_img[mask_bool] = vis_img[mask_bool] * 0.8 + np.array([255, 255, 255]) * 0.2
    
    # 绘制拟合的平行四边形
    bbox_array = np.array(bbox_points, dtype=np.int32)
    cv2.polylines(vis_img, [bbox_array], True, (0, 255, 0), 3)
    
    # 绘制角点
    for i, point in enumerate(bbox_points):
        x, y = int(point[0]), int(point[1])
        cv2.circle(vis_img, (x, y), 8, (255, 0, 0), -1)  # 蓝色角点
        cv2.putText(vis_img, str(i+1), (x+10, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # 计算并显示拟合参数
    # 如果没有提供预计算的角度和边长，则计算
    if angles is None or edges is None:
        angles, edges = calculate_angles_and_edges(bbox_points)
        if angles is None or edges is None:
            angles, edges = [], []
    
    # 计算拟合质量
    mask_area = np.sum(mask_np)
    bbox_area = cv2.contourArea(bbox_array)
    fit_quality = mask_area / bbox_area if bbox_area > 0 else 0
    
    # 在图像上显示参数
    param_text = f"Object {object_id} Fitting Parameters:"
    param_text += f"\nFit Quality: {fit_quality:.3f}"
    param_text += f"\nEdge Lengths: {[f'{e:.1f}' for e in edges]}"
    param_text += f"\nAngles: {[f'{a:.1f}°' for a in angles]}"
    param_text += f"\nAngle Deviation: {np.mean([abs(a-90) for a in angles]):.1f}°"
    
    # 在图像左上角显示参数
    y_offset = 30
    for line in param_text.split('\n'):
        cv2.putText(vis_img, line, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
    
    # 保存结果
    output_path = output_dir + base_name + f"_fitting_process_{object_id}.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
    
    return output_path


def evaluate_fitting_quality(mask_np, bbox_points, angles=None, edges=None):
    """评估平行四边形拟合质量"""
    # 计算掩码面积
    mask_area = np.sum(mask_np)
    
    # 计算边界框面积
    bbox_array = np.array(bbox_points, dtype=np.int32)
    bbox_area = cv2.contourArea(bbox_array)
    
    # 计算拟合质量（掩码面积与边界框面积的比例）
    fit_quality = mask_area / bbox_area if bbox_area > 0 else 0
    
    # 如果没有提供预计算的角度和边长，则计算
    if angles is None or edges is None:
        angles, edges = calculate_angles_and_edges(bbox_points)
        if angles is None or edges is None:
            return {
                'fit_quality': 0,
                'angle_deviation': 0,
                'edge_consistency': 0,
                'quality_score': 0,
                'quality_grade': "Poor",
                'angles': [],
                'edges': []
            }
    
    angle_deviation = np.mean([abs(angle - 90) for angle in angles]) if angles else 0
    
    edge_consistency = np.std(edges) / np.mean(edges) if np.mean(edges) > 0 else 0
    
    # 评估等级（针对平行四边形特征调整）
    quality_score = 0
    quality_grade = "Poor"
    
    if fit_quality > 0.75 and angle_deviation < 20 and edge_consistency < 0.15:
        quality_score = 3
        quality_grade = "Excellent"
    elif fit_quality > 0.65 and angle_deviation < 35 and edge_consistency < 0.25:
        quality_score = 2
        quality_grade = "Good"
    elif fit_quality > 0.55 and angle_deviation < 45 and edge_consistency < 0.35:
        quality_score = 1
        quality_grade = "Fair"
    
    return {
        'fit_quality': fit_quality,
        'angle_deviation': angle_deviation,
        'edge_consistency': edge_consistency,
        'quality_score': quality_score,
        'quality_grade': quality_grade,
        'angles': angles,
        'edges': edges
    }


def create_comprehensive_visualization(rgb_img, depth_img, annotations, corners_list, output_dir, base_name):
    """创建综合可视化结果"""
    # 创建一个大画布来显示所有结果
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Cube Detection and Corner Recognition Results - {base_name}', fontsize=16, fontweight='bold')
    
    # 1. 原始RGB图像
    axes[0, 0].imshow(rgb_img)
    axes[0, 0].set_title('Original RGB Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 2. 深度图像
    depth_normalized = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())
    axes[0, 1].imshow(depth_normalized, cmap='viridis')
    axes[0, 1].set_title('Depth Image', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # 3. 分割结果
    segmentation_vis = rgb_img.copy()
    for i, annotation in enumerate(annotations):
        mask = annotation['segmentation']
        mask_bool = safe_mask_to_bool(mask)
        color = np.random.randint(0, 255, 3).tolist()
        segmentation_vis[mask_bool] = segmentation_vis[mask_bool] * 0.7 + np.array(color) * 0.3
    axes[0, 2].imshow(segmentation_vis)
    axes[0, 2].set_title(f'Segmentation Results ({len(annotations)} objects)', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # 4. 角点检测结果
    corners_vis = rgb_img.copy()
    valid_corners_count = 0
    for i, corners in enumerate(corners_list):
        if corners is not None:
            valid_corners_count += 1
            # 绘制角点
            for j, (x, y) in enumerate(corners):
                cv2.circle(corners_vis, (x, y), 8, (0, 255, 0), -1)
                cv2.putText(corners_vis, str(j+1), (x+10, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 绘制边界
            for j in range(4):
                pt1 = corners[j]
                pt2 = corners[(j+1) % 4]
                cv2.line(corners_vis, pt1, pt2, (0, 0, 255), 2)
    axes[1, 0].imshow(corners_vis)
    axes[1, 0].set_title(f'Corner Detection ({valid_corners_count} valid)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # 5. 最终结果叠加
    final_vis = rgb_img.copy()
    for i, annotation in enumerate(annotations):
        mask = annotation['segmentation']
        mask_bool = safe_mask_to_bool(mask)
        color = np.random.randint(0, 255, 3).tolist()
        final_vis[mask_bool] = final_vis[mask_bool] * 0.8 + np.array(color) * 0.2
    
    for corners in corners_list:
        if corners is not None:
            for j, (x, y) in enumerate(corners):
                cv2.circle(final_vis, (x, y), 10, (0, 255, 0), -1)
                cv2.putText(final_vis, str(j+1), (x+15, y-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            for j in range(4):
                pt1 = corners[j]
                pt2 = corners[(j+1) % 4]
                cv2.line(final_vis, pt1, pt2, (0, 0, 255), 3)
    
    axes[1, 1].imshow(final_vis)
    axes[1, 1].set_title('Final Result', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # 6. 统计信息
    axes[1, 2].axis('off')
    stats_text = f"""Detection Statistics:

    Original segmentation count: {len(annotations)}
    Filtered count: {len(annotations)}
    Valid corners count: {valid_corners_count}

    Image size: {rgb_img.shape[1]} x {rgb_img.shape[0]}
    Depth range: {depth_img.min():.3f} - {depth_img.max():.3f}"""
    
    for i, annotation in enumerate(annotations):
        stats_text += f"\nObject {i+1}:"
        stats_text += f"\n  Area: {annotation['area']:.0f}"
        # stats_text += f"\n  Aspect ratio: {annotation['aspect_ratio']:.2f}"
        stats_text += f"\n  Angle deviation: {annotation['angle']:.1f}°"
        stats_text += f"\n  Depth mean: {annotation['depth_mean']:.3f}"
        stats_text += f"\n  Solidity: {annotation['solidity']:.3f}"
    
    axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存综合可视化结果
    comprehensive_output_path = output_dir + base_name + "_comprehensive_visualization.png"
    plt.savefig(comprehensive_output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return comprehensive_output_path


def visualize_fitted_rectangles(rgb_img, filtered_annotations, output_dir, base_name):
    """可视化拟合的矩形"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. 原始分割结果
    original_vis = rgb_img.copy()
    for i, annotation in enumerate(filtered_annotations):
        mask = annotation['segmentation']
        mask_bool = safe_mask_to_bool(mask)
        color = np.random.randint(0, 255, 3).tolist()
        original_vis[mask_bool] = original_vis[mask_bool] * 0.8 + np.array(color) * 0.2
    axes[0].imshow(original_vis)
    axes[0].set_title('Original Segmentation Results', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # 2. 拟合矩形结果
    fitted_vis = rgb_img.copy()
    for i, annotation in enumerate(filtered_annotations):
        mask = annotation['segmentation']
        mask_bool = safe_mask_to_bool(mask)
        color = np.random.randint(0, 255, 3).tolist()
        fitted_vis[mask_bool] = fitted_vis[mask_bool] * 0.8 + np.array(color) * 0.2
        
        # 绘制拟合的平行四边形
        bbox = annotation['bbox']
        bbox = np.array(bbox, dtype=np.int32)
        cv2.polylines(fitted_vis, [bbox], True, (0, 255, 0), 2)
        
        # 标注ID和参数
        centroid = np.mean(bbox, axis=0).astype(int)
        cv2.putText(fitted_vis, f"{i+1}", (centroid[0]-10, centroid[1]+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 在图像上显示参数
        quality_grade = annotation['quality_metrics']['quality_grade']
        # param_text = f"A:{annotation['aspect_ratio']:.1f} θ:{annotation['angle']:.0f}° {quality_grade}"
        param_text = f"θ:{annotation['angle']:.0f}° {quality_grade}"
        cv2.putText(fitted_vis, param_text, (centroid[0]-30, centroid[1]+25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    axes[1].imshow(fitted_vis)
    axes[1].set_title('Fitted Rectangles with Parameters', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # 保存结果
    fitted_output_path = output_dir + base_name + "_fitted_rectangles.png"
    plt.savefig(fitted_output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return fitted_output_path


def create_segmentation_process_visualization(rgb_img, all_annotations, filtered_annotations, output_dir, base_name):
    """创建分割过程可视化"""
    # 创建画布
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 所有分割结果（带ID标注）
    all_seg_vis = rgb_img.copy()
    for i, annotation in enumerate(all_annotations):
        mask = annotation['segmentation']
        mask_bool = safe_mask_to_bool(mask)
        color = np.random.randint(0, 255, 3).tolist()
        all_seg_vis[mask_bool] = all_seg_vis[mask_bool] * 0.8 + np.array(color) * 0.2
        
        # 计算质心并标注ID
        y_coords, x_coords = np.where(mask_bool)
        if len(y_coords) > 0:
            centroid_y, centroid_x = int(np.mean(y_coords)), int(np.mean(x_coords))
            # 使用OpenCV在图像上标注ID
            cv2.putText(all_seg_vis, str(i+1), (centroid_x-10, centroid_y+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.circle(all_seg_vis, (centroid_x, centroid_y), 5, (255, 255, 255), -1)
    
    axes[0, 0].imshow(all_seg_vis)
    axes[0, 0].axis('off')
    
    # 2. 筛选后的分割结果（带ID标注）
    filtered_seg_vis = rgb_img.copy()
    for i, annotation in enumerate(filtered_annotations):
        mask = annotation['segmentation']
        mask_bool = safe_mask_to_bool(mask)
        color = np.random.randint(0, 255, 3).tolist()
        filtered_seg_vis[mask_bool] = filtered_seg_vis[mask_bool] * 0.8 + np.array(color) * 0.2
        
        # 计算质心并标注ID
        y_coords, x_coords = np.where(mask_bool)
        if len(y_coords) > 0:
            centroid_y, centroid_x = int(np.mean(y_coords)), int(np.mean(x_coords))
            # 使用OpenCV在图像上标注ID
            cv2.putText(filtered_seg_vis, str(i+1), (centroid_x-10, centroid_y+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.circle(filtered_seg_vis, (centroid_x, centroid_y), 5, (255, 255, 255), -1)
    
    axes[0, 1].imshow(filtered_seg_vis)
    axes[0, 1].axis('off')
    
    # 3. 筛选统计信息
    axes[1, 0].axis('off')
    
    # 4. 筛选后的详细信息
    axes[1, 1].axis('off')
    
    # 调整布局
    # 设置matplotlib使用英文显示
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
    
    # 设置英文标题
    fig.suptitle(f'Segmentation Process Visualization - {base_name}', fontsize=16, fontweight='bold')
    axes[0, 0].set_title(f'All Segmentation Results ({len(all_annotations)} objects)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title(f'Filtered Results ({len(filtered_annotations)} objects)', fontsize=12, fontweight='bold')
    
    # 英文版本文本
    stats_text = f"""Segmentation Filter Statistics:

    Original count: {len(all_annotations)}
    Filtered count: {len(filtered_annotations)}
    Filter rate: {len(filtered_annotations)/len(all_annotations)*100:.1f}%

    Filter conditions:
    - Area range: 1000-50000
    - Aspect ratio: 0.5-2.0
    - Angle deviation: < 30°
    - Depth std: < 0.1
    - Solidity: > 0.7"""
    
    detail_text = "Filtered object details:\n\n" if len(filtered_annotations) > 0 else "No objects passed filter"
    if len(filtered_annotations) > 0:
        for i, annotation in enumerate(filtered_annotations):
            detail_text += f"Object {i+1}:\n"
            detail_text += f"  Area: {annotation['area']:.0f}\n"
            # detail_text += f"  Aspect ratio: {annotation['aspect_ratio']:.2f}\n"
            detail_text += f"  Angle deviation: {annotation['angle']:.1f}°\n"
            detail_text += f"  Depth mean: {annotation['depth_mean']:.3f}\n"
            detail_text += f"  Depth std: {annotation['depth_std']:.3f}\n"
            detail_text += f"  Solidity: {annotation['solidity']:.3f}\n\n"
    
    # 显示统计信息
    axes[1, 0].text(0.05, 0.95, stats_text, transform=axes[1, 0].transAxes, 
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    # 显示详细信息
    axes[1, 1].text(0.05, 0.95, detail_text, transform=axes[1, 1].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    
    plt.tight_layout()
    
    # 保存结果
    process_output_path = output_dir + base_name + "_segmentation_process.png"
    plt.savefig(process_output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return process_output_path


def visualize_box_prompt_result(rgb_img, box_prompt, selected_corners, output_path):
    """可视化box_prompt和选中的角点"""
    vis_img = rgb_img.copy()
    
    try:
        # 解析box_prompt
        if isinstance(box_prompt, str):
            box_prompt = ast.literal_eval(box_prompt)
        
        # 绘制所有box_prompt区域
        if isinstance(box_prompt, list):
            for i, box in enumerate(box_prompt):
                if len(box) == 4:  # [x, y, w, h] 格式
                    x, y, w, h = box
                    x2, y2 = x + w, y + h
                    
                    # 绘制box_prompt区域（半透明蓝色）
                    overlay = vis_img.copy()
                    cv2.rectangle(overlay, (int(x), int(y)), (int(x2), int(y2)), (255, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.3, vis_img, 0.7, 0, vis_img)
                    
                    # 绘制box_prompt边界（蓝色）
                    cv2.rectangle(vis_img, (int(x), int(y)), (int(x2), int(y2)), (255, 0, 0), 2)
                    
                    # 标注box编号
                    cv2.putText(vis_img, f"Box {i+1}", (int(x), int(y)-10), 
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
            
            # 添加标题
            cv2.putText(vis_img, "Selected Corners in Box Prompt", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            # 添加标题
            cv2.putText(vis_img, "No Corners Found in Box Prompt", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    except Exception as e:
        print(f"可视化box_prompt时出错: {e}")
    
    # 保存结果
    cv2.imwrite(output_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
    
    return vis_img


def visualize_results(rgb_img, depth_img, annotations, corners_list, output_path):
    """可视化结果"""
    # 创建RGB可视化
    rgb_vis = rgb_img.copy()
    
    # 绘制分割结果
    for i, annotation in enumerate(annotations):
        mask = annotation['segmentation']
        mask_bool = safe_mask_to_bool(mask)
        color = np.random.randint(0, 255, 3).tolist()
        rgb_vis[mask_bool] = rgb_vis[mask_bool] * 0.7 + np.array(color) * 0.3
    
    # 绘制角点
    for corners in corners_list:
        if corners is not None:
            for j, (x, y) in enumerate(corners):
                cv2.circle(rgb_vis, (x, y), 10, (0, 255, 0), -1)
                cv2.putText(rgb_vis, str(j+1), (x+15, y-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # 绘制边界
            for j in range(4):
                pt1 = corners[j]
                pt2 = corners[(j+1) % 4]
                cv2.line(rgb_vis, pt1, pt2, (0, 0, 255), 3)
    
    # 保存结果
    cv2.imwrite(output_path, cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2BGR))
    
    return rgb_vis


def simple_segmentation_visualization(rgb_img, annotations, output_dir, base_name):
    """简单的分割结果可视化，显示所有分割对象并标注ID"""
    # 创建可视化图像
    vis_img = rgb_img.copy()
    
    # 为每个分割对象添加颜色和ID标注
    for i, annotation in enumerate(annotations):
        mask = annotation['segmentation']
        mask_bool = safe_mask_to_bool(mask)
        
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
    output_path = output_dir + base_name + "_all_segmentations.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
    
    return output_path


def main(args):
    # 加载模型
    model = FastSAM(args.model_path)
    args.box_prompt = convert_box_xywh_to_xyxy(ast.literal_eval(args.box_prompt))
    
    # 加载RGB图像（使用PIL，与原始脚本一致）
    input_pil = Image.open(args.img_path)
    input_pil = input_pil.convert("RGB")
    
    # 加载深度图像
    depth_img = load_depth_image(args.depth_path)
    
    # 确保深度图像与RGB图像尺寸一致
    rgb_size = input_pil.size  # (width, height)
    if depth_img.shape[1] != rgb_size[0] or depth_img.shape[0] != rgb_size[1]:
        depth_img = cv2.resize(depth_img, (rgb_size[0], rgb_size[1]))
    
    # 转换为numpy数组用于后续处理
    rgb_img = np.array(input_pil)
    
    # FastSAM分割
    everything_results = model(
        input_pil,
        device=args.device,
        retina_masks=args.retina,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou    
    )
    
    # 解析结果
    print("=== FastSAM Results Analysis ===")
    print(f"检测到的对象数量: {len(everything_results[0].boxes.data)}")
    print(f"图像尺寸: {everything_results[0].orig_shape}")
    
    # 处理分割结果
    prompt_process = FastSAMPrompt(input_pil, everything_results, device=args.device)
    ann = prompt_process.everything_prompt()
    
    # 格式化结果
    annotations = []
    for i, mask in enumerate(ann):
        mask_np = safe_mask_to_numpy(mask)
        annotation = {
            'id': i,
            'segmentation': mask,  # 保持原始mask用于后续处理
            'area': np.sum(mask_np)
        }
        annotations.append(annotation)

    # 简单可视化所有分割结果
    print(f"\n检测到 {len(annotations)} 个分割对象，正在生成可视化...")
    base_name = args.img_path.split("/")[-1].replace('.jpg', '')
    simple_vis_path = simple_segmentation_visualization(rgb_img, annotations, args.output, base_name)
    print(f"所有分割结果可视化已保存到: {simple_vis_path}")
    
    # 筛选立方体上立面分割
    print("\n=== 筛选立方体上立面分割 ===")
    filtered_annotations = filter_cube_segments(annotations, depth_img)
    print(f"筛选后的分割数量: {len(filtered_annotations)}")
    
    # 显示筛选结果
    for i, annotation in enumerate(filtered_annotations):
        quality_grade = annotation['quality_metrics']['quality_grade']
        print(f"  分割 {i+1}: 面积={annotation['area']:.0f}, "
            #   f"宽高比={annotation['aspect_ratio']:.2f}, "
              f"角度偏差={annotation['angle']:.1f}°, "
              f"深度均值={annotation['depth_mean']:.3f}, "
              f"深度标准差={annotation['depth_std']:.3f}, "
              f"实心度={annotation['solidity']:.3f}, "
              f"质量等级={quality_grade}")
    
    # 创建分割过程可视化
    if len(filtered_annotations) > 0:
        segmentation_process_path = create_segmentation_process_visualization(
            rgb_img, annotations, filtered_annotations, args.output, args.img_path.split("/")[-1].replace('.jpg', '')
        )
        print(f"分割过程可视化已保存到: {segmentation_process_path}")
        
    # 创建拟合矩形可视化
    fitted_rectangles_path = visualize_fitted_rectangles(
        rgb_img, filtered_annotations, args.output, args.img_path.split("/")[-1].replace('.jpg', '')
    )
    print(f"拟合矩形可视化已保存到: {fitted_rectangles_path}")
    
    # 创建拟合过程可视化
    for i, annotation in enumerate(filtered_annotations):
        mask = annotation['segmentation']
        mask_np = safe_mask_to_numpy(mask)
        bbox_points = np.array(annotation['bbox'])
        # 获取预计算的角度和边长
        angles = annotation['quality_metrics']['angles']
        edges = annotation['quality_metrics']['edges']
        fitting_process_path = visualize_fitting_process(
            rgb_img, mask_np, bbox_points, args.output, 
            args.img_path.split("/")[-1].replace('.jpg', ''), i+1, angles, edges
        )
        print(f"拟合过程可视化 {i+1} 已保存到: {fitting_process_path}")
    
    # 找到角点
    print("\n=== 角点检测 ===")
    corners_list = []
    corner_vis_list = []
    
    for i, annotation in enumerate(filtered_annotations):
        mask = annotation['segmentation']
        bbox_points = np.array(annotation['bbox'])
        corners, corner_vis = find_cube_corners(mask, depth_img, bbox_points=bbox_points, visualize=True)
        
        if corners is not None:
            print(f"  分割 {i+1} 角点: {corners}")
            corners_list.append(corners)
            if corner_vis is not None:
                corner_vis_list.append(corner_vis)
        else:
            print(f"  分割 {i+1}: 未找到有效角点")
            corners_list.append(None)


    # 输出指定box_prompt范围内完整的四个角点做为结果
    if args.box_prompt is not None:
        # 在box_prompt范围内查找符合条件的角点
        selected_corners = find_corners_in_box_prompt(corners_list, args.box_prompt)
        
        if selected_corners is not None:
            print(f"\n=== 最终结果 ===")
            print(f"在box_prompt范围内找到的角点: {selected_corners}")
            
            # 保存最终结果到文件
            result_file = args.output + base_name + "_final_corners.txt"
            with open(result_file, 'w') as f:
                f.write("Final corners in box_prompt:\n")
                for i, corner in enumerate(selected_corners):
                    f.write(f"Corner {i+1}: ({corner[0]}, {corner[1]})\n")
            print(f"最终角点结果已保存到: {result_file}")
            
            # 可视化box_prompt和选中的角点
            box_prompt_vis_path = args.output + base_name + "_box_prompt_result.jpg"
            box_prompt_vis = visualize_box_prompt_result(rgb_img, args.box_prompt, selected_corners, box_prompt_vis_path)
            print(f"box_prompt结果可视化已保存到: {box_prompt_vis_path}")
        else:
            print(f"\n=== 结果 ===")
            print("未在box_prompt范围内找到完整的四个角点")
        


    
    # 可视化结果
    print("\n=== 保存结果 ===")
    base_name = args.img_path.split("/")[-1].replace('.jpg', '')
    
    # 创建综合可视化
    comprehensive_path = create_comprehensive_visualization(
        rgb_img, depth_img, filtered_annotations, corners_list, args.output, base_name
    )
    print(f"综合可视化结果已保存到: {comprehensive_path}")
    
    # 保存RGB可视化结果
    rgb_output_path = args.output + base_name + "_cube_corners.jpg"
    rgb_vis = visualize_results(rgb_img, depth_img, filtered_annotations, corners_list, rgb_output_path)
    print(f"RGB可视化结果已保存到: {rgb_output_path}")
    
    # 保存角点可视化结果
    for i, corner_vis in enumerate(corner_vis_list):
        if corner_vis is not None:
            corner_output_path = args.output + base_name + f"_corners_{i+1}.jpg"
            cv2.imwrite(corner_output_path, corner_vis)
            print(f"角点可视化 {i+1} 已保存到: {corner_output_path}")
    
    # 保存深度图可视化
    depth_vis = (depth_img * 255).astype(np.uint8)
    depth_output_path = args.output + base_name + "_depth.jpg"
    cv2.imwrite(depth_output_path, depth_vis)
    print(f"深度图已保存到: {depth_output_path}")
    
    # 保存最终结果的大图
    final_large_path = args.output + base_name + "_final_result_large.png"
    plt.figure(figsize=(20, 15))
    plt.imshow(rgb_vis)
    plt.title(f'Cube Detection Final Result - {base_name}', fontsize=18, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(final_large_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"最终结果大图已保存到: {final_large_path}")
    
    # 显示生成的文件总结
    print("\n=== 生成的文件总结 ===")
    generated_files = [
        simple_vis_path,  # 添加简单可视化
        comprehensive_path,
        rgb_output_path,
        depth_output_path,
        final_large_path
    ]
    
    # 添加box_prompt相关文件
    if args.box_prompt is not None and args.box_prompt != "[[0,0,0,0]]":
        result_file = args.output + base_name + "_final_corners.txt"
        box_prompt_vis_path = args.output + base_name + "_box_prompt_result.jpg"
        generated_files.append(result_file)
        generated_files.append(box_prompt_vis_path)
    
    if len(filtered_annotations) > 0:
        generated_files.append(segmentation_process_path)
        generated_files.append(fitted_rectangles_path)
        # 添加拟合过程可视化文件
        for i, annotation in enumerate(filtered_annotations):
            fitting_process_path = args.output + base_name + f"_fitting_process_{i+1}.jpg"
            generated_files.append(fitting_process_path)
        for i, corner_vis in enumerate(corner_vis_list):
            if corner_vis is not None:
                corner_output_path = args.output + base_name + f"_corners_{i+1}.jpg"
                generated_files.append(corner_output_path)
    
    for i, file_path in enumerate(generated_files, 1):
        print(f"{i}. {file_path}")
    
    print(f"\n总共生成了 {len(generated_files)} 个可视化文件")
    print("所有可视化结果已保存完成，请查看输出目录中的图像文件")
    
    print("\n=== 处理完成 ===")


if __name__ == "__main__":
    args = parse_args()
    main(args) 