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


def filter_cube_segments(annotations, depth_img, min_area=1000, max_area=50000):
    """基于几何特征和深度信息筛选立方体分割"""
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
            
        bbox = [np.min(x_coords), np.min(y_coords), np.max(x_coords), np.max(y_coords)]
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        # 检查宽高比（立方体应该有合理的宽高比）
        aspect_ratio = width / height
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            continue
        
        # 计算深度统计信息
        depth_values = depth_img[safe_mask_to_bool(mask)]
        if len(depth_values) == 0:
            continue
            
        depth_mean = np.mean(depth_values)
        depth_std = np.std(depth_values)
        
        # 检查深度变化（上立面应该有较小的深度变化）
        if depth_std > 0.1:  # 深度变化太大，可能不是上立面
            continue
        
        # 计算凸包
        points = np.column_stack((x_coords, y_coords))
        if len(points) < 4:
            continue
            
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume  # 对于2D，volume实际上是面积
            solidity = area / hull_area
            
            # 立方体上立面应该有较高的实心度
            if solidity < 0.7:
                continue
                
        except:
            continue
        
        # 添加到筛选结果
        annotation['bbox'] = bbox
        annotation['area'] = area
        annotation['aspect_ratio'] = aspect_ratio
        annotation['depth_mean'] = depth_mean
        annotation['depth_std'] = depth_std
        annotation['solidity'] = solidity
        filtered_annotations.append(annotation)
    
    return filtered_annotations


def find_cube_corners(mask, depth_img, visualize=True):
    """找到立方体上立面的四个角点"""
    # 确保mask是numpy数组
    mask_np = safe_mask_to_numpy(mask)
    
    # 获取掩码边界
    y_coords, x_coords = np.where(mask_np)
    if len(y_coords) < 4:
        return None, None
    
    points = np.column_stack((x_coords, y_coords))
    
    # 使用凸包找到边界点
    try:
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
    except:
        return None, None
    
    # 如果凸包点数不等于4，尝试找到最接近矩形的4个点
    if len(hull_points) != 4:
        # 计算最小外接矩形
        rect = cv2.minAreaRect(hull_points)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        hull_points = box
    
    # 按顺时针顺序排列角点
    hull_points = np.array(hull_points)
    center = np.mean(hull_points, axis=0)
    angles = np.arctan2(hull_points[:, 1] - center[1], hull_points[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    hull_points = hull_points[sorted_indices]
    
    # 验证角点
    corners = []
    for point in hull_points:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < depth_img.shape[1] and 0 <= y < depth_img.shape[0]:
            corners.append((x, y))
    
    if len(corners) != 4:
        return None, None
    
    # 可视化角点
    if visualize:
        # 创建可视化图像
        vis_img = np.zeros((depth_img.shape[0], depth_img.shape[1], 3), dtype=np.uint8)
        vis_img[safe_mask_to_bool(mask)] = [255, 255, 255]  # 白色掩码
        
        # 绘制角点
        for i, (x, y) in enumerate(corners):
            cv2.circle(vis_img, (x, y), 8, (0, 255, 0), -1)  # 绿色角点
            cv2.putText(vis_img, str(i+1), (x+10, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 绘制边界
        for i in range(4):
            pt1 = corners[i]
            pt2 = corners[(i+1) % 4]
            cv2.line(vis_img, pt1, pt2, (0, 0, 255), 2)  # 红色边界
        
        return corners, vis_img
    
    return corners, None


def create_comprehensive_visualization(rgb_img, depth_img, annotations, corners_list, output_dir, base_name):
    """创建综合可视化结果"""
    # 创建一个大画布来显示所有结果
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'立方体检测与角点识别结果 - {base_name}', fontsize=16, fontweight='bold')
    
    # 1. 原始RGB图像
    axes[0, 0].imshow(rgb_img)
    axes[0, 0].set_title('原始RGB图像', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 2. 深度图像
    depth_normalized = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())
    axes[0, 1].imshow(depth_normalized, cmap='viridis')
    axes[0, 1].set_title('深度图像', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # 3. 分割结果
    segmentation_vis = rgb_img.copy()
    for i, annotation in enumerate(annotations):
        mask = annotation['segmentation']
        mask_bool = safe_mask_to_bool(mask)
        color = np.random.randint(0, 255, 3).tolist()
        segmentation_vis[mask_bool] = segmentation_vis[mask_bool] * 0.7 + np.array(color) * 0.3
    axes[0, 2].imshow(segmentation_vis)
    axes[0, 2].set_title(f'分割结果 ({len(annotations)}个对象)', fontsize=12, fontweight='bold')
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
    axes[1, 0].set_title(f'角点检测 ({valid_corners_count}个有效)', fontsize=12, fontweight='bold')
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
    axes[1, 1].set_title('最终结果', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # 6. 统计信息
    axes[1, 2].axis('off')
    stats_text = f"""
    检测统计信息:
    
    原始分割数量: {len(annotations)}
    筛选后数量: {len(annotations)}
    有效角点数量: {valid_corners_count}
    
    图像尺寸: {rgb_img.shape[1]} x {rgb_img.shape[0]}
    深度范围: {depth_img.min():.3f} - {depth_img.max():.3f}
    """
    
    for i, annotation in enumerate(annotations):
        stats_text += f"\n对象 {i+1}:"
        stats_text += f"\n  面积: {annotation['area']:.0f}"
        stats_text += f"\n  宽高比: {annotation['aspect_ratio']:.2f}"
        stats_text += f"\n  深度均值: {annotation['depth_mean']:.3f}"
        stats_text += f"\n  实心度: {annotation['solidity']:.3f}"
    
    axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存综合可视化结果
    comprehensive_output_path = output_dir + base_name + "_comprehensive_visualization.png"
    plt.savefig(comprehensive_output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return comprehensive_output_path


def create_segmentation_process_visualization(rgb_img, all_annotations, filtered_annotations, output_dir, base_name):
    """创建分割过程可视化"""
    # 创建画布
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'分割过程可视化 - {base_name}', fontsize=16, fontweight='bold')
    
    # 1. 所有分割结果
    all_seg_vis = rgb_img.copy()
    for i, annotation in enumerate(all_annotations):
        mask = annotation['segmentation']
        mask_bool = safe_mask_to_bool(mask)
        color = np.random.randint(0, 255, 3).tolist()
        all_seg_vis[mask_bool] = all_seg_vis[mask_bool] * 0.8 + np.array(color) * 0.2
    axes[0, 0].imshow(all_seg_vis)
    axes[0, 0].set_title(f'所有分割结果 ({len(all_annotations)}个)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 2. 筛选后的分割结果
    filtered_seg_vis = rgb_img.copy()
    for i, annotation in enumerate(filtered_annotations):
        mask = annotation['segmentation']
        mask_bool = safe_mask_to_bool(mask)
        color = np.random.randint(0, 255, 3).tolist()
        filtered_seg_vis[mask_bool] = filtered_seg_vis[mask_bool] * 0.8 + np.array(color) * 0.2
    axes[0, 1].imshow(filtered_seg_vis)
    axes[0, 1].set_title(f'筛选后结果 ({len(filtered_annotations)}个)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # 3. 筛选统计信息
    axes[1, 0].axis('off')
    stats_text = f"""
    分割筛选统计:
    
    原始分割数量: {len(all_annotations)}
    筛选后数量: {len(filtered_annotations)}
    筛选率: {len(filtered_annotations)/len(all_annotations)*100:.1f}%
    
    筛选条件:
    - 面积范围: 1000-50000
    - 宽高比: 0.5-2.0
    - 深度标准差: < 0.1
    - 实心度: > 0.7
    """
    axes[1, 0].text(0.1, 0.9, stats_text, transform=axes[1, 0].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    # 4. 筛选后的详细信息
    axes[1, 1].axis('off')
    if len(filtered_annotations) > 0:
        detail_text = "筛选后的对象详情:\n\n"
        for i, annotation in enumerate(filtered_annotations):
            detail_text += f"对象 {i+1}:\n"
            detail_text += f"  面积: {annotation['area']:.0f}\n"
            detail_text += f"  宽高比: {annotation['aspect_ratio']:.2f}\n"
            detail_text += f"  深度均值: {annotation['depth_mean']:.3f}\n"
            detail_text += f"  深度标准差: {annotation['depth_std']:.3f}\n"
            detail_text += f"  实心度: {annotation['solidity']:.3f}\n\n"
    else:
        detail_text = "没有对象通过筛选条件"
    
    axes[1, 1].text(0.1, 0.9, detail_text, transform=axes[1, 1].transAxes, 
                    fontsize=9, verticalalignment='top', fontfamily='monospace')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存结果
    process_output_path = output_dir + base_name + "_segmentation_process.png"
    plt.savefig(process_output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return process_output_path


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


def main(args):
    # 加载模型
    model = FastSAM(args.model_path)
    
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
    
    # 筛选立方体分割
    print("\n=== 筛选立方体分割 ===")
    filtered_annotations = filter_cube_segments(annotations, depth_img)
    print(f"筛选后的分割数量: {len(filtered_annotations)}")
    
    # 显示筛选结果
    for i, annotation in enumerate(filtered_annotations):
        print(f"  分割 {i+1}: 面积={annotation['area']:.0f}, "
              f"宽高比={annotation['aspect_ratio']:.2f}, "
              f"深度均值={annotation['depth_mean']:.3f}, "
              f"深度标准差={annotation['depth_std']:.3f}, "
              f"实心度={annotation['solidity']:.3f}")
    
    # 创建分割过程可视化
    if len(filtered_annotations) > 0:
        segmentation_process_path = create_segmentation_process_visualization(
            rgb_img, annotations, filtered_annotations, args.output, args.img_path.split("/")[-1].replace('.jpg', '')
        )
        print(f"分割过程可视化已保存到: {segmentation_process_path}")
    
    # 找到角点
    print("\n=== 角点检测 ===")
    corners_list = []
    corner_vis_list = []
    
    for i, annotation in enumerate(filtered_annotations):
        mask = annotation['segmentation']
        corners, corner_vis = find_cube_corners(mask, depth_img, visualize=True)
        
        if corners is not None:
            print(f"  分割 {i+1} 角点: {corners}")
            corners_list.append(corners)
            if corner_vis is not None:
                corner_vis_list.append(corner_vis)
        else:
            print(f"  分割 {i+1}: 未找到有效角点")
            corners_list.append(None)
    
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
    plt.title(f'立方体检测最终结果 - {base_name}', fontsize=18, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(final_large_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"最终结果大图已保存到: {final_large_path}")
    
    # 显示生成的文件总结
    print("\n=== 生成的文件总结 ===")
    generated_files = [
        comprehensive_path,
        rgb_output_path,
        depth_output_path,
        final_large_path
    ]
    
    if len(filtered_annotations) > 0:
        generated_files.append(segmentation_process_path)
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