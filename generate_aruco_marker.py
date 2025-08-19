#!/usr/bin/env python3
"""
生成ArUco标签并保存为PDF文件
生成DICT_5X5_1000字典中ID为24的ArUco标签
标签尺寸为60mm×60mm，适合A4纸打印
"""

import cv2
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
import os

def generate_aruco_marker(dict_type=cv2.aruco.DICT_5X5_1000, marker_id=24, marker_size_pixels=600):
    """
    生成ArUco标签图像
    
    Args:
        dict_type: ArUco字典类型
        marker_id: 标签ID
        marker_size_pixels: 标签图像尺寸（像素）
    
    Returns:
        marker_image: 生成的标签图像
    """
    try:
        # 获取ArUco字典
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
        
        # 生成标签图像
        marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size_pixels)
        
        print(f"成功生成ArUco标签:")
        print(f"  字典类型: DICT_5X5_1000")
        print(f"  标签ID: {marker_id}")
        print(f"  图像尺寸: {marker_size_pixels}×{marker_size_pixels} 像素")
        
        return marker_image
        
    except Exception as e:
        print(f"生成ArUco标签失败: {e}")
        return None

def create_pdf_with_marker(marker_image, output_pdf_path, marker_size_mm=60):
    """
    创建包含ArUco标签的PDF文件
    
    Args:
        marker_image: ArUco标签图像
        output_pdf_path: 输出PDF文件路径
        marker_size_mm: 标签在PDF中的尺寸（毫米）
    """
    try:
        # A4纸尺寸（毫米）
        a4_width_mm, a4_height_mm = 210, 297
        
        # 创建PDF画布
        c = canvas.Canvas(output_pdf_path, pagesize=A4)
        
        # 计算标签在A4纸上的位置（居中）
        marker_x = (a4_width_mm - marker_size_mm) / 2
        marker_y = (a4_height_mm - marker_size_mm) / 2
        
        # 将OpenCV图像转换为PIL图像格式
        # OpenCV使用BGR，需要转换为RGB
        marker_rgb = cv2.cvtColor(marker_image, cv2.COLOR_GRAY2RGB)
        
        # 保存临时图像文件
        temp_image_path = "temp_marker.png"
        cv2.imwrite(temp_image_path, marker_image)
        
        # 在PDF中绘制图像
        c.drawImage(temp_image_path, marker_x*mm, marker_y*mm, 
                   width=marker_size_mm*mm, height=marker_size_mm*mm)
        
        # 添加标题和说明
        title_y = a4_height_mm - 20
        c.setFont("Helvetica-Bold", 16)
        c.drawString(20*mm, title_y*mm, "ArUco Marker - DICT_5X5_1000")
        
        # 添加标签信息
        info_y = title_y - 15
        c.setFont("Helvetica", 12)
        c.drawString(20*mm, info_y*mm, f"Marker ID: 24")
        c.drawString(20*mm, (info_y-8)*mm, f"Marker Size: {marker_size_mm}mm × {marker_size_mm}mm")
        c.drawString(20*mm, (info_y-16)*mm, f"Dictionary: DICT_5X5_1000")
        
        # 添加使用说明
        usage_y = 30
        c.setFont("Helvetica", 10)
        c.drawString(20*mm, usage_y*mm, "使用说明:")
        c.drawString(20*mm, (usage_y-6)*mm, "1. 使用A4纸打印此PDF文件")
        c.drawString(20*mm, (usage_y-12)*mm, "2. 确保打印时不要缩放，保持100%比例")
        c.drawString(20*mm, (usage_y-18)*mm, "3. 标签尺寸将精确为60mm × 60mm")
        c.drawString(20*mm, (usage_y-24)*mm, "4. 可用于相机标定、位姿估计等应用")
        
        # 保存PDF
        c.save()
        
        # 删除临时图像文件
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        
        print(f"PDF文件已生成:")
        print(f"  文件路径: {output_pdf_path}")
        print(f"  标签尺寸: {marker_size_mm}mm × {marker_size_mm}mm")
        print(f"  标签位置: A4纸中央")
        
        return True
        
    except Exception as e:
        print(f"创建PDF文件失败: {e}")
        return False

def main():
    
    
    # 参数设置
    dict_type = cv2.aruco.DICT_5X5_1000
    marker_id = 12
    marker_size_pixels = 600  # 高分辨率图像
    marker_size_mm = 30       # PDF中的尺寸
    output_pdf_path = f"aruco_marker_id{marker_id}.pdf"

    """主函数"""
    print("=== ArUco标签生成器 ===")
    print(f"生成DICT_5X5_1000字典中ID为{marker_id}的ArUco标签")
    print("标签尺寸: 60mm × 60mm")
    print()
    
    # 生成ArUco标签
    print("正在生成ArUco标签...")
    marker_image = generate_aruco_marker(dict_type, marker_id, marker_size_pixels)
    
    if marker_image is None:
        print("标签生成失败，程序退出")
        return
    
    # 保存标签图像（可选）
    cv2.imwrite("aruco_marker_id24.png", marker_image)
    print("标签图像已保存为: aruco_marker_id24.png")
    
    # 创建PDF文件
    print("\n正在创建PDF文件...")
    success = create_pdf_with_marker(marker_image, output_pdf_path, marker_size_mm)
    
    if success:
        print("\n=== 生成完成 ===")
        print(f"PDF文件: {output_pdf_path}")
        print(f"请使用A4纸打印此PDF文件，标签尺寸将为{marker_size_mm}mm × {marker_size_mm}mm")
    else:
        print("PDF文件生成失败")

if __name__ == "__main__":
    main() 