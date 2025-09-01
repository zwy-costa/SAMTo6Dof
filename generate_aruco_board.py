#!/usr/bin/env python3
"""
生成黑白相间棋盘图案并在指定位置放置ArUco二维码，输出为PDF
- 中心位置放置给定ID的二维码
- 至少包含5张二维码：中心 + 左上/右上/左下/右下 四个对角位置
- 其他格子按黑白相间绘制
"""

import cv2
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
import argparse
import os


def get_aruco_dict(dict_name_or_code):
    if isinstance(dict_name_or_code, int):
        return cv2.aruco.getPredefinedDictionary(dict_name_or_code)
    name = str(dict_name_or_code).upper()
    # 常用映射
    mapping = {
        'DICT_4X4_50': cv2.aruco.DICT_4X4_50,
        'DICT_4X4_100': cv2.aruco.DICT_4X4_100,
        'DICT_4X4_250': cv2.aruco.DICT_4X4_250,
        'DICT_4X4_1000': cv2.aruco.DICT_4X4_1000,
        'DICT_5X5_50': cv2.aruco.DICT_5X5_50,
        'DICT_5X5_100': cv2.aruco.DICT_5X5_100,
        'DICT_5X5_250': cv2.aruco.DICT_5X5_250,
        'DICT_5X5_1000': cv2.aruco.DICT_5X5_1000,
        'DICT_6X6_50': cv2.aruco.DICT_6X6_50,
        'DICT_6X6_100': cv2.aruco.DICT_6X6_100,
        'DICT_6X6_250': cv2.aruco.DICT_6X6_250,
        'DICT_6X6_1000': cv2.aruco.DICT_6X6_1000,
        'DICT_7X7_50': cv2.aruco.DICT_7X7_50,
        'DICT_7X7_100': cv2.aruco.DICT_7X7_100,
        'DICT_7X7_250': cv2.aruco.DICT_7X7_250,
        'DICT_7X7_1000': cv2.aruco.DICT_7X7_1000,
        'ARUCO_ORIGINAL': cv2.aruco.DICT_ARUCO_ORIGINAL,
    }
    code = mapping.get(name, cv2.aruco.DICT_5X5_1000)
    return cv2.aruco.getPredefinedDictionary(code)


def generate_marker_image(aruco_dict, marker_id: int, size_pixels: int = 600) -> np.ndarray:
    img = cv2.aruco.generateImageMarker(aruco_dict, int(marker_id), int(size_pixels))
    return img


def draw_board_pdf(output_pdf_path: str,
                   board_rows: int,
                   board_cols: int,
                   square_mm: float,
                   marker_mm: float,
                   center_id: int,
                   dict_name: str = 'DICT_4X4_1000',
                   page_size=A4,
                   margin_mm: float = 10.0):
    # 基础校验：要求行列为奇数，确保唯一中心
    if board_rows % 2 == 0 or board_cols % 2 == 0:
        raise ValueError('board_rows 与 board_cols 均需为奇数，确保存在唯一中心格。')
    if marker_mm > square_mm:
        raise ValueError('marker_mm 不能大于 square_mm。')

    aruco_dict = get_aruco_dict(dict_name)

    # 页面尺寸
    page_w_mm, page_h_mm = page_size[0] / mm, page_size[1] / mm
    board_w_mm = board_cols * square_mm
    board_h_mm = board_rows * square_mm

    # 居中起点
    origin_x_mm = (page_w_mm - board_w_mm) / 2.0
    origin_y_mm = (page_h_mm - board_h_mm) / 2.0

    # 创建PDF
    c = canvas.Canvas(output_pdf_path, pagesize=page_size)

    # 绘制棋盘黑白格
    for r in range(board_rows):
        for col in range(board_cols):
            x_mm = origin_x_mm + col * square_mm
            y_mm = origin_y_mm + (board_rows - 1 - r) * square_mm  # PDF坐标y向上
            # 黑白相间：左上(0,0)为白；(r+c)奇偶决定
            if (r + col) % 2 == 1:
                c.setFillGray(0.0)  # 黑色
                c.rect(x_mm * mm, y_mm * mm, square_mm * mm, square_mm * mm, stroke=0, fill=1)
            else:
                c.setFillGray(1.0)  # 白色
                c.rect(x_mm * mm, y_mm * mm, square_mm * mm, square_mm * mm, stroke=0, fill=1)

    # 需要放置的5个id：中心 + 四个对角
    center_r, center_c = board_rows // 2, board_cols // 2
    placements = [(center_r, center_c, center_id)]
    neighbor_ids = [center_id - 4, center_id - 3, center_id + 3, center_id + 4]
    diag_rc = [(center_r - 1, center_c - 1), (center_r - 1, center_c + 1),
               (center_r + 1, center_c - 1), (center_r + 1, center_c + 1)]
    for (rr, cc), mid in zip(diag_rc, neighbor_ids):
        if 0 <= rr < board_rows and 0 <= cc < board_cols:
            placements.append((rr, cc, mid))

    # 生成并放置二维码（以方格中心为对齐基准）
    temp_files = []
    try:
        for (rr, cc, mid) in placements:
            # 生成Marker图像（较高分辨率以便打印）
            marker_px = 800
            m_img = generate_marker_image(aruco_dict, mid, marker_px)
            # 保存临时文件
            tmp_name = f"_tmp_marker_{mid}.png"
            cv2.imwrite(tmp_name, m_img)
            temp_files.append(tmp_name)

            # 计算放置位置
            cell_x_mm = origin_x_mm + cc * square_mm
            cell_y_mm = origin_y_mm + (board_rows - 1 - rr) * square_mm
            # 居中放置marker
            draw_x_mm = cell_x_mm + (square_mm - marker_mm) / 2.0
            draw_y_mm = cell_y_mm + (square_mm - marker_mm) / 2.0
            # 在PDF绘制图像（ReportLab以mm为单位）
            c.drawImage(tmp_name, draw_x_mm * mm, draw_y_mm * mm,
                        width=marker_mm * mm, height=marker_mm * mm, preserveAspectRatio=True, mask='auto')

        # 文本说明
        c.setFillGray(0.0)
        c.setFont('Helvetica', 10)
        c.drawString(10 * mm, (page_h_mm - 10) * mm,
                     f"ArUco Board: {dict_name}, center_id={center_id}, size={board_cols}x{board_rows}, square={square_mm}mm, marker={marker_mm}mm")
        c.save()
    finally:
        for f in temp_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description='Generate chessboard-like ArUco board PDF with center marker and four diagonal markers')
    parser.add_argument('--dict', default='DICT_4X4_1000', help='ArUco dictionary name, e.g., DICT_5X5_1000')
    parser.add_argument('--center-id', type=int, default=8, help='Center ArUco ID')
    parser.add_argument('--rows', type=int, default=3, help='Board rows (odd)')
    parser.add_argument('--cols', type=int, default=3, help='Board cols (odd)')
    parser.add_argument('--square-mm', type=float, default=20.0, help='Square size in mm')
    parser.add_argument('--marker-mm', type=float, default=15.0, help='Marker size in mm (<= square-mm)')
    parser.add_argument('--output', default='aruco_board.pdf', help='Output PDF path')
    args = parser.parse_args()

    draw_board_pdf(args.output, args.rows, args.cols, args.square_mm, args.marker_mm, args.center_id, args.dict)
    print(f"✅ 生成完成: {args.output}")


if __name__ == '__main__':
    main() 