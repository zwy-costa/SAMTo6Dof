import argparse
import json
import math
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np


def load_intrinsics(json_path: str) -> Tuple[np.ndarray, np.ndarray]:
	with open(json_path, 'r') as f:
		data = json.load(f)
	intr = data['intrinsic']
	camera_matrix = np.array([
		[intr['fx'], 0.0, intr['ppx']],
		[0.0, intr['fy'], intr['ppy']],
		[0.0, 0.0, 1.0]
	], dtype=np.float32)
	# OpenCV plumb bob order: [k1, k2, p1, p2, k3]
	k1 = intr.get('k1', 0.0)
	k2 = intr.get('k2', 0.0)
	p1 = intr.get('p1', 0.0)
	p2 = intr.get('p2', 0.0)
	k3 = intr.get('k3', 0.0)
	dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32).reshape(1, 5)
	return camera_matrix, dist_coeffs


def get_marker_grid_coords(marker_id: int) -> Tuple[int, int]:
	marker_coordinates = {
		0: (0, 0), 1: (0, 2), 2: (0, 4), 3: (0, 6),
		4: (1, 1), 5: (1, 3), 6: (1, 5),
		7: (2, 0), 8: (2, 2), 9: (2, 4), 10: (2, 6),
		11: (3, 1), 12: (3, 3), 13: (3, 5),
		14: (4, 0), 15: (4, 2), 16: (4, 4), 17: (4, 6),
		18: (5, 1), 19: (5, 3), 20: (5, 5),
		21: (6, 0), 22: (6, 2), 23: (6, 4), 24: (6, 6),
		25: (7, 1), 26: (7, 3), 27: (7, 5),
		28: (8, 0), 29: (8, 2), 30: (8, 4), 31: (8, 6),
		32: (9, 1), 33: (9, 3), 34: (9, 5)
	}
	if marker_id in marker_coordinates:
		return marker_coordinates[marker_id]
	# Fallback
	return marker_id // 4, marker_id % 4


def build_board_marker_corners_3d(marker_id: int, square_length_mm: float, marker_length_mm: float) -> np.ndarray:
	row, col = get_marker_grid_coords(marker_id)
	# Centers are at (col + 0.5) and (row + 0.5) squares; col/row already skip black squares via mapping
	cx = (col + 0.5) * square_length_mm  # 毫米单位
	cy = (row + 0.5) * square_length_mm  # 毫米单位
	h = marker_length_mm / 2.0  # 毫米单位
	# Marker corner order should match cv2.aruco: typically clockwise starting at top-left
	corners = np.array([
		[cx - h, cy - h, 0.0],
		[cx + h, cy - h, 0.0],
		[cx + h, cy + h, 0.0],
		[cx - h, cy + h, 0.0]
	], dtype=np.float32)
	return corners


def detect_markers(image: np.ndarray, dict_name: str = 'DICT_4X4_50') -> Tuple[List[np.ndarray], np.ndarray]:
	d = getattr(cv2.aruco, dict_name)
	aruco_dict = cv2.aruco.getPredefinedDictionary(d)
	detector = cv2.aruco.ArucoDetector(aruco_dict)
	corners, ids, _ = detector.detectMarkers(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
	return corners, ids


def estimate_board_pose_all_corners(image: np.ndarray,
		corners: List[np.ndarray],
		ids: np.ndarray,
		square_length_mm: float,
		marker_length_mm: float,
		camera_matrix: np.ndarray,
		dist_coeffs: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray, List[int], np.ndarray, np.ndarray]:
	object_points = []
	image_points = []
	used_ids: List[int] = []
	if ids is None or len(ids) == 0:
		return False, None, None, used_ids, None, None
	for i, marker_id in enumerate(ids.flatten()):
		corners_2d = corners[i].reshape(4, 2).astype(np.float32)
		try:
			corners_3d = build_board_marker_corners_3d(marker_id, square_length_mm, marker_length_mm)
			object_points.append(corners_3d)
			image_points.append(corners_2d)
			used_ids.append(int(marker_id))
		except Exception:
			continue
	if len(object_points) < 4:
		return False, None, None, used_ids, None, None
	object_points = np.concatenate(object_points, axis=0)  # (N*4, 3)
	image_points = np.concatenate(image_points, axis=0)    # (N*4, 2)
	# RANSAC to remove outliers
	ok, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image_points, camera_matrix, dist_coeffs,
		flags=cv2.SOLVEPNP_ITERATIVE, iterationsCount=200, reprojectionError=2.5, confidence=0.99)
	print(f"RANSAC ok: {ok}, inliers: {len(inliers)}")
	if not ok or inliers is None or len(inliers) < 8:
		# fallback to iterative
		ok2, rvec2, tvec2 = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
		return ok2, rvec2, tvec2, used_ids, object_points, image_points
	# refine on inliers
	inl = inliers.reshape(-1)
	obj_inl = object_points[inl]
	img_inl = image_points[inl]
	try:
		rvec, tvec = cv2.solvePnPRefineLM(obj_inl, img_inl, camera_matrix, dist_coeffs, rvec, tvec)
	except Exception:
		pass
	return True, rvec, tvec, used_ids, object_points, image_points


def estimate_board_pose_centers(image: np.ndarray,
		corners: List[np.ndarray],
		ids: np.ndarray,
		square_length_mm: float,
		marker_length_mm: float,
		camera_matrix: np.ndarray,
		dist_coeffs: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray, List[int], np.ndarray, np.ndarray]:
	object_points = []
	image_points = []
	used_ids: List[int] = []
	if ids is None or len(ids) == 0:
		return False, None, None, used_ids, None, None
	for i, marker_id in enumerate(ids.flatten()):
		corners_2d = corners[i].reshape(4, 2).astype(np.float32)
		try:
			center_2d = corners_2d.mean(axis=0)
			row, col = get_marker_grid_coords(marker_id)
			center_3d = np.array([
				(col + 0.5) * square_length_mm,
				(row + 0.5) * square_length_mm,
				0.0
			], dtype=np.float32)
			object_points.append(center_3d)
			image_points.append(center_2d)
			used_ids.append(int(marker_id))
		except Exception:
			continue
	if len(object_points) < 4:
		return False, None, None, used_ids, None, None
	object_points = np.array(object_points)
	image_points = np.array(image_points)
	ok, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image_points, camera_matrix, dist_coeffs,
		flags=cv2.SOLVEPNP_ITERATIVE, iterationsCount=200, reprojectionError=2.5, confidence=0.99)
	print(f"RANSAC ok: {ok}, inliers: {len(inliers)}")
	if not ok or inliers is None or len(inliers) < 4:
		ok2, rvec2, tvec2 = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
		return ok2, rvec2, tvec2, used_ids, object_points, image_points
	# refine
	inl = inliers.reshape(-1)
	obj_inl = object_points[inl]
	img_inl = image_points[inl]
	try:
		rvec, tvec = cv2.solvePnPRefineLM(obj_inl, img_inl, camera_matrix, dist_coeffs, rvec, tvec)
	except Exception:
		pass
	return True, rvec, tvec, used_ids, object_points, image_points


def reproject_points(points_3d: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> np.ndarray:
	proj, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix, dist_coeffs)
	return proj.reshape(-1, 2)


def draw_overlay(image: np.ndarray,
		corners: List[np.ndarray],
		ids: np.ndarray,
		used_ids: List[int],
		reproj_centers: Dict[int, np.ndarray],
		reproj_corners: Dict[int, np.ndarray],
		marker_centers_2d: Dict[int, np.ndarray],
		out_path: str,
		marker_analysis: Dict[int, Dict] = None) -> np.ndarray:
	# 放大系数
	scale_factor = 2.0
	# 放大底图
	h, w = image.shape[:2]
	vis = cv2.resize(image.copy(), (int(w * scale_factor), int(h * scale_factor)))
	# 放大角点坐标以便绘制
	scaled_corners: List[np.ndarray] = []
	if corners is not None:
		for c in corners:
			# c 形状通常为 (1,4,2)，直接按系数缩放
			scaled_corners.append(c * scale_factor)
	# 绘制检测到的标记（绿色）
	if ids is not None and len(ids) > 0 and len(scaled_corners) == len(corners):
		cv2.aruco.drawDetectedMarkers(vis, scaled_corners, ids)
	# 绘制重投影（红色）及误差（黄线）
	for mid in used_ids:
		if mid in reproj_corners:
			poly = (reproj_corners[mid] * scale_factor).astype(int)
			cv2.polylines(vis, [poly], True, (0, 0, 255), 2)
			for p in poly:
				px, py = int(p[0]), int(p[1])
				# 确保坐标在图像范围内
				if 0 <= px < vis.shape[1] and 0 <= py < vis.shape[0]:
					cv2.circle(vis, (px, py), 3, (0, 0, 255), -1)
		if mid in reproj_centers:
			pc_pt = (reproj_centers[mid] * scale_factor).astype(int)
			pc = (int(pc_pt[0]), int(pc_pt[1]))
			# 确保坐标在图像范围内
			if 0 <= pc[0] < vis.shape[1] and 0 <= pc[1] < vis.shape[0]:
				cv2.circle(vis, pc, 4, (0, 0, 255), -1)
				# 绘制误差线
				if mid in marker_centers_2d:
					dc_pt = (marker_centers_2d[mid] * scale_factor).astype(int)
					dc = (int(dc_pt[0]), int(dc_pt[1]))
					cv2.line(vis, dc, pc, (0, 255, 255), 2)
					err = float(np.linalg.norm(marker_centers_2d[mid] - reproj_centers[mid]))
					label = f"{mid}-{err:.1f}"
					cv2.putText(vis, label, (pc[0] + 8, pc[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
	# 绘制每个标记的局部坐标轴与差值（如果提供）
	if marker_analysis:
		axis_len_px = 40
		for mid, data in marker_analysis.items():
			center_2d = np.array(data.get('center_2d', [0, 0]), dtype=np.float32)
			x_axis_2d = np.array(data.get('x_axis_2d', [1, 0]), dtype=np.float32)
			y_axis_2d = np.array(data.get('y_axis_2d', [0, 1]), dtype=np.float32)
			trans_diff = np.array(data.get('translation_difference', [0, 0, 0]), dtype=np.float32).reshape(-1)
			c = (center_2d * scale_factor).astype(int)
			c_pt = (int(c[0]), int(c[1]))
			# 坐标轴终点
			x_end = (center_2d + x_axis_2d * axis_len_px) * scale_factor
			y_end = (center_2d + y_axis_2d * axis_len_px) * scale_factor
			x_end = (int(x_end[0]), int(x_end[1]))
			y_end = (int(y_end[0]), int(y_end[1]))
			# 画轴
			cv2.line(vis, c_pt, x_end, (0, 0, 255), 2)  # X 轴红色
			cv2.line(vis, c_pt, y_end, (0, 255, 0), 2)  # Y 轴绿色
			cv2.putText(vis, 'x', (x_end[0] + 3, x_end[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
			cv2.putText(vis, 'y', (y_end[0] + 3, y_end[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
			# 差值文本
			lbl = f"d=({trans_diff[0]:.0f},{trans_diff[1]:.0f},{trans_diff[2]:.0f})mm"
			cv2.putText(vis, lbl, (c_pt[0] + 6, c_pt[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
	# 保存与返回
	cv2.imwrite(out_path, vis)
	return vis


def _mean_reproj_error(objp: np.ndarray, imgp: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, K: np.ndarray, D: np.ndarray) -> float:
	proj = reproject_points(objp, rvec, tvec, K, D)
	return float(np.mean(np.linalg.norm(imgp - proj, axis=1)))


def _cheirality_ok(objp: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> bool:
	R, _ = cv2.Rodrigues(rvec)
	Pc = (R @ objp.T + tvec.reshape(3, 1)).T
	return np.all(Pc[:, 2] > 0)


def _angle_deg_between_rvecs(rvec_a: np.ndarray, rvec_b: np.ndarray) -> float:
	Ra, _ = cv2.Rodrigues(rvec_a)
	Rb, _ = cv2.Rodrigues(rvec_b)
	Rrel = Ra.T @ Rb
	ang = float(np.degrees(np.arccos(np.clip((np.trace(Rrel) - 1.0) / 2.0, -1.0, 1.0))))
	return ang


def _normal_angle_deg(rvec_a: np.ndarray, rvec_b: np.ndarray) -> float:
	Ra, _ = cv2.Rodrigues(rvec_a)
	Rb, _ = cv2.Rodrigues(rvec_b)
	Rrel = Ra.T @ Rb
	ang = float(np.degrees(np.arccos(np.clip((np.trace(Rrel) - 1.0) / 2.0, -1.0, 1.0))))
	return ang


def _normal_angle_deg_unoriented(rvec_a: np.ndarray, rvec_b: np.ndarray) -> float:
	ang = _normal_angle_deg(rvec_a, rvec_b)
	return float(min(ang, 180.0 - ang))

def _rotation_matrix_to_euler_angles(rvec_a: np.ndarray, rvec_b: np.ndarray) -> np.ndarray:
	Ra, _ = cv2.Rodrigues(rvec_a)
	Rb, _ = cv2.Rodrigues(rvec_b)
	Rrel = Ra.T @ Rb
	# 将旋转矩阵转换为欧拉角 (ZYX顺序)
	sy = np.sqrt(Rrel[0, 0] * Rrel[0, 0] + Rrel[1, 0] * Rrel[1, 0])
	singular = sy < 1e-6
	if not singular:
		x = np.arctan2(Rrel[2, 1], Rrel[2, 2])
		y = np.arctan2(-Rrel[2, 0], sy)
		z = np.arctan2(Rrel[1, 0], Rrel[0, 0])
	else:
		x = np.arctan2(-Rrel[1, 2], Rrel[1, 1])
		y = np.arctan2(-Rrel[2, 0], sy)
		z = 0.0
	# 转换为度
	euler_angles = np.array([x, y, z]) * 180.0 / np.pi
	# euler_angles_error = float(np.mean(np.abs(euler_angles), axis=0))
	return euler_angles


def _solve_candidates_ippe(objp: np.ndarray, imgp: np.ndarray, K: np.ndarray, D: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
	flag = getattr(cv2, 'SOLVEPNP_IPPE_SQUARE', cv2.SOLVEPNP_ITERATIVE)
	# try:
	# 	retval, rvecs, tvecs, reproj = cv2.solvePnPGeneric(objp, imgp, K, D, flags=flag)
	# 	cands = []
	# 	for rv, tv in zip(rvecs, tvecs):
	# 		cands.append((rv, tv))
	# 	return cands
	# except Exception:
	# 	# fallback single solution
	# ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, D, flags=flag)
	# return [(rvec, tvec)] if ok else []
	ok, rvec, tvec, inliers = cv2.solvePnPRansac(objp, imgp, K, D,
		flags=cv2.SOLVEPNP_ITERATIVE, iterationsCount=200, reprojectionError=2.5, confidence=0.99)
	if (not ok) or (inliers is None) or (len(inliers) < 4):
		ok2, rvec2, tvec2 = cv2.solvePnP(objp, imgp, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
		return [(rvec2, tvec2)] if ok2 else []
	# refine on inliers
	try:
		inl = inliers.reshape(-1)
		obj_inl = objp[inl]
		img_inl = imgp[inl]
		rvec, tvec = cv2.solvePnPRefineLM(obj_inl, img_inl, K, D, rvec, tvec)
	except Exception:
		pass
	return [(rvec, tvec)] if ok else []


def estimate_marker_pose_from_corners(corners_2d: np.ndarray,
		marker_length_mm: float,
		camera_matrix: np.ndarray,
		dist_coeffs: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray, float]:
	# Fixed corner order already ensured at detection time
	h = float(marker_length_mm) / 2.0
	objp_base = np.array([
		[-h, -h, 0.0],
		[ h, -h, 0.0],
		[ h,  h, 0.0],
		[-h,  h, 0.0],
	], dtype=np.float32)
	imgp_full = corners_2d.reshape(4, 2).astype(np.float32)
	cands = _solve_candidates_ippe(objp_base, imgp_full, camera_matrix, dist_coeffs)
	best = (False, None, None, float('inf'))
	best_key = (float('inf'), float('inf'))  # (normal_angle, reproj_err)
	for rvec_c, tvec_c in cands:
		if not _cheirality_ok(objp_base, rvec_c, tvec_c):
			continue
		err = _mean_reproj_error(objp_base, imgp_full, rvec_c, tvec_c, camera_matrix, dist_coeffs)
		if not np.isfinite(err):
			continue
		key = (0.0, err)
		if key < best_key:
			best_key = key
			best = (True, rvec_c, tvec_c, err)
	if best[0]:
		try:
			rref, tref = cv2.solvePnPRefineLM(objp_base, imgp_full, camera_matrix, dist_coeffs, best[1], best[2])
			err_ref = _mean_reproj_error(objp_base, imgp_full, rref, tref, camera_matrix, dist_coeffs)
			return True, rref, tref, float(err_ref)
		except Exception:
			return best
	return best


def estimate_all_markers_poses_from_corners(corners: List[np.ndarray],
		ids: np.ndarray,
		marker_length_mm: float,
		camera_matrix: np.ndarray,
		dist_coeffs: np.ndarray) -> Dict[int, Dict[str, np.ndarray]]:
	"""
	Compute per-marker pose from its own 4 corners only, using board-translated pose as prior if provided.
	Returns dict[id] -> {'rvec':..., 'tvec':..., 'reproj_err_px':...}
	"""
	results: Dict[int, Dict[str, np.ndarray]] = {}
	if ids is None or len(ids) == 0:
		return results
	for i, marker_id in enumerate(ids.flatten().tolist()):
		c2d = corners[i].reshape(4, 2).astype(np.float32)
		ok, rvec, tvec, err = estimate_marker_pose_from_corners(c2d, marker_length_mm, camera_matrix, dist_coeffs)
		if not ok:
			continue
		results[int(marker_id)] = {'rvec': rvec, 'tvec': tvec, 'reproj_err_px': float(err)}
	return results


def compute_board_translated_pose_for_marker(marker_id: int,
		board_rvec: np.ndarray,
		board_tvec: np.ndarray,
		square_length_mm: float) -> Tuple[np.ndarray, np.ndarray]:
	row, col = get_marker_grid_coords(marker_id)
	center_3d = np.array([(col + 0.5) * square_length_mm, (row + 0.5) * square_length_mm, 0.0], dtype=np.float32)
	R_board, _ = cv2.Rodrigues(board_rvec)
	board_tvec_reshaped = board_tvec.reshape(3, 1)
	offset = (R_board @ center_3d.reshape(3, 1))
	new_tvec = board_tvec_reshaped + offset
	new_rvec = board_rvec.copy()
	return new_rvec.reshape(3, 1), new_tvec.reshape(3, 1)


def rotation_angle_between(rvec_a: np.ndarray, rvec_b: np.ndarray) -> float:
	Ra, _ = cv2.Rodrigues(rvec_a)
	Rb, _ = cv2.Rodrigues(rvec_b)
	Rrel = Ra.T @ Rb
	angle = float(np.degrees(np.arccos(np.clip((np.trace(Rrel) - 1.0) / 2.0, -1.0, 1.0))))
	return angle


def _to_serializable(obj):
	if isinstance(obj, np.ndarray):
		return obj.tolist()
	if isinstance(obj, (np.integer,)):
		return int(obj)
	if isinstance(obj, (np.floating,)):
		return float(obj)
	if isinstance(obj, dict):
		return {k: _to_serializable(v) for k, v in obj.items()}
	if isinstance(obj, (list, tuple)):
		return [_to_serializable(v) for v in obj]
	return obj


def _compose_rvecs(rvec_a: np.ndarray, rvec_b: np.ndarray) -> np.ndarray:
	Ra, _ = cv2.Rodrigues(rvec_a)
	Rb, _ = cv2.Rodrigues(rvec_b)
	R = Ra @ Rb
	rv, _ = cv2.Rodrigues(R)
	return rv


def _mean_corner_err_for_theta(theta_deg: float,
		marker_id: int,
		board_rvec: np.ndarray,
		board_tvec: np.ndarray,
		square_length_mm: float,
		marker_length_mm: float,
		corners_2d: np.ndarray,
		camera_matrix: np.ndarray,
		dist_coeffs: np.ndarray) -> float:
	# build 3D corners in board frame
	corners_3d = build_board_marker_corners_3d(marker_id, square_length_mm, marker_length_mm).astype(np.float32)
	center = np.mean(corners_3d, axis=0, keepdims=True)
	# rotate in-plane around center in board XY
	theta = math.radians(theta_deg)
	cos_t, sin_t = math.cos(theta), math.sin(theta)
	rot2 = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
	xy = corners_3d[:, :2] - center[0, :2]
	xy_rot = (rot2 @ xy.T).T + center[0, :2]
	corners_3d_rot = np.hstack([xy_rot, corners_3d[:, 2:3]])
	proj = reproject_points(corners_3d_rot.astype(np.float32), board_rvec, board_tvec, camera_matrix, dist_coeffs)
	return float(np.mean(np.linalg.norm(proj - corners_2d.reshape(4, 2).astype(np.float32), axis=1)))


def compute_board_translated_pose_for_marker_oriented(marker_id: int,
		board_rvec: np.ndarray,
		board_tvec: np.ndarray,
		square_length_mm: float,
		marker_length_mm: float,
		corners_2d: np.ndarray,
		camera_matrix: np.ndarray,
		dist_coeffs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
	"""
	Translate board pose to the marker center and adjust orientation by selecting the best in-plane rotation (0,90,180,270)
	that minimizes corner reprojection error.
	Returns rvec_adj, tvec_new, chosen_theta_deg
	"""
	# translation same as original
	rvec_bm, tvec_bm = compute_board_translated_pose_for_marker(marker_id, board_rvec, board_tvec, square_length_mm)
	# search theta
	cands = [0.0, 90.0, 180.0, 270.0]
	best_theta = 0.0
	best_err = float('inf')
	for th in cands:
		err = _mean_corner_err_for_theta(th, marker_id, board_rvec, board_tvec, square_length_mm, marker_length_mm, corners_2d, camera_matrix, dist_coeffs)
		if err < best_err:
			best_err = err
			best_theta = th
	# compose orientation: R_board * Rz(theta)
	rvec_z = np.array([[0.0], [0.0], [math.radians(best_theta)]], dtype=np.float32)
	rvec_adj = _compose_rvecs(board_rvec.reshape(3, 1), rvec_z)
	return rvec_adj.reshape(3, 1), tvec_bm.reshape(3, 1), float(best_theta)


def main():
	parser = argparse.ArgumentParser(description='Estimate Charuco-like board pose and reproject markers')
	parser.add_argument('--image', required=True, help='Input image path')
	parser.add_argument('--intrinsic', default='head_intrinsic_params.json', help='Camera intrinsic JSON path')
	parser.add_argument('--square-length-mm', type=float, default=30.0, help='Square length in mm (default 30)')
	parser.add_argument('--marker-length-mm', type=float, default=22.0, help='Marker length in mm (default 22)')
	parser.add_argument('--dict', default='DICT_4X4_50', help='ArUco dictionary name (default DICT_4X4_50)')
	parser.add_argument('--out', default='reprojection_overlay.jpg', help='Output overlay image path')
	parser.add_argument('--show', action='store_true', help='Show overlay window after drawing')
	parser.add_argument('--json-out', default=None, help='Optional JSON file to export per-marker analysis results')
	args = parser.parse_args()

	image = cv2.imread(args.image)
	if image is None:
		print(f"❌ Failed to read image: {args.image}")
		return
	camera_matrix, dist_coeffs = load_intrinsics(args.intrinsic)
	# 保持毫米单位，不转换为米
	square_length_mm = args.square_length_mm
	marker_length_mm = args.marker_length_mm

	corners, ids = detect_markers(image, args.dict)
	if ids is None or len(ids) == 0:
		print('❌ No ArUco markers detected')
		return
	print(f"✅ Detected {len(ids)} markers: {ids.flatten().tolist()}")

	# ======= 1. 使用板上的所有角点输入SolvePnP计算RT =======
	# 比较两种方法：角点 vs 中心点
	print("\n=== 角点方法 ===")
	success1, rvec1, tvec1, used_ids1, object_points1, image_points1 = estimate_board_pose_all_corners(
		image, corners, ids, square_length_mm, marker_length_mm, camera_matrix, dist_coeffs
	)
	if success1:
		R1, _ = cv2.Rodrigues(rvec1)
		dist1 = float(np.linalg.norm(tvec1))
		print(f"✅ 角点方法成功 - 距离: {dist1:.1f} mm")
		
		# 计算重投影误差 - 使用标记中心点进行公平比较
		center_errors1 = []
		for i, marker_id in enumerate(used_ids1):
			# 获取检测到的标记中心
			corners_2d = corners[ids.flatten().tolist().index(marker_id)].reshape(4, 2)
			detected_center = corners_2d.mean(axis=0)
			
			# 计算理论中心3D坐标
			row, col = get_marker_grid_coords(marker_id)
			center_3d = np.array([[(col + 0.5) * square_length_mm, (row + 0.5) * square_length_mm, 0.0]], dtype=np.float32)
			
			# 重投影中心点
			reproj_center = reproject_points(center_3d, rvec1, tvec1, camera_matrix, dist_coeffs)[0]
			
			# 计算误差
			error = np.linalg.norm(detected_center - reproj_center)
			center_errors1.append(error)
		
		print(f"重投影误差(中心点): 平均={np.mean(center_errors1):.2f}px, 最大={np.max(center_errors1):.2f}px, 标准差={np.std(center_errors1):.2f}px")
	else:
		print("❌ 角点方法失败")
		return

	print("\n=== 中心点方法 ===")
	success2, rvec2, tvec2, used_ids2, object_points2, image_points2 = estimate_board_pose_centers(
		image, corners, ids, square_length_mm, marker_length_mm, camera_matrix, dist_coeffs
	)
	if success2:
		R2, _ = cv2.Rodrigues(rvec2)
		dist2 = float(np.linalg.norm(tvec2))
		print(f"✅ 中心点方法成功 - 距离: {dist2:.1f} mm")
		
		# 计算重投影误差 - 使用标记中心点（与角点方法相同）
		center_errors2 = []
		for i, marker_id in enumerate(used_ids2):
			# 获取检测到的标记中心
			corners_2d = corners[ids.flatten().tolist().index(marker_id)].reshape(4, 2)
			detected_center = corners_2d.mean(axis=0)
			
			# 计算理论中心3D坐标
			row, col = get_marker_grid_coords(marker_id)
			center_3d = np.array([[(col + 0.5) * square_length_mm, (row + 0.5) * square_length_mm, 0.0]], dtype=np.float32)
			
			# 重投影中心点
			reproj_center = reproject_points(center_3d, rvec2, tvec2, camera_matrix, dist_coeffs)[0]
			
			# 计算误差
			error = np.linalg.norm(detected_center - reproj_center)
			center_errors2.append(error)
		
		print(f"重投影误差(中心点): 平均={np.mean(center_errors2):.2f}px, 最大={np.max(center_errors2):.2f}px, 标准差={np.std(center_errors2):.2f}px")
		
		# 比较两种方法的结果
		dist_diff = abs(dist1 - dist2)
		error_diff = abs(np.mean(center_errors1) - np.mean(center_errors2))
		print(f"\n距离差异: {dist_diff:.1f} mm")
		print(f"重投影误差差异: {error_diff:.2f} px")
	else:
		print("❌ 中心点方法失败")
		return

	# 使用角点方法的结果进行可视化（更准确）
	success, rvec, tvec, used_ids, object_points, image_points = success1, rvec1, tvec1, used_ids1, object_points1, image_points1

	# ======= 2. 使用每个二维码对应的四个角点输入SolvePnP计算每个二维码RT =======
	# 基于四角点的单标记RT（中心为原点）
	print("\n=== 单标记四角点位姿（以该标记中心为原点，相机坐标系下） ===")
	per_marker_rt = estimate_all_markers_poses_from_corners(
		corners, ids, marker_length_mm, camera_matrix, dist_coeffs
	)
	print(f"共 {len(per_marker_rt)} 个标记估计成功")
	for mid in sorted(per_marker_rt.keys()):
		pm = per_marker_rt[mid]
		tvec = pm['tvec'].reshape(3)
		rvec = pm['rvec'].reshape(3)
		re = pm['reproj_err_px']
		print(f"ID {mid:2d}: T=({tvec[0]:7.1f},{tvec[1]:7.1f},{tvec[2]:7.1f}) mm, R=({rvec[0]:.3f},{rvec[1]:.3f},{rvec[2]:.3f}) rad, reproj={re:.2f}px")

	# ======= 3. 每个二维码的RT与 
	# 单标记 RT 与 基于标定板平移后的 RT 做差值
	print("\n=== 单标记RT vs 板平移RT 差值 ===")
	delta_t_list = []
	delta_angle_list = []
	delta_euler_mean_list = []
	for mid in sorted(per_marker_rt.keys()):
		pm = per_marker_rt[mid]
		rvec_m = pm['rvec'].reshape(3, 1)
		tvec_m = pm['tvec'].reshape(3, 1)
		# *_bm 使用所有标定板的角点求出来的RT求某二维码中心在相机坐标系下的位姿, *_m 使用该二维码的四个角点求出来的RT(即该二维码中心在相机坐标系下的位姿)，两个位姿做差对比
		# Compare against pure board-translated pose without using this marker's image corners
		rvec_bm, tvec_bm = compute_board_translated_pose_for_marker(mid, rvec1, tvec1, square_length_mm)
		dt = (tvec_m - tvec_bm).reshape(3)
		delta_t_list.append(dt)
		# use normal angle difference for planar consistency (unoriented)
		dang = _normal_angle_deg_unoriented(rvec_bm, rvec_m)
		delta_angle_list.append(dang)
		
		euler_rel = _rotation_matrix_to_euler_angles(rvec_bm, rvec_m)
		euler_angles_error = float(np.mean(np.abs(euler_rel), axis=0))
		delta_euler_mean_list.append(euler_angles_error)
		print(f"ID {mid:2d}: dT=({dt[0]:7.1f},{dt[1]:7.1f},{dt[2]:7.1f}) mm, dAngle={dang:6.2f} deg, dEuler=({float(euler_rel[0]):6.2f},{float(euler_rel[1]):6.2f},{float(euler_rel[2]):6.2f}) deg")
	if delta_t_list:
		delta_t_arr = np.stack(delta_t_list, axis=0)
		mean_abs = np.mean(np.abs(delta_t_arr), axis=0)
		std_abs = np.std(np.abs(delta_t_arr), axis=0)
		mean_ang = float(np.mean(delta_angle_list))
		std_ang = float(np.std(delta_angle_list))
		print(f"汇总: |dT|均值=({mean_abs[0]:.1f},{mean_abs[1]:.1f},{mean_abs[2]:.1f}) mm, |dT|标准差=({std_abs[0]:.1f},{std_abs[1]:.1f},{std_abs[2]:.1f}) mm")
		print(f"汇总: dAngle 平均={mean_ang:.2f} deg, 标准差={std_ang:.2f} deg")

	# Build per-marker centers 3D and corners 3D for reprojection
	reproj_centers: Dict[int, np.ndarray] = {}
	reproj_corners: Dict[int, np.ndarray] = {}
	marker_centers_2d: Dict[int, np.ndarray] = {}

	# Map id -> detected center and corners
	id_to_corners_idx: Dict[int, int] = {}
	for i, mid in enumerate(ids.flatten().tolist()):
		id_to_corners_idx[mid] = i
	# Reproject for used ids only
	for mid in used_ids:
		corners_3d = build_board_marker_corners_3d(mid, square_length_mm, marker_length_mm)
		center_3d = np.mean(corners_3d, axis=0, keepdims=True)
		proj_corners = reproject_points(corners_3d, rvec, tvec, camera_matrix, dist_coeffs)
		proj_center = reproject_points(center_3d, rvec, tvec, camera_matrix, dist_coeffs)[0]
		reproj_corners[mid] = proj_corners
		reproj_centers[mid] = proj_center
		# Detected center from image
		ci = id_to_corners_idx.get(mid, None)
		if ci is not None:
			det_center = np.mean(corners[ci].reshape(4, 2), axis=0)
			marker_centers_2d[mid] = det_center

	# Compute final reprojection errors reusing center_errors1 for consistency
	if 'center_errors1' in locals() and center_errors1:
		print(f"Reprojection error: mean={np.mean(center_errors1):.2f}px, max={np.max(center_errors1):.2f}px, std={np.std(center_errors1):.2f}px")

	# 导出 JSON（如需且有分析结果）
	marker_analysis_local = locals().get('marker_analysis', None)
	print(f"Debug: marker_analysis_local = {marker_analysis_local}")
	if args.json_out and marker_analysis_local:
		export_data = {}
		for mid, data in marker_analysis_local.items():
			# 扁平化 rvec/tvec
			bp = data.get('board_pose_at_marker', {})
			exp = dict(data)
			if 'board_pose_at_marker' in exp:
				exp['board_pose_at_marker'] = {
					'rvec': _to_serializable(np.asarray(bp.get('rvec')).reshape(-1)),
					'tvec': _to_serializable(np.asarray(bp.get('tvec')).reshape(-1)),
				}
			export_data[int(mid)] = _to_serializable(exp)
		with open(args.json_out, 'w') as f:
			json.dump(export_data, f, ensure_ascii=False, indent=2)
		print(f"✅ 分析结果已导出: {args.json_out}")
	else:
		print(f"⚠️  JSON导出跳过: args.json_out={args.json_out}, marker_analysis_local={marker_analysis_local}")

	vis = draw_overlay(image, corners, ids, used_ids, reproj_centers, reproj_corners, marker_centers_2d, args.out, marker_analysis_local)
	print(f"✅ Overlay saved to: {args.out}")
	if args.show:
		cv2.imshow('overlay', cv2.imread(args.out))
		cv2.waitKey(0)
		cv2.destroyAllWindows()


if __name__ == '__main__':
	main() 