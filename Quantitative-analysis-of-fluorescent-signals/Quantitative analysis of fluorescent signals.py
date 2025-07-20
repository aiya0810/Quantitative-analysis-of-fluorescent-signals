import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_ct_lines(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("图像读取失败")

    height, width = img.shape[:2]
    mid_x = width // 2

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # CLAHE预处理
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    # ========== T线检测（左半区）==========
    lower_purple = np.array([120, 40, 40])
    upper_purple = np.array([160, 255, 255])
    mask_purple_t = cv2.inRange(hsv, lower_purple, upper_purple)
    mask_purple_t[:, mid_x:] = 0

    black_lab_t = cv2.inRange(lab, np.array([0, 85, 85]), np.array([65, 175, 175]))
    black_hsv_t = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 80, 150]))
    mask_black_t = cv2.bitwise_or(black_lab_t, black_hsv_t)
    mask_black_t[:, mid_x:] = 0

    mask_t = cv2.bitwise_or(mask_purple_t, mask_black_t)
    mask_t = cv2.medianBlur(mask_t, 7)

    # ========== C线检测（右半区）==========
    black_lab_c = cv2.inRange(lab, np.array([0, 85, 85]), np.array([65, 175, 175]))
    black_hsv_c = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 80, 150]))
    mask_black_c = cv2.bitwise_or(black_lab_c, black_hsv_c)
    mask_black_c[:, :mid_x] = 0

    mask_purple_c = cv2.inRange(hsv, lower_purple, upper_purple)
    mask_purple_c[:, :mid_x] = 0

    mask_c = cv2.bitwise_or(mask_black_c, mask_purple_c)
    mask_c = cv2.medianBlur(mask_c, 7)

    # ========== 定位算法==========
    def find_search_start(mask):
        horizontal_proj = np.sum(mask, axis=0)
        smooth_proj = np.convolve(horizontal_proj, np.ones(7)/7, mode='same')
        return np.argmax(smooth_proj[int(width*0.3):int(width*0.7)]) + int(width*0.3)

    c_search_start = find_search_start(mask_c)

    def find_dense_region(mask, search_step=2):
        density = [np.sum(mask[:, x:x + search_step]) for x in range(0, mask.shape[1], search_step)]
        density = np.array(density, dtype=np.float32)
        density_2d = density.reshape(1, -1)
        density_smooth = cv2.GaussianBlur(density_2d, (5, 1), 0)
        density_smooth = density_smooth.flatten()
        peak_region = np.where(density_smooth > np.max(density_smooth) * 0.7)[0]
        return int(np.mean(peak_region)) * search_step if len(peak_region) > 0 else 0

    t_pos = find_dense_region(mask_t[:, :mid_x], 3)
    c_offset = find_dense_region(mask_c[:, max(c_search_start, mid_x):], 1)
    c_pos = min(max(c_search_start, mid_x) + c_offset, width-1)

    return t_pos, c_pos, mask_t, mask_c

def calculate_tc_ratio(img, t_pos, c_pos):
    # 稳健窗口调整（固定窗口+间距补偿）
    base_window = 20
    line_distance = abs(c_pos - t_pos)
    window = base_window + int(line_distance * 0.1)
    window = max(15, min(window, 40))

    # ROI提取与边界保护
    t_roi = img[:, max(0, t_pos - window):t_pos + window]
    c_roi = img[:, max(0, c_pos - window):c_pos + window]

    # LAB空间转换
    t_lab = cv2.cvtColor(t_roi, cv2.COLOR_BGR2LAB)
    c_lab = cv2.cvtColor(c_roi, cv2.COLOR_BGR2LAB)

    # 稳健通道权重策略（固定权重+动态微调）
    def get_stable_weights(roi_lab):
        l_std = np.std(roi_lab[:, :, 0])
        a_std = np.std(roi_lab[:, :, 1])
        base_weights = [0.7, 0.2, 0.1]

        a_adjust = min(a_std / (l_std + 1e-6) * 0.1, 0.15)
        return [
            base_weights[0] - a_adjust,
            base_weights[1] + a_adjust,
            base_weights[2]
        ]

    w_t = get_stable_weights(t_lab)
    w_c = get_stable_weights(c_lab)

    # 通道融合（带归一化）
    t_values = (w_t[0] * t_lab[:, :, 0] + w_t[1] * t_lab[:, :, 1]) / (sum(w_t[:2]) + 1e-6)
    c_values = (w_c[0] * c_lab[:, :, 0] + w_c[1] * c_lab[:, :, 1]) / (sum(w_c[:2]) + 1e-6)

    # 浓度自适应增强策略
    def adaptive_enhance(values, pos1, pos2):
        line_distance = abs(pos2 - pos1)
        alpha = 1.2 + 0.4 * (1 - np.clip(line_distance / 500, 0, 1))
        beta = 15 * (0.8 + 0.4 * (line_distance / 500))

        enhanced = np.where(
            values < 150,
            values * alpha,
            np.sqrt(values) * beta
        )
        suppress_factor = np.clip((np.max(values) - 160) / 50, 0.8, 1.2)
        return enhanced / (180 * suppress_factor)

    t_enhanced = adaptive_enhance(t_values, t_pos, c_pos)
    c_enhanced = adaptive_enhance(c_values, c_pos, t_pos)