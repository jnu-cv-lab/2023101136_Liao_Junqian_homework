import cv2
import numpy as np
import os

output_dir = "../temp"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 内角点规格：横向9个交点，纵向6个交点 → 方格矩阵是10列 × 7行
inner_w, inner_h = 9, 6
square_px = 150
margin = 120

# 画布尺寸
img_width = (inner_w + 1) * square_px + 2 * margin
img_height = (inner_h + 1) * square_px + 2 * margin
# 纯白背景
board_img = np.ones((img_height, img_width), dtype=np.uint8) * 255

# 修正绘制逻辑：从第二行第二列开始交错填黑，四周保留完整白方格包裹网格
for y_idx in range(inner_h + 1):
    for x_idx in range(inner_w + 1):
        # 错开填充，保证网格四周有白色方格包围，无外露单排黑块
        if (x_idx + y_idx) % 2 == 1:
            x_start = margin + x_idx * square_px
            y_start = margin + y_idx * square_px
            x_end = x_start + square_px
            y_end = y_start + square_px
            board_img[y_start:y_end, x_start:x_end] = 0

save_path = os.path.join(output_dir, "standard_9x6_chessboard.jpg")
cv2.imwrite(save_path, board_img)
print(f" 标准可识别棋盘已生成：{save_path}")

# 立刻测试角点检测
gray = board_img.copy()
ret, corners = cv2.findChessboardCorners(gray, (9, 6))
print(f"模板原图检测结果 ret = {ret}")
if ret:
    cv2.drawChessboardCorners(board_img, (9,6), corners, ret)
    cv2.imwrite(os.path.join(output_dir, "template_corner_check.jpg"), board_img)
    print("角点绘制图已保存，证明棋盘完全合格！")