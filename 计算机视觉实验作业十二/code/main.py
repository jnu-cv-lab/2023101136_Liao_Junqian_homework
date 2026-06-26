import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import os

# ====================== 定义各级文件夹路径 ======================
# 总输出根目录
output_dir = "../results"
# 角点标记图存放文件夹
corner_dir = os.path.join(output_dir, "corner_detect")
# 原图+去畸变对比图存放文件夹
compare_dir = os.path.join(output_dir, "compare")

# 自动创建全部文件夹
for folder in [output_dir, corner_dir, compare_dir]:
    if not os.path.exists(folder):
        os.makedirs(folder)
print(f"已创建文件夹：")
print(f"角点图目录 -> {corner_dir}")
print(f"对比图目录 -> {compare_dir}")

# ====================== 全局统一标定参数 ======================
chessboard_size = (9, 6)
square_size = 25
img_path = glob.glob("../images/*.jpg")

subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

obj_points = []
img_points = []
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp = objp * square_size

valid_img_num = 0
total_img_count = len(img_path)
print(f"\n共检索到标定图片总数：{total_img_count} 张")

img_cache = []  # 缓存读取成功的原图，后续批量生成对比图
# 第一轮循环：检测角点、保存角点标记图
for idx, img_file in enumerate(img_path):
    print(f"\n---------- 处理第{idx+1}/{total_img_count}张：{img_file} ----------")
    img = cv2.imread(img_file)
    if img is None:
        print(f"图片读取失败，跳过")
        continue
    img_cache.append((idx, img))  # 缓存索引与原图

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if ret:
        valid_img_num += 1
        print(f"角点检测成功！当前有效图：{valid_img_num}")
        corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), subpix_criteria)
        obj_points.append(objp)
        img_points.append(corners_sub)
        # 角点图保存到 corner_detect 文件夹
        cv2.drawChessboardCorners(img, chessboard_size, corners_sub, ret)
        corner_save = os.path.join(corner_dir, f"corner_detect_{idx:02d}.jpg")
        cv2.imwrite(corner_save, img)
    else:
        print(f"未检测到棋盘内角点")

print(f"\n==================== 统计结果 ====================")
print(f"有效标定图片总数：{valid_img_num}")

# 无有效图片直接退出程序
if valid_img_num == 0:
    print("警告：无任何可检测棋盘角点的图片，无法执行标定，程序退出")
    exit()

# ====================== 相机标定求解内参与畸变系数 ======================
total_reproj_err, K, D, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, gray.shape[::-1], None, None
)
mean_reproj_err = total_reproj_err / len(obj_points)

print("=" * 60)
print("【标定求解-相机内参矩阵 K】")
print(K)
print("\n【标定求解-畸变系数 D = [k1,k2,p1,p2,k3]】")
print(D.ravel())
print(f"\n【平均重投影误差】：{mean_reproj_err:.4f} 像素")
print("=" * 60)

# ====================== 批量生成每张图片的原图/去畸变对比图 ======================
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 全局映射矩阵仅计算一次
sample_img = img_cache[0][1]
h, w = sample_img.shape[:2]
new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (w, h), cv2.CV_16SC2)

# 遍历所有缓存图片生成对比图
for idx, src_img in img_cache:
    # 去畸变 + 裁剪黑边
    undist_img = cv2.remap(src_img, map1, map2, cv2.INTER_LINEAR)
    x, y, w_roi, h_roi = roi
    undist_img_crop = undist_img[y:y+h_roi, x:x+w_roi]

    # 绘制左右对比画布
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.title(f"Original Img {idx:02d} (Distorted)")
    plt.imshow(cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(f"Undistorted Img {idx:02d}")
    plt.imshow(cv2.cvtColor(undist_img_crop, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.tight_layout()

    # 对比图保存到 compare 文件夹
    compare_save = os.path.join(compare_dir, f"compare_{idx:02d}.jpg")
    plt.savefig(compare_save, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"对比图已保存：{compare_save}")

# ====================== 保存标定参数文件 ======================
param_save = os.path.join(output_dir, "camera_calib_params.npz")
np.savez(param_save, K=K, D=D, new_K=new_K, roi=roi)
print(f"\n完整标定参数文件已保存：{param_save}")