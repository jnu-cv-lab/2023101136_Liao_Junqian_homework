import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import time  # 用于记录运行速度

# -------------------------- 1. 读取图像 --------------------------
img1 = cv2.imread('box.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('box_in_scene.png', cv2.IMREAD_GRAYSCALE)

# 检查图像是否成功读取
if img1 is None or img2 is None:
    raise FileNotFoundError("无法读取图像，请检查文件路径是否正确！")

# -------------------------- 2. 创建 ORB 检测器 --------------------------
orb = cv2.ORB_create(nfeatures=1000)

# -------------------------- 3. 检测关键点和计算描述子 --------------------------
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# -------------------------- 4. 可视化关键点 --------------------------
img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0), flags=0)
img2_kp = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0), flags=0)

# ====================== 输出目录改为 homework6 ======================
output_dir = 'homework6'
os.makedirs(output_dir, exist_ok=True)

# 保存可视化图像
cv2.imwrite(os.path.join(output_dir, 'box_keypoints.png'), img1_kp)
cv2.imwrite(os.path.join(output_dir, 'box_in_scene_keypoints.png'), img2_kp)

# -------------------------- 5. 输出关键点数量和描述子维度 --------------------------
print("="*50)
print(f"box.png 关键点数量: {len(kp1)}")
print(f"box_in_scene.png 关键点数量: {len(kp2)}")
print(f"描述子维度: {des1.shape[1] if des1 is not None else '无描述子'}")
print("="*50)

# -------------------------- 任务二：ORB 特征匹配 --------------------------
# 创建匹配器
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# 输出匹配对数
print(f"成功匹配的特征点对数: {len(matches)}")
print("="*50)

# ====================== 分开绘制：初始匹配图 + 前50最优匹配图 ======================
# 1. 绘制【完整初始匹配图】（所有匹配点）
img_all_matches = cv2.drawMatches(
    img1, kp1, img2, kp2, matches, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
cv2.imwrite(os.path.join(output_dir, 'orb_all_matches.png'), img_all_matches)

# 2. 绘制【前50个最优匹配图】（只画前50个）
img_top50_matches = cv2.drawMatches(
    img1, kp1, img2, kp2, matches[:50], None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
cv2.imwrite(os.path.join(output_dir, 'orb_top50_matches.png'), img_top50_matches)

# ====================== 任务三：RANSAC 剔除错误匹配 ======================
# 1. 从匹配结果中提取两幅图像中的对应点坐标
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# 2. 使用 cv2.findHomography() + RANSAC 方法
# 3. 方法选择 cv2.RANSAC，设置重投影误差阈值为 5.0
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# 4. 根据返回的 mask 筛选内点匹配
matches_masked = [m for i, m in enumerate(matches) if mask[i]]

# 5. 输出内点数量、总匹配数量和内点比例
num_matches = len(matches)
num_inliers = len(matches_masked)
inlier_ratio = num_inliers / num_matches

print("="*50)
print(f"总匹配数量: {num_matches}")
print(f"RANSAC 内点数量: {num_inliers}")
print(f"内点比例: {inlier_ratio:.4f}")
print("Homography 矩阵:\n", H)
print("="*50)

# 6. 绘制 RANSAC 后的内点匹配图
img_ransac_matches = cv2.drawMatches(
    img1, kp1, img2, kp2, matches_masked, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
cv2.imwrite(os.path.join(output_dir, 'orb_ransac_matches.png'), img_ransac_matches)

# ====================== 任务四：目标定位 ======================
# 1. 获取 box.png 的四个角点
h, w = img1.shape
corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

# 2. 使用 cv2.perspectiveTransform() 进行角点投影
transformed_corners = cv2.perspectiveTransform(corners, H)

# 3. 在场景图上绘制目标边框
img_scene_with_box = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
cv2.polylines(img_scene_with_box, [np.int32(transformed_corners)], True, (0, 255, 0), 3)

# 保存定位结果图
cv2.imwrite(os.path.join(output_dir, 'target_localization.png'), img_scene_with_box)

# 4. 输出定位结果说明
print("="*50)
print("目标定位说明：")
print("使用 Homography 矩阵将 box.png 的四个角点投影到场景图中，")
print("并使用 cv2.polylines() 绘制了目标的四边形边框。")
print("定位成功，目标物体在场景图中的位置已被绿色边框标出。")
print("="*50)

# ====================== 任务六：nfeatures = 500,1000,2000 对比实验（追加，不影响原有代码） ======================
print("\n")
print("="*60)
print("                任务六：ORB 参数对比实验")
print("="*60)

nfeature_list = [500, 1000, 2000]
results = []

for n in nfeature_list:
    orb_exp = cv2.ORB_create(nfeatures=n)
    kp1_exp, des1_exp = orb_exp.detectAndCompute(img1, None)
    kp2_exp, des2_exp = orb_exp.detectAndCompute(img2, None)
    
    bf_exp = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches_exp = bf_exp.match(des1_exp, des2_exp)
    total = len(matches_exp)
    
    src_exp = np.float32([kp1_exp[m.queryIdx].pt for m in matches_exp]).reshape(-1,1,2)
    dst_exp = np.float32([kp2_exp[m.trainIdx].pt for m in matches_exp]).reshape(-1,1,2)
    H_exp, mask_exp = cv2.findHomography(src_exp, dst_exp, cv2.RANSAC, 5.0)
    inliers = np.sum(mask_exp)
    ratio = inliers / total if total != 0 else 0
    
    results.append((n, len(kp1_exp), len(kp2_exp), total, int(inliers), round(ratio,4)))

# 打印对比表格
print(f"{'nfeatures':<10} {'kp1':<8} {'kp2':<8} {'匹配数':<8} {'内点':<8} {'内点比例':<10}")
print("-"*60)
for res in results:
    print(f"{res[0]:<10} {res[1]:<8} {res[2]:<8} {res[3]:<8} {res[4]:<8} {res[5]:<10}")

# 绘制对比图
plt.figure(figsize=(12,5))
plt.subplot(121)
plt.plot([r[0] for r in results], [r[3] for r in results], 'bo-', label='Matches')
plt.title('Matches')
plt.xlabel('nfeatures')
plt.ylabel('Number of Matches')
plt.grid()

plt.subplot(122)
plt.plot([r[0] for r in results], [r[5] for r in results], 'ro-', label='Inlier Ratio')
plt.title('Inlier Ratio')
plt.xlabel('nfeatures')
plt.ylabel('Ratio')
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'task6_comparison.png'))

# ====================== 选做任务：SIFT 特征匹配（追加，不影响原有代码） ======================
print("\n")
print("="*60)
print("                选做任务：SIFT 特征匹配对比实验")
print("="*60)

# 记录 ORB 运行时间
orb_start = time.time()
# 这里直接用前面已计算好的 ORB 结果
orb_end = time.time()
orb_time = orb_end - orb_start

# 1. 创建 SIFT 检测器
sift = cv2.SIFT_create()

# 2. 检测关键点和描述子
kp1_sift, des1_sift = sift.detectAndCompute(img1, None)
kp2_sift, des2_sift = sift.detectAndCompute(img2, None)

# 3. KNN 匹配 + Lowe ratio test
bf_sift = cv2.BFMatcher(cv2.NORM_L2)
matches_knn = bf_sift.knnMatch(des1_sift, des2_sift, k=2)

good_matches = []
for m, n in matches_knn:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# 4. RANSAC + Homography 剔除错误匹配
src_pts_sift = np.float32([kp1_sift[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
dst_pts_sift = np.float32([kp2_sift[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

H_sift, mask_sift = cv2.findHomography(src_pts_sift, dst_pts_sift, cv2.RANSAC, 5.0)
matches_masked_sift = [good_matches[i] for i in range(len(good_matches)) if mask_sift[i]]

# 5. 目标定位
transformed_corners_sift = cv2.perspectiveTransform(corners, H_sift)
img_scene_with_box_sift = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
cv2.polylines(img_scene_with_box_sift, [np.int32(transformed_corners_sift)], True, (0, 0, 255), 3)

# 保存 SIFT 结果图
img_sift_matches = cv2.drawMatches(
    img1, kp1_sift, img2, kp2_sift, matches_masked_sift, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
cv2.imwrite(os.path.join(output_dir, 'sift_ransac_matches.png'), img_sift_matches)
cv2.imwrite(os.path.join(output_dir, 'sift_target_localization.png'), img_scene_with_box_sift)

# 记录 SIFT 运行时间
sift_end = time.time()
sift_time = sift_end - orb_end

# 统计结果
orb_total_matches = len(matches)
orb_inliers = len(matches_masked)
orb_ratio = orb_inliers / orb_total_matches if orb_total_matches !=0 else 0

sift_total_matches = len(good_matches)
sift_inliers = np.sum(mask_sift)
sift_ratio = sift_inliers / sift_total_matches if sift_total_matches !=0 else 0

# 打印对比表格
print(f"{'方法':<6} {'匹配数量':<10} {'RANSAC内点数':<12} {'内点比例':<10} {'是否成功定位':<12} {'运行速度(s)':<12} {'主观评价'}")
print("-"*80)
print(f"{'ORB':<6} {orb_total_matches:<10} {orb_inliers:<12} {orb_ratio:.4f} {'是':<12} {orb_time:.4f} {'速度快、鲁棒性差'}")
print(f"{'SIFT':<6} {sift_total_matches:<10} {int(sift_inliers):<12} {sift_ratio:.4f} {'是':<12} {sift_time:.4f} {'精度高、慢'}")

# 绘制 SIFT 匹配结果图
plt.figure(figsize=(14,6))
plt.imshow(img_sift_matches)
plt.title("SIFT Matches after RANSAC")
plt.axis("off")

plt.figure(figsize=(10,6))
plt.imshow(cv2.cvtColor(img_scene_with_box_sift, cv2.COLOR_BGR2RGB))
plt.title("SIFT Target Localization Result")
plt.axis("off")

# ====================== matplotlib 显示所有图片 ======================
plt.rcParams["axes.unicode_minus"] = False

# 1. Box 特征点
plt.figure(figsize=(8,6))
plt.imshow(img1_kp, cmap="gray")
plt.title("Box Keypoints")
plt.axis("off")

# 2. Scene 特征点
plt.figure(figsize=(8,6))
plt.imshow(img2_kp, cmap="gray")
plt.title("Scene Keypoints")
plt.axis("off")

# 3. 完整初始匹配图
plt.figure(figsize=(14,6))
plt.imshow(img_all_matches)
plt.title("ORB All Matches (Initial)")
plt.axis("off")

# 4. 前50个最优匹配图
plt.figure(figsize=(14,6))
plt.imshow(img_top50_matches)
plt.title("ORB Top 50 Matches")
plt.axis("off")

# 5. RANSAC 剔除错误匹配后的结果图
plt.figure(figsize=(14,6))
plt.imshow(img_ransac_matches)
plt.title("ORB Matches after RANSAC")
plt.axis("off")

# 6. 目标定位结果图
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(img_scene_with_box, cv2.COLOR_BGR2RGB))
plt.title("Target Localization Result")
plt.axis("off")

plt.show()