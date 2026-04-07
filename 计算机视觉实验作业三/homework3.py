import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import log10, sqrt

# ===================== 自动创建输出文件夹 =====================
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ===================== 1. 读入灰度图 =====================
img = cv2.imread('gray_test.jpg', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("未找到 gray_test.jpg，请检查路径！")
h, w = img.shape
cv2.imwrite(f"{output_dir}/Original.jpg", img)

# ===================== 2. 1/2 下采样（两种方式） =====================
new_w, new_h = w // 2, h // 2

# 直接下采样
img_down_direct = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
cv2.imwrite(f"{output_dir}/Downsampled_Direct.jpg", img_down_direct)

# 高斯平滑后下采样
img_blur = cv2.GaussianBlur(img, (5, 5), sigmaX=1.0)
img_down_gauss = cv2.resize(img_blur, (new_w, new_h), interpolation=cv2.INTER_AREA)
cv2.imwrite(f"{output_dir}/Downsampled_Gaussian.jpg", img_down_gauss)

# ===================== 3. 三种插值恢复 =====================
print("========== 1/2 Direct 下采样恢复 ==========")
# 最近邻
res_nn = cv2.resize(img_down_direct, (w, h), interpolation=cv2.INTER_NEAREST)
mse_nn = np.mean((img - res_nn) ** 2)
psnr_nn = 20 * log10(255.0 / sqrt(mse_nn)) if mse_nn != 0 else float('inf')
print(f"Nearest    | MSE: {mse_nn:7.2f} | PSNR: {psnr_nn:6.2f} dB")

# 双线性
res_bl = cv2.resize(img_down_direct, (w, h), interpolation=cv2.INTER_LINEAR)
mse_bl = np.mean((img - res_bl) ** 2)
psnr_bl = 20 * log10(255.0 / sqrt(mse_bl)) if mse_bl != 0 else float('inf')
print(f"Bilinear   | MSE: {mse_bl:7.2f} | PSNR: {psnr_bl:6.2f} dB")

# 双三次
res_bc = cv2.resize(img_down_direct, (w, h), interpolation=cv2.INTER_CUBIC)
mse_bc = np.mean((img - res_bc) ** 2)
psnr_bc = 20 * log10(255.0 / sqrt(mse_bc)) if mse_bc != 0 else float('inf')
print(f"Bicubic    | MSE: {mse_bc:7.2f} | PSNR: {psnr_bc:6.2f} dB")

print("\n========== 1/2 Gaussian 下采样恢复 ==========")
# 最近邻
res_nn_g = cv2.resize(img_down_gauss, (w, h), interpolation=cv2.INTER_NEAREST)
mse_nn_g = np.mean((img - res_nn_g) ** 2)
psnr_nn_g = 20 * log10(255.0 / sqrt(mse_nn_g)) if mse_nn_g != 0 else float('inf')
print(f"Nearest    | MSE: {mse_nn_g:7.2f} | PSNR: {psnr_nn_g:6.2f} dB")

# 双线性
res_bl_g = cv2.resize(img_down_gauss, (w, h), interpolation=cv2.INTER_LINEAR)
mse_bl_g = np.mean((img - res_bl_g) ** 2)
psnr_bl_g = 20 * log10(255.0 / sqrt(mse_bl_g)) if mse_bl_g != 0 else float('inf')
print(f"Bilinear   | MSE: {mse_bl_g:7.2f} | PSNR: {psnr_bl_g:6.2f} dB")

# 双三次
res_bc_g = cv2.resize(img_down_gauss, (w, h), interpolation=cv2.INTER_CUBIC)
mse_bc_g = np.mean((img - res_bc_g) ** 2)
psnr_bc_g = 20 * log10(255.0 / sqrt(mse_bc_g)) if mse_bc_g != 0 else float('inf')
print(f"Bicubic    | MSE: {mse_bc_g:7.2f} | PSNR: {psnr_bc_g:6.2f} dB")

# ===================== ✅ 保存所有恢复后的图像（你要的功能） =====================
cv2.imwrite(f"{output_dir}/Restored_Direct_Nearest.jpg", res_nn)
cv2.imwrite(f"{output_dir}/Restored_Direct_Bilinear.jpg", res_bl)
cv2.imwrite(f"{output_dir}/Restored_Direct_Bicubic.jpg", res_bc)

cv2.imwrite(f"{output_dir}/Restored_Gaussian_Nearest.jpg", res_nn_g)
cv2.imwrite(f"{output_dir}/Restored_Gaussian_Bilinear.jpg", res_bl_g)
cv2.imwrite(f"{output_dir}/Restored_Gaussian_Bicubic.jpg", res_bc_g)

# ===================== 4. FFT 分析 =====================
def fft_analysis(image):
    fft = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft)
    mag = 20 * np.log(np.abs(fft_shift) + 1e-8)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

cv2.imwrite(f"{output_dir}/FFT_Original.jpg", fft_analysis(img))
cv2.imwrite(f"{output_dir}/FFT_Downsampled_Direct.jpg", fft_analysis(img_down_direct))
cv2.imwrite(f"{output_dir}/FFT_Downsampled_Gaussian.jpg", fft_analysis(img_down_gauss))
cv2.imwrite(f"{output_dir}/FFT_Direct_Bilinear.jpg", fft_analysis(res_bl))
cv2.imwrite(f"{output_dir}/FFT_Gaussian_Bilinear.jpg", fft_analysis(res_bl_g))

# ===================== 5. DCT 分析 + 能量统计 =====================
def dct_analysis(image):
    dct = cv2.dct(np.float32(image))
    dct_log = np.log(np.abs(dct) + 1e-8)
    return dct, cv2.normalize(dct_log, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

def dct_energy_ratio(dct_coeff, ratio=0.25):
    h, w = dct_coeff.shape
    h_low = int(h * ratio)
    w_low = int(w * ratio)
    low_energy = np.sum(np.abs(dct_coeff[:h_low, :w_low]) ** 2)
    total_energy = np.sum(np.abs(dct_coeff) ** 2)
    return low_energy / total_energy if total_energy != 0 else 0

# 原图DCT
dct_ori, dct_img_ori = dct_analysis(img)
ratio_ori = dct_energy_ratio(dct_ori)
cv2.imwrite(f"{output_dir}/DCT_Original.jpg", dct_img_ori)

# Direct 恢复 DCT
dct_nn, dct_img_nn = dct_analysis(res_nn)
dct_bl, dct_img_bl = dct_analysis(res_bl)
dct_bc, dct_img_bc = dct_analysis(res_bc)
ratio_nn = dct_energy_ratio(dct_nn)
ratio_bl = dct_energy_ratio(dct_bl)
ratio_bc = dct_energy_ratio(dct_bc)

cv2.imwrite(f"{output_dir}/DCT_Direct_Nearest.jpg", dct_img_nn)
cv2.imwrite(f"{output_dir}/DCT_Direct_Bilinear.jpg", dct_img_bl)
cv2.imwrite(f"{output_dir}/DCT_Direct_Bicubic.jpg", dct_img_bc)

# Gaussian 恢复 DCT
dct_nn_g, dct_img_nn_g = dct_analysis(res_nn_g)
dct_bl_g, dct_img_bl_g = dct_analysis(res_bl_g)
dct_bc_g, dct_img_bc_g = dct_analysis(res_bc_g)
ratio_nn_g = dct_energy_ratio(dct_nn_g)
ratio_bl_g = dct_energy_ratio(dct_bl_g)
ratio_bc_g = dct_energy_ratio(dct_bc_g)

cv2.imwrite(f"{output_dir}/DCT_Gaussian_Nearest.jpg", dct_img_nn_g)
cv2.imwrite(f"{output_dir}/DCT_Gaussian_Bilinear.jpg", dct_img_bl_g)
cv2.imwrite(f"{output_dir}/DCT_Gaussian_Bicubic.jpg", dct_img_bc_g)

# 打印能量占比
print("\n========== DCT 低频能量占比（左上角25%区域） ==========")
print(f"原图        : {ratio_ori:.2%}")
print(f"Direct-Nearest: {ratio_nn:.2%}")
print(f"Direct-Bilinear: {ratio_bl:.2%}")
print(f"Direct-Bicubic: {ratio_bc:.2%}")
print(f"Gauss-Nearest: {ratio_nn_g:.2%}")
print(f"Gauss-Bilinear: {ratio_bl_g:.2%}")
print(f"Gauss-Bicubic: {ratio_bc_g:.2%}")

# ===================== 6. 图像显示 =====================
plt.figure(figsize=(22, 12))

# 第1行：原图 + 缩小图
plt.subplot(4, 6, 1); plt.imshow(img, cmap='gray'); plt.title('Original'); plt.axis('off')
plt.subplot(4, 6, 2); plt.imshow(img_down_direct, cmap='gray'); plt.title('Direct 1/2'); plt.axis('off')
plt.subplot(4, 6, 3); plt.imshow(img_down_gauss, cmap='gray'); plt.title('Gaussian 1/2'); plt.axis('off')

# 第1行DCT
plt.subplot(4, 6, 4); plt.imshow(dct_img_ori, cmap='gray'); plt.title('DCT Original'); plt.axis('off')

# 第2行：Direct 恢复
plt.subplot(4, 6, 7); plt.imshow(res_nn, cmap='gray'); plt.title('Direct-Nearest'); plt.axis('off')
plt.subplot(4, 6, 8); plt.imshow(res_bl, cmap='gray'); plt.title('Direct-Bilinear'); plt.axis('off')
plt.subplot(4, 6, 9); plt.imshow(res_bc, cmap='gray'); plt.title('Direct-Bicubic'); plt.axis('off')

# 第2行DCT
plt.subplot(4, 6, 10); plt.imshow(dct_img_nn, cmap='gray'); plt.title('DCT Nearest'); plt.axis('off')
plt.subplot(4, 6, 11); plt.imshow(dct_img_bl, cmap='gray'); plt.title('DCT Bilinear'); plt.axis('off')
plt.subplot(4, 6, 12); plt.imshow(dct_img_bc, cmap='gray'); plt.title('DCT Bicubic'); plt.axis('off')

# 第3行：Gaussian 恢复
plt.subplot(4, 6, 13); plt.imshow(res_nn_g, cmap='gray'); plt.title('Gauss-Nearest'); plt.axis('off')
plt.subplot(4, 6, 14); plt.imshow(res_bl_g, cmap='gray'); plt.title('Gauss-Bilinear'); plt.axis('off')
plt.subplot(4, 6, 15); plt.imshow(res_bc_g, cmap='gray'); plt.title('Gauss-Bicubic'); plt.axis('off')

# 第3行DCT
plt.subplot(4, 6, 16); plt.imshow(dct_img_nn_g, cmap='gray'); plt.title('DCT Gauss-Nearest'); plt.axis('off')
plt.subplot(4, 6, 17); plt.imshow(dct_img_bl_g, cmap='gray'); plt.title('DCT Gauss-Bilinear'); plt.axis('off')
plt.subplot(4, 6, 18); plt.imshow(dct_img_bc_g, cmap='gray'); plt.title('DCT Gauss-Bicubic'); plt.axis('off')

plt.tight_layout()
plt.show()

