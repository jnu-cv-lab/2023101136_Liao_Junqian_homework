import cv2
import numpy as np
import matplotlib.pyplot as plt

# 使用 OpenCV 读取测试图片
img = cv2.imread("test.jpg")  # 图片路径
if img is None:
    print("无法读取图片，请检查文件路径是否正确！")
    exit()

# 输出图像基本信息
height, width, channels = img.shape
dtype = img.dtype
print(f"图像尺寸（宽×高）: {width} × {height}")
print(f"图像通道数: {channels}")
print(f"图像数据类型: {dtype}")

# 显示原图
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure("Original Image")
plt.imshow(img_rgb)
plt.axis("off") 
plt.title("Original Image")

# 转换为灰度图并显示
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.figure("Grayscale Image")
plt.imshow(gray_img, cmap="gray")
plt.axis("off")
plt.title("Grayscale Image")

# 保存灰度图
cv2.imwrite("gray_test.jpg", gray_img)
print("灰度图已保存为: gray_test.jpg")

# NumPy操作
# 1.输出某个像素值
pixel_value = img[1000, 100] # 输出坐标 (1000, 100)像素值
print(f"像素 (1000, 100) 的 BGR 值: {pixel_value}")

# 2. 裁剪区域、保存并显示
cropped_img = img[150:1300, 900:1700]
cv2.imwrite("cropped_test.jpg", cropped_img)
print(f"裁剪后的图像已保存为: cropped_test.jpg")
cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
plt.figure("cropped Image")
plt.imshow(cropped_img)
plt.axis("off") 
plt.title("cropped Image")

# 显示所有图像窗口
plt.show()


