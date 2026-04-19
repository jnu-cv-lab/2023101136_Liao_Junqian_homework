import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. 配置参数
output_dir = "perspective_correction_results"
os.makedirs(output_dir, exist_ok=True)

# A4纸比例
OUTPUT_WIDTH = 600
OUTPUT_HEIGHT = int(OUTPUT_WIDTH * 1.414)

# 2. 加载图像
def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像：{image_path}")
    return img

# 3. 透视校正
def perspective_correction(img, corners):
    # A4纸的四个角点
    dst_pts = np.array([
        [0, 0],
        [OUTPUT_WIDTH-1, 0],
        [OUTPUT_WIDTH-1, OUTPUT_HEIGHT-1],
        [0, OUTPUT_HEIGHT-1]
    ], dtype=np.float32)
    
    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(corners, dst_pts)
    
    # 应用变换
    corrected = cv2.warpPerspective(img, M, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    
    return corrected, M

# 4. 主函数
if __name__ == "__main__":
    image_path = "A4paper.jpg"
    img = load_image(image_path)
    corners = np.float32([
        [270, 310],   # 左上
        [980, 350],   # 右上
        [1260, 1330], # 右下
        [10, 1328]    # 左下
    ])
    
    # 3. 执行透视校正
    corrected_img, M = perspective_correction(img, corners)
    
    # 4. 保存结果
    cv2.imwrite(os.path.join(output_dir, "corrected_A4.jpg"), corrected_img)
    print(f"校正后的图像已保存到: {output_dir}/corrected_A4.jpg")
    print(f"透视变换矩阵:\n{M}")
    
    plt.figure(figsize=(14, 8))
    
    # 左图原始图像
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # 右图校正后图像
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(corrected_img, cv2.COLOR_BGR2RGB))
    plt.title('Perspective Corrected Image', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison.jpg"), dpi=300, bbox_inches='tight')
    
    plt.show()
    
    print(f"对比图已保存到: {output_dir}/comparison.jpg")