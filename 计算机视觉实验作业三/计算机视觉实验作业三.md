# 计算机视觉实验三：图像缩小、恢复与频域分析
## 实验题目
- 使用OpenCV读入一幅灰度图像，先对图像进行下采样，再分别用不同内插方法恢复图像尺寸，并结合傅里叶变换和DCT变换对原图与恢复图进行分析比较
## 实验内容与要求
- **1、图像读入与预处理**
- **2、下采样**：将原图缩小为原来的1/2或1/4，分别测试不做预滤波直接缩小，先做高斯平滑再缩小
- **3、图像恢复**：将缩小后的图像恢复到原始尺寸，分别使用以下方法：最近邻内插，双线性内插，双三次内插
- **4、空间域比较**：对原图与恢复图进行比较，至少完成：显示原图、缩小图、恢复图，计算MSE，计算PSNR
- **5、傅里叶变换分析**：分别对原图、缩小后图像、用双线性恢复后的图像计算二位傅里叶变换并显示频谱，要求：将频谱中心移动到图像中心、对2幅度谱取对数显示、比较它们的高频成分差异，并说明原因。
- **6、DCT分析**：分别对原图和恢复图像二维DCT：显示DCT系数图，统计左上角低频区域能量占总能量的比例，比较不同恢复方法下DCT能量分布差异，并给出解释
## 实验结果与分析
- 见计算机视觉实验报告
## 基本信息
- Original：原灰度图
- Downsampled_Direct：直接1/2缩小图
- Downsampled_Gaussian：先做高斯平滑的缩小图
- Restored_Direct_Bicubic/Bilinear/Nearest：双三次/双线性/最近邻内插恢复图
- Restored_Gaussian_Bicubic/Bilinear/Nearest：双三次/双线性/最近邻内插恢复图（高斯平滑）
- FFT_Original：原图频谱
- FFT_Downsampled_Direct/Gaussian：（直接/高斯预处理）缩小后图像频谱
- FFT_Direct/Gaussian_Bilinear：双线性恢复后图像频谱（直接/高斯缩小后）
- DCT_Original：原图的DCT图
- DCT_Direct_Bicubic/Bilinear/Nearest：双三次/双线性/最近邻内插恢复图的DCT图（直接缩小）
- DCT_Gaussian_Bicubic/Bilinear/Nearest：双三次/双线性/最近邻内插恢复图的DCT图（高斯缩小后）
- gray_test：原灰度图


