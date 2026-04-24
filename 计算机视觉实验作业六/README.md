# 计算机视觉实验五
## 实验内容与要求
### 任务1：ORB关键点检测与描述子提取
- 使用 OpenCV 的 ORB 算法检测两幅图像中的关键点和描述子。
1. 使用 cv2.ORB_create() 创建 ORB 检测器；
2. 设置 nfeatures=1000；
3. 使用 detectAndCompute() 得到关键点和描述子；
4. 使用 cv2.drawKeypoints() 可视化关键点；
5. 输出两幅图像中的关键点数量；
6. 输出描述子的维度。
- 需要提交：
- box.png 的特征点可视化图；
- box_in_scene.png 的特征点可视化图；
- 关键点数量和描述子维度。


### 任务2：ORB 特征匹配
- 使用 ORB 描述子对两幅图像进行特征匹配。
1. 使用 cv2.BFMatcher() 创建暴力匹配器；
2. ORB 描述子使用 cv2.NORM_HAMMING；
3. 可以使用 crossCheck=True；
4. 按照匹配距离从小到大排序；
5. 显示前 30 或前 50 个匹配结果；
6. 输出总匹配数量。

- 需要提交：
- ORB 初始匹配图；
- 总匹配数量；
- 前 30 或前 50 个匹配的可视化结果。

### 任务3：RANSAC 剔除错误匹配
- 使用匹配点估计单应矩阵 Homography，并利用 RANSAC 剔除错误匹配。
1. 从匹配结果中提取两幅图像中的对应点坐标；
2. 使用 cv2.findHomography()；
3. 方法选择 cv2.RANSAC；
4. 设置合适的重投影误差阈值，例如 5.0；
5. 根据返回的 mask 显示 RANSAC 后的内点匹配；
6. 输出内点数量、总匹配数量和内点比例。
### 任务4：目标定位
- 使用估计出的 Homography，将 box.png 的四个角点投影到 box_in_scene.png 中，并画出目标物体的位置。
1. 获取 box.png 的四个角点；
2. 使用 cv2.perspectiveTransform() 进行角点投影；
3. 使用 cv2.polylines() 在场景图中画出四边形边框；
4. 显示最终目标定位结果。
### 任务5：参数对比实验
- 改变 ORB 的 nfeatures 参数，观察匹配效果变化。至少测试以下三组参数：
- nfeatures = 500
- nfeatures = 1000
- nfeatures = 2000
1. 比较不同 nfeatures 对匹配数量的影响；
2. 比较不同 nfeatures 对 RANSAC 内点比例的影响；
3. 说明是否特征点越多，定位效果就一定越好。
### 选做任务：SIFT 特征匹配
1. 使用 cv2.SIFT_create()；
2. 使用 cv2.NORM_L2 进行匹配；
3. 使用 KNN matching；
4. 使用 Lowe ratio test 筛选匹配；
5. 使用 RANSAC + Homography 完成目标定位；
6. 与 ORB 的结果进行比较。
## 实验结果与分析
- 见计算机视觉实验报告六
## 图像信息
- original：原始测试图
- similarity：相似变换
- affine：仿射变换
- perspective：透视变换
- comparison：比较图
- A4paper：A4纸原图
- corrected_A4：校正后的A4纸
- comparison_A4：比较图（A4）





