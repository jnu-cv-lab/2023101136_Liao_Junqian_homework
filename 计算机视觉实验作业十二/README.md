# 使用棋盘格进行相机标定

## 图像信息
- results/corner_detect：所有角点标记图存放处
- results/compare：每张图片的原图+校正对比图
- images：标定图

## 代码信息
- generate_board.py：生成标准的9*6棋盘格
- main.py：主程序

## 实验内容与要求
### 任务：
- 定义棋盘格角点在标定板坐标系中的三维坐标；
- 读取所有标定图片，使用 OpenCV 检测棋盘格角点；
- 对检测到的角点进行亚像素精度优化；
- 估计相机内参矩阵 K、畸变参数和每张图片的外参；
- 输出重投影误差，并对至少一张图片进行去畸变处理；
- 对比原图和去畸变后的图像。

## 项目目录结构
```text
计算机视觉实验作业十二/
├── images/
│   ├── img1
│   ├── img2
│   └── ...
│      
├── code/
│   ├── generate_board.py
│   └── main.py
├── results/
│   ├─corner_detect/          # 所有角点标记图存放处
│   │  ├─ corner_detect_00.jpg
│   │  ├─ corner_detect_01.jpg
│   │  └─ ...
│   │
│   ├─ compare/                # 每张图片的原图+校正对比图
│   │  ├─ compare_00.jpg
│   │  ├─ compare_01.jpg
│   │  └─ ...
│   └─ camera_calib_params.npz # 标定参数文件
│
└──

