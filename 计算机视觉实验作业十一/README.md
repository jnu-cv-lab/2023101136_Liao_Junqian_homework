# 基于 MediaPipe Pose 与骨架序列 Transformer 的羽毛球击球动作识别

## 图像信息
- training_curves：训练损失曲线与训练测试准确率曲线图
- confusion_matrix：六类动作混淆矩阵热力图
- attention_heatmap：Transformer 最后一层 CLS token 对视频帧的注意力权重分布图
- demo_skeleton.mp4：骨架可视化视频 

## 代码信息
- preprocess.py：视频预处理、骨架提取、归一化、数据集划分与保存
- model.py：SkeletonTransformer 模型定义
- train.py：模型训练
- inference.py：单视频推理与预测
- attention_analysis.py：注意力权重提取与可视化
- visualize_skeleton.py：骨架连线可视化生成视频

## 实验内容与要求
### 任务：
1. 使用 MediaPipe Pose 提取视频中每帧 33 个人体关键点，展平为 132 维特征，重采样至固定帧数 30 帧，并进行以髋部为中心、肩宽为尺度的归一化。
2. 设计轻量级 Skeleton Transformer 模型：线性嵌入、可学习位置编码、两层 Transformer Encoder、平均池化、MLP 分类器，输出 6 类击球动作 logits。
3. 完成模型训练（交叉熵损失、Adam 优化器、早停），输出测试准确率、分类报告与混淆矩阵。
4. 实现单视频推理，输出预测类别与置信度。
5. 实现骨架可视化（画出人体关键点与连线）与注意力权重分析（提取 CLS token 对帧的注意力分布）。
6. 分析过拟合原因，提出改进方案（数据增强、正则化、早停、简化模型等）。
