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

1. 数据预处理
- python src/preprocess.py
2. 训练模型
- python src/train.py
3. 单样本推理
python src/inference.py

- 注意力分析
- python src/attention_analysis.py
- 骨架可视化
- python src/visualize_skeleton.py

## 实验内容与要求
### 任务：
- 理解视频动作识别任务如何转化为骨架时间序列分类任务。
- 掌握使用 MediaPipe Pose 从视频帧中提取人体 33 个关键点的方法。
- 掌握将每段视频统一转换为固定长度骨架序列的方法，例如 [60, 132]。
- 实现一个轻量级 Skeleton Transformer，用 Transformer Encoder 对动作序列分类。
- 完成模型训练、测试集评估与单个视频样本推理。
- 理解该方法在羽毛球商业化视频分析中的优势与局限。
