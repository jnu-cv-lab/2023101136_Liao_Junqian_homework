# 实现并比较 Sinusoidal Position Encoding 与 RoPE
## 图像信息
- sinusoidal_pe：Sinusoidal Position Encoding热力图
- rope_high_dim：高维 RoPE 特征热力图
- input_comparison：E+pos 与 RoPE 左右对比热力图
- relative_position：RoPE 相对位置验证柱状图
## 代码信息
- 2023101136_Liao_Junqian
## 实验结果与分析
- 见实验报告
## 实验内容与要求
### 任务：
1. 实现 sinusoidal position encoding； 
2. 实现二维向量旋转； 
3. 实现高维 RoPE；
4. 对比 E+pos 和 RoPE 的输入方式；
5. 用数值实验验证 RoPE 的相对位置性质；
6. 说明：为什么 RoPE 比简单的 E+pos 更巧妙？
### 回答：
1. Transformer 为什么需要位置编码；
2. 传统 sinusoidal position encoding 是如何生成的；
3. E+pos 的位置注入方式为什么有“内容和位置混合”的问题；
4. RoPE 不是加法，而是旋转；
5. RoPE 作用在 Q 和 K 上；
6. RoPE 的点积天然包含相对位置；
7. attention score 里的相对位置关系可以通过旋转结构自然出现。
