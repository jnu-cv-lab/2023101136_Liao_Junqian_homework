# model.py
import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    """正弦余弦位置编码（改进：更稳定的相对位置建模）"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class SkeletonTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, num_classes, dropout):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)   # 改进1：正弦余弦位置编码
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 改进2：CLS token + Mean Pooling 拼接
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 64),   # 两个池化输出拼接后维度 d_model*2
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (B, T, input_dim)
        x = self.embedding(x)                     # (B, T, d_model)
        B, T, _ = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)   # (B, 1, d_model)
        x = torch.cat((cls_tokens, x), dim=1)          # (B, T+1, d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x)                        # (B, T+1, d_model)
        cls_out = x[:, 0, :]                           # CLS token 输出
        mean_out = x[:, 1:, :].mean(dim=1)             # 其他帧的均值池化
        combined = torch.cat([cls_out, mean_out], dim=1)  # (B, d_model*2)
        logits = self.classifier(combined)
        return logits