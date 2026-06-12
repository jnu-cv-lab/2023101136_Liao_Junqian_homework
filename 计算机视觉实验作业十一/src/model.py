import torch
import torch.nn as nn
import math

class SkeletonTransformer(nn.Module):
    def __init__(self, input_dim=132, d_model=128, nhead=4, num_layers=2, 
                 dim_feedforward=256, num_classes=6, dropout=0.1, target_frames=30):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, target_frames, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='relu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, num_classes)
        )
    
    def forward(self, x):
        # x: (B, T, input_dim)
        x = self.embedding(x)               # (B, T, d_model)
        x = x + self.pos_embedding[:, :x.size(1), :]  # 加位置编码
        x = self.transformer_encoder(x)     # (B, T, d_model)
        x = x.mean(dim=1)                   # 全局平均池化 (或使用CLS token)
        x = self.norm(x)
        logits = self.classifier(x)         # (B, num_classes)
        return logits