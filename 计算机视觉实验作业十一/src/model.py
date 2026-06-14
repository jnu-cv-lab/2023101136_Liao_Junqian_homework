import torch
import torch.nn as nn

class TransformerEncoderLayerWithAttn(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', batch_first=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU() if activation == 'relu' else nn.GELU()
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention with need_weights=True
        src2, attn = self.self_attn(src, src, src, attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask,
                                    need_weights=True)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # Feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn

class SkeletonTransformer(nn.Module):
    def __init__(self, input_dim=132, d_model=128, nhead=4, num_layers=2,
                 dim_feedforward=256, num_classes=6, dropout=0.1, target_frames=30):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, target_frames, d_model))
        # 使用自定义的编码层列表
        self.layers = nn.ModuleList([
            TransformerEncoderLayerWithAttn(d_model, nhead, dim_feedforward, dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, num_classes)
        )

    def forward(self, x, return_attention=False):
        # x: (B, T, input_dim)
        x = self.embedding(x)                           # (B, T, d_model)
        T = x.size(1)
        pos_embed = self.pos_embedding[:, :T, :]        # (1, T, d_model)
        x = x + pos_embed

        attentions = []
        for layer in self.layers:
            if return_attention:
                x, attn = layer(x)
                attentions.append(attn)
            else:
                x, _ = layer(x)

        x = x.mean(dim=1)                               # (B, d_model)
        x = self.norm(x)
        logits = self.classifier(x)                     # (B, num_classes)

        if return_attention:
            return logits, attentions
        return logits
