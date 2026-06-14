import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerEncoderLayerWithAttn(nn.Module):
    """返回注意力的编码层"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', batch_first=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU() if activation == 'relu' else nn.GELU()
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, attn = self.self_attn(src, src, src, attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask,
                                    need_weights=True)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn

class SkeletonTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, num_classes, dropout):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderLayerWithAttn(d_model, nhead, dim_feedforward, dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    def forward(self, x, return_attention=False):
        B, T, _ = x.shape
        x = self.embedding(x)                     # (B, T, d_model)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)     # (B, T+1, d_model)
        x = self.pos_encoder(x)
        attentions = []
        for layer in self.layers:
            if return_attention:
                x, attn = layer(x)
                attentions.append(attn)
            else:
                x, _ = layer(x)
        cls_out = x[:, 0, :]                      # CLS token 输出
        mean_out = x[:, 1:, :].mean(dim=1)        # 平均池化
        combined = torch.cat([cls_out, mean_out], dim=1)
        logits = self.classifier(combined)
        if return_attention:
            return logits, attentions
        return logits

# MediaPipe 初始化
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5
)

def extract_pose_sequence(video_path, target_frames=30):
    # ... (不变，与用户提供的相同)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            frame_features = []
            for lm in landmarks:
                frame_features.extend([lm.x, lm.y, lm.z, lm.visibility])
            frames.append(frame_features)
        else:
            frames.append([0.0] * 132)
    cap.release()
    if len(frames) == 0:
        return None
    frames = np.array(frames, dtype=np.float32)
    current_len = frames.shape[0]
    if current_len == target_frames:
        resampled = frames
    else:
        x_old = np.linspace(0, 1, current_len)
        x_new = np.linspace(0, 1, target_frames)
        resampled = np.zeros((target_frames, 132), dtype=np.float32)
        for i in range(132):
            resampled[:, i] = np.interp(x_new, x_old, frames[:, i])
        frames = resampled
    frames = normalize_pose_sequence(frames)
    return frames

def normalize_pose_sequence(seq):
    T = seq.shape[0]
    normalized = np.zeros_like(seq)
    for t in range(T):
        frame = seq[t].reshape(33, 4)
        left_hip = frame[23, :2]
        right_hip = frame[24, :2]
        hip_center = (left_hip + right_hip) / 2.0
        left_shoulder = frame[11, :2]
        right_shoulder = frame[12, :2]
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
        if shoulder_width < 1e-6:
            shoulder_width = 1.0
        frame[:, 0] = (frame[:, 0] - hip_center[0]) / shoulder_width
        frame[:, 1] = (frame[:, 1] - hip_center[1]) / shoulder_width
        normalized[t] = frame.reshape(132)
    return normalized

def visualize_attention(video_path, model_path='outputs/best_model.pth'):
    INPUT_DIM = 132
    D_MODEL = 128
    NHEAD = 4
    NUM_LAYERS = 2
    DIM_FEEDFORWARD = 256
    NUM_CLASSES = 6
    DROPOUT = 0.1
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(model_path):
        print(f"模型文件未找到: {model_path}")
        return
    model = SkeletonTransformer(INPUT_DIM, D_MODEL, NHEAD, NUM_LAYERS,
                                DIM_FEEDFORWARD, NUM_CLASSES, DROPOUT).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    seq = extract_pose_sequence(video_path, target_frames=30)
    if seq is None:
        print("无法从视频中提取骨架序列")
        return
    seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits, attentions = model(seq_tensor, return_attention=True)
        if not attentions:
            print("模型未返回注意力权重，请确认已使用修改后的 train.py 重新训练模型")
            return
        attn_last = attentions[-1]   # 最后一层注意力
        print(f"注意力张量形状: {attn_last.shape}")  # 预期 (1, 4, 31, 31) 或类似

        # 取第一张样本，所有 head 平均，得到 (31, 31)
        if attn_last.dim() == 4:
            attn_matrix = attn_last[0].mean(dim=0)   # (T+1, T+1)
        elif attn_last.dim() == 3:
            attn_matrix = attn_last[0]
        else:
            print(f"错误的注意力维度: {attn_last.dim()}，期望3或4")
            return

        # CLS token 对应索引 0，提取 CLS 对其他帧的注意力
        attn_cls = attn_matrix[0, 1:].cpu().numpy()   # 长度 = 30

    plt.figure(figsize=(10, 4))
    plt.bar(range(30), attn_cls, color='orange')
    plt.xlabel('Frame index')
    plt.ylabel('Attention weight (CLS to frame)')
    plt.title('Transformer Attention Distribution (last layer)')
    plt.tight_layout()
    plt.savefig('attention_heatmap.png', dpi=300)
    plt.close()
    print("注意力热图已保存为 attention_heatmap.png")

    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_class = np.argmax(probs)
    label_map = {0:'forehand_drive',1:'forehand_lift',2:'forehand_net_shot',
                 3:'forehand_clear',4:'backhand_drive',5:'backhand_net_shot'}
    print(f"Predicted class: {label_map[pred_class]}, Confidence: {probs[pred_class]:.4f}")

if __name__ == '__main__':
    visualize_attention('demo_video.mp4')
