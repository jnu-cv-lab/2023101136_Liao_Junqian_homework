# attention_analysis.py (最终修正版)
import cv2
import mediapipe as mp
import numpy as np
import torch
from train import SkeletonTransformer
import matplotlib.pyplot as plt
import os

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5
)

def extract_pose_sequence(video_path, target_frames=30):
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
        print(f"注意力张量形状: {attn_last.shape}")  # 调试信息

        # 处理不同维度的注意力
        if attn_last.dim() == 4:
            # (B, nhead, T+1, T+1)
            attn_matrix = attn_last[0].mean(dim=0)   # (T+1, T+1)
        elif attn_last.dim() == 3:
            # (B, T+1, T+1)
            attn_matrix = attn_last[0]               # (T+1, T+1)
        else:
            print(f"错误的注意力维度: {attn_last.dim()}，期望3或4")
            return

        attn_cls = attn_matrix[0, 1:].cpu().numpy()   # (30,)

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