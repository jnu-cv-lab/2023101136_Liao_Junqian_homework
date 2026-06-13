# inference.py
import cv2
import mediapipe as mp
import numpy as np
import torch
import os
import sys

# 导入 train.py 中的模型定义（确保 train.py 在相同目录或可导入）
# 如果 train.py 在 src/ 目录下，可以直接导入
from train import SkeletonTransformer, PositionalEncoding

# 复用 preprocess.py 中的 MediaPipe 初始化
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5
)

def extract_pose_sequence(video_path, target_frames=30):
    """从视频中提取骨架序列，返回 (T, 132) 或 None"""
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
            # 未检测到骨架，用零填充（保持与预处理一致）
            frames.append([0.0] * 132)
    cap.release()
    if len(frames) == 0:
        return None
    frames = np.array(frames, dtype=np.float32)

    # 重采样到固定帧数
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

    # 归一化（与 preprocess.py 中的 normalize_pose_sequence 相同）
    frames = normalize_pose_sequence(frames)
    return frames

def normalize_pose_sequence(seq):
    """对骨架序列进行归一化（与训练预处理完全一致）"""
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

def predict(video_path, model_path='outputs/best_model.pth'):
    """单视频推理"""
    # 模型参数（必须与训练时完全一致）
    INPUT_DIM = 132
    D_MODEL = 128
    NHEAD = 4
    NUM_LAYERS = 2
    DIM_FEEDFORWARD = 256
    NUM_CLASSES = 6
    DROPOUT = 0.1
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    if not os.path.exists(model_path):
        print(f"模型文件未找到: {model_path}")
        return
    model = SkeletonTransformer(INPUT_DIM, D_MODEL, NHEAD, NUM_LAYERS,
                                DIM_FEEDFORWARD, NUM_CLASSES, DROPOUT).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # 提取骨架序列
    seq = extract_pose_sequence(video_path, target_frames=30)
    if seq is None:
        print("无法从视频中提取骨架序列")
        return

    # 转换为张量并推理
    seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(seq_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_class = np.argmax(probs)

    # 标签映射（与训练时的顺序一致）
    label_map = {
        0: 'forehand_drive',
        1: 'forehand_lift',
        2: 'forehand_net_shot',
        3: 'forehand_clear',
        4: 'backhand_drive',
        5: 'backhand_net_shot'
    }
    print(f"Predicted class: {label_map[pred_class]}")
    print(f"Confidence: {probs[pred_class]:.4f}")

if __name__ == '__main__':
    # 请确保 demo_video.mp4 存在于项目根目录或提供完整路径
    predict('demo_video.mp4')