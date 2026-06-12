import torch
import numpy as np
import cv2
import mediapipe as mp
import json
import os
from model import SkeletonTransformer

# ========== 必须添加这两个函数定义 ==========
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                    enable_segmentation=False, min_detection_confidence=0.5)

def extract_pose_sequence(video_path, target_frames=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
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
        return np.zeros((target_frames, 132))
    frames = np.array(frames)
    x_old = np.linspace(0, 1, len(frames))
    x_new = np.linspace(0, 1, target_frames)
    resampled = np.array([np.interp(x_new, x_old, frames[:, i]) for i in range(132)]).T
    return resampled

def normalize_pose_sequence(seq, target_frames=30):
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
# ==========================================

def load_model(model_path, target_frames=30, device='cpu'):
    model = SkeletonTransformer(input_dim=132, d_model=128, nhead=4, num_layers=2,
                                dim_feedforward=256, num_classes=6, dropout=0.1, target_frames=target_frames)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def infer_video(video_path, model, target_frames=30, device='cpu'):
    seq = extract_pose_sequence(video_path, target_frames)
    seq = normalize_pose_sequence(seq, target_frames)
    tensor = torch.from_numpy(seq).float().unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()
    return pred_class, confidence

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model('outputs/skeleton_transformer.pth', device=device)
    demo_video = 'demo_video.mp4'
    if not os.path.exists(demo_video):
        print(f"请放置测试视频{demo_video}，或修改 demo_video 变量路径")
    else:
        with open('outputs/label_map.json', 'r') as f:
            label_map = json.load(f)
        pred_idx, conf = infer_video(demo_video, model, device=device)
        pred_class = label_map[str(pred_idx)]
        print(f"Predicted class: {pred_class}")
        print(f"Confidence: {conf:.2f}")