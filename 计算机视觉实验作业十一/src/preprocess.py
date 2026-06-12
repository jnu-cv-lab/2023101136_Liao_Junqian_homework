import cv2
import mediapipe as mp
import numpy as np
import os
from sklearn.model_selection import train_test_split
import json

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

def process_dataset(data_root, target_frames=30, test_size=0.2, random_state=42, output_dir='outputs'):
    print(f"=== process_dataset called with data_root={data_root} ===")
    class_names = ['forehand_drive', 'forehand_lift', 'forehand_net_shot',
                   'forehand_clear', 'backhand_drive', 'backhand_net_shot']
    label_to_idx = {name: i for i, name in enumerate(class_names)}
    
    X = []
    y = []
    for class_name in class_names:
        class_path = os.path.join(data_root, class_name)
        print(f"Checking {class_path} ...")
        if not os.path.isdir(class_path):
            print(f"  Warning: {class_path} not found, skipping.")
            continue
        video_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.mp4','.avi','.mov','.mkv'))]
        print(f"  Found {len(video_files)} videos in {class_name}")
        for video_file in video_files:
            video_path = os.path.join(class_path, video_file)
            try:
                seq = extract_pose_sequence(video_path, target_frames)
                seq = normalize_pose_sequence(seq, target_frames)
                X.append(seq)
                y.append(label_to_idx[class_name])
                print(f"    Processed {video_file}")
            except Exception as e:
                print(f"    Error processing {video_file}: {e}")
    
    print(f"Total videos processed: {len(X)}")
    if len(X) == 0:
        raise ValueError("No videos found. Check your data_root path and class_names.")
    # ... 后续 train_test_split 和保存
    
if __name__ == '__main__':
    DATA_ROOT = 'data/archive'      # 相对于项目根目录
    OUTPUT_DIR = 'outputs'
    process_dataset(DATA_ROOT, target_frames=30, output_dir=OUTPUT_DIR)