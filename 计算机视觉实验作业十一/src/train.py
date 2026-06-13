import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import os
import json

DATA_DIR = 'outputs'
TARGET_FRAMES = 30
INPUT_DIM = 132
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 256
NUM_CLASSES = 6
DROPOUT = 0.1
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PoseSequenceDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        seq = self.X[idx]
        label = self.y[idx]
        if self.augment:
            seq = self.time_warp(seq)
            seq = self.add_keypoint_noise(seq)
        return seq, label
    def time_warp(self, seq):
        T = seq.shape[0]
        warp_factor = np.random.uniform(0.85, 1.15)
        new_len = int(T * warp_factor)
        if new_len < 2:
            return seq
        indices = np.linspace(0, T-1, new_len)
        indices = np.clip(indices, 0, T-1).astype(int)
        warped = seq[indices]
        x_old = np.linspace(0, new_len-1, new_len)
        x_new = np.linspace(0, new_len-1, T)
        resampled = np.zeros((T, seq.shape[1]))
        for d in range(seq.shape[1]):
            f = np.interp(x_new, x_old, warped[:, d])
            resampled[:, d] = f
        return torch.tensor(resampled, dtype=torch.float32)
    def add_keypoint_noise(self, seq, std=0.02):
        noise = torch.randn_like(seq) * std
        return seq + noise


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoderLayerWithAttn(nn.TransformerEncoderLayer):
    """自定义编码层，在 forward 中返回注意力权重"""
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
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, num_classes, dropout):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        # 使用自定义编码层（支持返回注意力）
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
        """
        Args:
            x: (B, T, input_dim)
            return_attention: 是否返回注意力权重
        Returns:
            logits: (B, num_classes)
            attentions: list of (B, nhead, T+1, T+1) 或 None
        """
        x = self.embedding(x)                     # (B, T, d_model)
        B, T, _ = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)     # (B, T+1, d_model)
        x = self.pos_encoder(x)

        attentions = []
        for layer in self.layers:
            if return_attention:
                x, attn = layer(x)   # 自定义层返回 (output, attn)
                attentions.append(attn)
            else:
                x = layer(x)[0]      # 丢弃注意力

        cls_out = x[:, 0, :]
        mean_out = x[:, 1:, :].mean(dim=1)
        combined = torch.cat([cls_out, mean_out], dim=1)
        logits = self.classifier(combined)

        if return_attention:
            return logits, attentions
        return logits


# ---------- 训练主函数 ----------
def train():
    os.makedirs(DATA_DIR, exist_ok=True)

    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
    
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train, random_state=42)
    
    train_dataset = PoseSequenceDataset(X_train, y_train, augment=True)
    val_dataset = PoseSequenceDataset(X_val, y_val, augment=False)
    test_dataset = PoseSequenceDataset(X_test, y_test, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SkeletonTransformer(INPUT_DIM, D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD, NUM_CLASSES, DROPOUT).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    best_val_acc = 0.0
    patience = 10
    trigger_times = 0

    model_save_path = os.path.join(DATA_DIR, 'best_model.pth')

    for epoch in range(EPOCHS):
        model.train()
        total_loss, total_correct = 0, 0
        for seqs, labels in train_loader:
            seqs, labels = seqs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(seqs)   # 训练时不返回注意力
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * seqs.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
        train_acc = total_correct / len(train_dataset)
        train_loss = total_loss / len(train_dataset)

        model.eval()
        val_correct = 0
        with torch.no_grad():
            for seqs, labels in val_loader:
                seqs, labels = seqs.to(DEVICE), labels.to(DEVICE)
                logits = model(seqs)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
        val_acc = val_correct / len(val_dataset)
        scheduler.step()
        print(f"Epoch {epoch+1:3d} | Train Loss {train_loss:.4f} | Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping!")
                break

    # 测试
    model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for seqs, labels in test_loader:
            seqs = seqs.to(DEVICE)
            logits = model(seqs)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    print("\nTest Classification Report:")
    print(classification_report(all_labels, all_preds, digits=4))
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    class_names = [f'Class {i}' for i in range(NUM_CLASSES)]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Confusion matrix image saved as 'confusion_matrix.png'")

if __name__ == '__main__':
    train()