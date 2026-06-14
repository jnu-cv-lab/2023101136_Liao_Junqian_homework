import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
from model import SkeletonTransformer

# 设置数据目录
DATA_DIR = 'outputs'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 加载预处理好的数据
X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))

# 显式转换为 float32 和 long
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).long()
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).long()

# 创建 DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 初始化模型
model = SkeletonTransformer(input_dim=132, d_model=128, nhead=4, num_layers=2,
                            dim_feedforward=256, num_classes=6, dropout=0.1, target_frames=30).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 用于记录训练曲线
train_losses = []
train_accs = []
test_accs = []

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())
    
    train_acc = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(train_loader)
    
    # 测试
    model.eval()
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            preds = torch.argmax(logits, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(y_batch.numpy())
    test_acc = accuracy_score(test_labels, test_preds)
    
    # 记录
    train_losses.append(avg_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    
    print(f"Epoch {epoch+1:2d} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

# 最终评估
print("\nFinal Test Accuracy:", test_acc)
print(classification_report(test_labels, test_preds))

# 混淆矩阵（保存到主文件夹）
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png')   # 修改：直接保存到当前目录（主文件夹）
plt.show()

# 绘制训练曲线（保存到主文件夹）
epochs = range(1, num_epochs + 1)

plt.figure(figsize=(12, 5))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'b-', label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)

# 准确率曲线
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accs, 'r-', label='Training Accuracy')
plt.plot(epochs, test_accs, 'g-', label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy Curves')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png')   # 修改：直接保存到当前目录（主文件夹）
plt.show()

torch.save(model.state_dict(), os.path.join(DATA_DIR, 'skeleton_transformer.pth'))
print(f"Model saved to {os.path.join(DATA_DIR, 'skeleton_transformer.pth')}")
print(f"Confusion matrix saved to confusion_matrix.png")
print(f"Training curves saved to training_curves.png")