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

# 创建 DataLoader
train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 初始化模型
model = SkeletonTransformer(input_dim=132, d_model=128, nhead=4, num_layers=2,
                            dim_feedforward=256, num_classes=6, dropout=0.1, target_frames=30).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 训练循环
num_epochs = 30
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
    print(f"Epoch {epoch+1:2d} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

# 最终评估
print("\nFinal Test Accuracy:", test_acc)
print(classification_report(test_labels, test_preds))

# 混淆矩阵
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig(os.path.join(DATA_DIR, 'confusion_matrix.png'))
plt.show()

# 保存模型
torch.save(model.state_dict(), os.path.join(DATA_DIR, 'skeleton_transformer.pth'))
print(f"Model saved to {os.path.join(DATA_DIR, 'skeleton_transformer.pth')}")