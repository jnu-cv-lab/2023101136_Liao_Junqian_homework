# 环境准备
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ====================== 统一保存路径 ======================
save_dir = "homework9/images_task3-7"
os.makedirs(save_dir, exist_ok=True)

# 测试PyTorch导入
print("PyTorch 版本:", torch.__version__)

# 判断GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

# 测试张量操作
test_tensor = torch.tensor([1.0, 2.0, 3.0], device=device)
print("测试张量:", test_tensor)
print("张量运算 (平方):", test_tensor ** 2)

# 任务2：加载图像数据集
# 数据预处理：转为张量并标准化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载原始数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 训练集划分为训练集80%、验证集20%
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

# 数据加载器
batch_size = 512
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 数据集信息
print(f"训练集大小: {len(train_subset)}")
print(f"验证集大小: {len(val_subset)}")
print(f"测试集大小: {len(test_dataset)}")

# 训练样本图像
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
plt.figure(figsize=(12, 4))
dataiter = iter(train_loader)
images, labels = next(dataiter)

for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(images[i].squeeze(), cmap='gray')
    plt.title(f"true label: {classes[labels[i]]}")
    plt.axis('off')
plt.suptitle("Training Samples")
plt.savefig(f"{save_dir}/training_samples.png", dpi=300)
plt.show()

# 任务3：定义CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 卷积层组1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        # 卷积层组2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        # 全连接层
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# ====================== 【任务3】学习率对比实验 (Adam, lr=0.1,0.01,0.001) ======================
def train_with_lr(lr, epochs=10):
    model_new = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_new.parameters(), lr=lr)
    
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    print(f"\n===== 训练学习率: {lr} =====")
    for epoch in range(epochs):
        model_new.train()
        t_loss, t_correct, t_total = 0,0,0
        for imgs, labs in train_loader:
            imgs, labs = imgs.to(device), labs.to(device)
            optimizer.zero_grad()
            outs = model_new(imgs)
            loss = criterion(outs, labs)
            loss.backward()
            optimizer.step()
            
            t_loss += loss.item()*imgs.size(0)
            _, pred = torch.max(outs,1)
            t_total += labs.size(0)
            t_correct += (pred==labs).sum().item()
        
        avg_t_loss = t_loss/t_total
        avg_t_acc = 100*t_correct/t_total
        
        model_new.eval()
        v_loss, v_correct, v_total = 0,0,0
        with torch.no_grad():
            for imgs, labs in val_loader:
                imgs, labs = imgs.to(device), labs.to(device)
                outs = model_new(imgs)
                loss = criterion(outs, labs)
                v_loss += loss.item()*imgs.size(0)
                _, pred = torch.max(outs,1)
                v_total += labs.size(0)
                v_correct += (pred==labs).sum().item()
        
        avg_v_loss = v_loss/v_total
        avg_v_acc = 100*v_correct/v_total
        
        train_losses.append(avg_t_loss)
        train_accs.append(avg_t_acc)
        val_losses.append(avg_v_loss)
        val_accs.append(avg_v_acc)
        
        print(f"Epoch {epoch+1} | 训练损失:{avg_t_loss:.4f} | 训练准确率:{avg_t_acc:.2f}% | 验证损失:{avg_v_loss:.4f} | 验证准确率:{avg_v_acc:.2f}%")
    
    return model_new, train_losses, train_accs, val_losses, val_accs

# 运行学习率对比，统一10轮
lr_list = [0.1, 0.01, 0.001]
lr_results = []
best_model = None
best_idx = -1
max_acc = 0

for idx, lr in enumerate(lr_list):
    model_tmp, t_loss, t_acc, v_loss, v_acc = train_with_lr(lr, epochs=10)
    lr_results.append([t_loss, t_acc, v_loss, v_acc])
    if v_acc[-1] > max_acc:
        max_acc = v_acc[-1]
        best_acc = v_acc
        best_loss = v_loss
        best_model = model_tmp
        best_idx = idx

# 绘制学习率对比曲线
plt.figure(figsize=(14,10))
plt.subplot(2,2,1)
for i,lr in enumerate(lr_list):
    plt.plot(lr_results[i][0], label=f"lr={lr}")
plt.title("Train Loss vs LR")
plt.legend()
plt.grid()

plt.subplot(2,2,2)
for i,lr in enumerate(lr_list):
    plt.plot(lr_results[i][2], label=f"lr={lr}")
plt.title("Val Loss vs LR")
plt.legend()
plt.grid()

plt.subplot(2,2,3)
for i,lr in enumerate(lr_list):
    plt.plot(lr_results[i][1], label=f"lr={lr}")
plt.title("Train Acc vs LR")
plt.legend()
plt.grid()

plt.subplot(2,2,4)
for i,lr in enumerate(lr_list):
    plt.plot(lr_results[i][3], label=f"lr={lr}")
plt.title("Val Acc vs LR")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig(f"{save_dir}/lr_comparison.png", dpi=300)
plt.show()

# 选用最优模型开展后续测试与可视化
model = best_model

# 测试模型
model.eval()
test_loss = 0.0
test_correct = 0
test_total = 0
wrong_images, wrong_labels, wrong_preds = [], [], []
all_preds = []
all_labels = []
criterion = nn.CrossEntropyLoss()

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        mask = predicted != labels
        wrong_images.append(images[mask].cpu())
        wrong_labels.append(labels[mask].cpu())
        wrong_preds.append(predicted[mask].cpu())

avg_test_loss = test_loss / test_total
test_acc = 100 * test_correct / test_total
print(f'\n测试集损失: {avg_test_loss:.4f} | 测试集准确率: {test_acc:.2f}%')

wrong_images = torch.cat(wrong_images)
wrong_labels = torch.cat(wrong_labels)
wrong_preds = torch.cat(wrong_preds)

# 测试图像展示
plt.figure(figsize=(12, 6))
dataiter = iter(test_loader)
images, labels = next(dataiter)
images = images.to(device)
outputs = model(images)
_, predicted = torch.max(outputs, 1)

for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(images[i].cpu().squeeze(), cmap='gray')
    plt.title(f"True: {classes[labels[i]]}\nPred: {classes[predicted[i]]}")
    plt.axis('off')
plt.suptitle("Test Predictions")
plt.savefig(f"{save_dir}/test_predictions.png", dpi=300)
plt.show()

# 最优模型训练曲线
best_train_loss, best_train_acc, best_val_loss, best_val_acc = lr_results[best_idx]
plt.figure(figsize=(12, 4))
plt.subplot(1,2,1)
plt.plot(best_train_loss, label='Train Loss')
plt.plot(best_val_loss, label='Val Loss')
plt.legend()
plt.grid()
plt.subplot(1,2,2)
plt.plot(best_train_acc, label='Train Acc')
plt.plot(best_val_acc, label='Val Acc')
plt.legend()
plt.grid()
plt.savefig(f"{save_dir}/training_curves.png", dpi=300)
plt.show()

# ====================== 【任务4】第一层卷积核可视化 (至少8个) ======================
kernels = model.conv1.weight.data.cpu()
plt.figure(figsize=(12,6))
for i in range(min(16, 16)):
    plt.subplot(4,4,i+1)
    plt.imshow(kernels[i,0], cmap='gray')
    plt.title(f'Kernel {i+1}')
    plt.axis('off')
plt.suptitle("Conv1 Kernels (Trained)")
plt.savefig(f"{save_dir}/conv1_kernels.png", dpi=300)
plt.show()

# ====================== 【任务5】Feature Map 可视化 (第一层输出，至少8张) ======================
image_sample = next(iter(test_loader))[0][0:1].to(device)
with torch.no_grad():
    feat_map = model.conv1(image_sample)
feat_map = feat_map.squeeze(0).cpu()

plt.figure(figsize=(14,7))
for i in range(min(16,16)):
    plt.subplot(4,4,i+1)
    plt.imshow(feat_map[i], cmap='gray')
    plt.title(f'Feat Map {i+1}')
    plt.axis('off')
plt.suptitle("Conv1 Feature Maps")
plt.savefig(f"{save_dir}/feature_maps.png", dpi=300)
plt.show()

# ====================== 【任务6】错误分类样本展示 (≥8张) ======================
plt.figure(figsize=(14,8))
for i in range(min(12, len(wrong_labels))):
    plt.subplot(3,4,i+1)
    plt.imshow(wrong_images[i].squeeze(), cmap='gray')
    plt.title(f'True:{wrong_labels[i]}\nPred:{wrong_preds[i]}')
    plt.axis('off')
plt.suptitle("Misclassified Samples")
plt.savefig(f"{save_dir}/misclassified.png", dpi=300)
plt.show()

# 错误统计
cnt = Counter(wrong_labels.numpy())
print("\n最容易错分的数字：")
for k,v in cnt.most_common(5):
    print(f"数字 {k} 错分 {v} 次")

# ====================== 【任务7】混淆矩阵 ======================
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel("Pred")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig(f"{save_dir}/confusion_matrix.png", dpi=300)
plt.show()