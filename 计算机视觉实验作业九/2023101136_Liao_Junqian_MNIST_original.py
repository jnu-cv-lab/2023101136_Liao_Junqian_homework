# 环境准备
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("homework8", exist_ok=True)

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

# 训练集划分为训练集80% 、验证集20%
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
plt.savefig("homework8/training_samples.png", dpi=300) 
plt.show()

#  任务3：定义CNN模型 
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 卷积层组1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # 输入通道1，输出16
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)  # 池化层
        
        # 卷积层组2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # 全连接层
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # MNIST池化后尺寸7x7
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)  # 输出10分类

    def forward(self, x):
        # 前向传播
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7) 
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# 初始化模型并移至设备
model = SimpleCNN().to(device)
print("\nCNN模型结构:")
print(model)

#  任务4、5：训练、验证模型 
epochs = 10
criterion = nn.CrossEntropyLoss()  # 分类损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 优化器

# 训练曲线数据
train_losses = []
train_accs = []
val_losses = []
val_accs = []

print("\n开始训练...")
for epoch in range(epochs):
    #  训练阶段
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()  # 梯度清零
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        
        # 统计损失和准确率
        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    # 计算平均训练损失和准确率
    avg_train_loss = train_loss / train_total
    train_acc = 100 * train_correct / train_total
    train_losses.append(avg_train_loss)
    train_accs.append(train_acc)
    
    #  验证阶段 
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad(): 
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    avg_val_loss = val_loss / val_total
    val_acc = 100 * val_correct / val_total
    val_losses.append(avg_val_loss)
    val_accs.append(val_acc)
    
    # 每个epoch结果
    print(f'第 [{epoch+1}/{epochs}] 轮')
    print(f'训练损失: {avg_train_loss:.4f} | 训练准确率: {train_acc:.2f}%')
    print(f'验证损失: {avg_val_loss:.4f} | 验证准确率: {val_acc:.2f}%\n')

# 任务6：测试模型 
model.eval()
test_loss = 0.0
test_correct = 0
test_total = 0

# 收集错误分类样本
wrong_images = []
wrong_labels = []
wrong_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        test_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        
        # 错误分类的样本
        mask = predicted != labels
        wrong_images.append(images[mask].cpu())
        wrong_labels.append(labels[mask].cpu())
        wrong_preds.append(predicted[mask].cpu())

avg_test_loss = test_loss / test_total
test_acc = 100 * test_correct / test_total

print(f'测试集最终损失: {avg_test_loss:.4f}')
print(f'测试集最终准确率: {test_acc:.2f}%')

# 测试图像
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
plt.savefig("homework8/test_predictions.png", dpi=300) 
plt.show()

# 任务7：绘制训练曲线 
plt.figure(figsize=(12, 4))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), train_losses, label='Train Loss', marker='o')
plt.plot(range(1, epochs+1), val_losses, label='Val Loss', marker='s')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 准确率曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), train_accs, label='Train Accuracy', marker='o')
plt.plot(range(1, epochs+1), val_accs, label='Val Accuracy', marker='s')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("homework8/training_curves.png", dpi=300)  # 保存曲线
plt.show()

# 任务8：显示被错误分类的数字 
wrong_images = torch.cat(wrong_images)
wrong_labels = torch.cat(wrong_labels)
wrong_preds = torch.cat(wrong_preds)

print(f"测试集错误分类总数: {len(wrong_labels)}")

# 显示错误分类的图片
plt.figure(figsize=(12, 6))
for i in range(min(10, len(wrong_labels))):
    plt.subplot(2, 5, i+1)
    plt.imshow(wrong_images[i].squeeze(), cmap='gray')
    plt.title(f"True: {classes[wrong_labels[i]]}\nPred: {classes[wrong_preds[i]]}")
    plt.axis('off')
plt.suptitle("Misclassified Samples (True vs Pred)")
plt.savefig("homework8/misclassified_samples.png", dpi=300)
plt.show()

# 易错分类的数字
from collections import Counter
wrong_label_counts = Counter(wrong_labels.numpy())
print("\n易错分类数字（真实标签）：")
for num, cnt in wrong_label_counts.most_common(5):
    print(f"数字 {num} : 被错误分类 {cnt} 次")