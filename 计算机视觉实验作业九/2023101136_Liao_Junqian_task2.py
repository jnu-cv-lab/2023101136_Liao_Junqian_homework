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

# ====================== 修改保存路径 ======================
save_dir = "homework9/images_task2"
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
plt.savefig(f"{save_dir}/training_samples.png", dpi=300)
plt.show()

# 任务3：定义CNN模型
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

# ====================== 定义统一训练函数（用于对比优化器） ======================
def train_model(model, optimizer_name, epochs=10):
    # 重置模型参数
    model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # 定义三种优化器
    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=0.01)
    elif optimizer_name == "SGD+Momentum":
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    else:
        raise ValueError("优化器名称错误！")

    # 记录曲线
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    print(f"\n===== 开始训练：{optimizer_name} =====")
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / train_total
        train_acc = 100 * train_correct / train_total
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)

        # 验证阶段
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

        print(f'[{epoch+1}/{epochs}] {optimizer_name} | '
              f'训练损失: {avg_train_loss:.4f} | 训练准确率: {train_acc:.2f}% | '
              f'验证损失: {avg_val_loss:.4f} | 验证准确率: {val_acc:.2f}%')

    # 测试阶段
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    wrong_images, wrong_labels, wrong_preds = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            mask = predicted != labels
            wrong_images.append(images[mask].cpu())
            wrong_labels.append(labels[mask].cpu())
            wrong_preds.append(predicted[mask].cpu())

    avg_test_loss = test_loss / test_total
    test_acc = 100 * test_correct / test_total

    print(f"\n{optimizer_name} 测试结果：")
    print(f"测试损失: {avg_test_loss:.4f} | 测试准确率: {test_acc:.2f}%")

    # 拼接错误样本
    if wrong_images:
        wrong_images = torch.cat(wrong_images)
        wrong_labels = torch.cat(wrong_labels)
        wrong_preds = torch.cat(wrong_preds)
    else:
        wrong_images = torch.empty(0)
        wrong_labels = torch.empty(0, dtype=torch.long)
        wrong_preds = torch.empty(0, dtype=torch.long)

    # 返回所有指标和错误样本
    return {
        "name": optimizer_name,
        "train_losses": train_losses,
        "train_accs": train_accs,
        "val_losses": val_losses,
        "val_accs": val_accs,
        "test_acc": test_acc,
        "test_loss": avg_test_loss,
        "wrong_images": wrong_images,
        "wrong_labels": wrong_labels,
        "wrong_preds": wrong_preds
    }

# ====================== 训练三种优化器 ======================
epochs = 10
base_model = SimpleCNN()

# 依次训练
sgd_result = train_model(base_model, "SGD", epochs)
sgd_momentum_result = train_model(base_model, "SGD+Momentum", epochs)
adam_result = train_model(base_model, "Adam", epochs)

all_results = [sgd_result, sgd_momentum_result, adam_result]

# ====================== 输出最终对比表格 ======================
print("\n" + "="*80)
print("优化器最终对比结果")
print("="*80)
print(f"{'优化器':<15} {'训练损失':<10} {'验证损失':<10} {'训练准确率':<12} {'验证准确率':<12} {'测试准确率':<12}")
for res in all_results:
    print(f"{res['name']:<15} "
          f"{res['train_losses'][-1]:<10.4f} "
          f"{res['val_losses'][-1]:<10.4f} "
          f"{res['train_accs'][-1]:<11.2f}% "
          f"{res['val_accs'][-1]:<11.2f}% "
          f"{res['test_acc']:<11.2f}%")

# ====================== 绘制统一对比曲线 ======================
plt.figure(figsize=(15, 10))

# 损失曲线
plt.subplot(2, 2, 1)
for res in all_results:
    plt.plot(range(1, epochs+1), res['train_losses'], marker='o', label=f"{res['name']} Train")
plt.title('Training Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
for res in all_results:
    plt.plot(range(1, epochs+1), res['val_losses'], marker='s', label=f"{res['name']} Val")
plt.title('Validation Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 准确率曲线
plt.subplot(2, 2, 3)
for res in all_results:
    plt.plot(range(1, epochs+1), res['train_accs'], marker='o', label=f"{res['name']} Train")
plt.title('Training Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
for res in all_results:
    plt.plot(range(1, epochs+1), res['val_accs'], marker='s', label=f"{res['name']} Val")
plt.title('Validation Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(f"{save_dir}/optimizer_comparison.png", dpi=300)
plt.show()

# ====================== 绘制测试集预测样例（以Adam为例） ======================
model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 简单训练一轮用于展示
model.train()
for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    break

model.eval()
plt.figure(figsize=(12, 6))
dataiter = iter(test_loader)
images, labels = next(dataiter)
images = images.to(device)
with torch.no_grad():
    outputs = model(images)
_, predicted = torch.max(outputs, 1)

for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(images[i].cpu().squeeze(), cmap='gray')
    plt.title(f"True: {classes[labels[i]]}\nPred: {classes[predicted[i]]}")
    plt.axis('off')
plt.suptitle("Test Predictions (Adam)")
plt.savefig(f"{save_dir}/test_predictions.png", dpi=300)
plt.show()

# ====================== 错误分类样本展示（以Adam为例） ======================
adam_wrong_images = adam_result["wrong_images"]
adam_wrong_labels = adam_result["wrong_labels"]
adam_wrong_preds = adam_result["wrong_preds"]

print(f"\nAdam优化器测试集错误分类总数: {len(adam_wrong_labels)}")

if len(adam_wrong_labels) > 0:
    plt.figure(figsize=(12, 6))
    for i in range(min(10, len(adam_wrong_labels))):
        plt.subplot(2, 5, i+1)
        plt.imshow(adam_wrong_images[i].squeeze(), cmap='gray')
        plt.title(f"True: {classes[adam_wrong_labels[i]]}\nPred: {classes[adam_wrong_preds[i]]}")
        plt.axis('off')
    plt.suptitle("Misclassified Samples (Adam)")
    plt.savefig(f"{save_dir}/misclassified_samples.png", dpi=300)
    plt.show()

    # 易错数字统计
    wrong_label_counts = Counter(adam_wrong_labels.numpy())
    print("\n易错分类数字（真实标签）：")
    for num, cnt in wrong_label_counts.most_common(5):
        print(f"数字 {num} : {cnt} 次")
else:
    print("测试集无错误分类样本！")