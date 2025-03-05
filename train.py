import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.amp import GradScaler, autocast


# 定义训练参数
num_epochs = 10
batch_size = 256
learning_rate = 0.001
num_workers = 16
data_address = os.getenv('DATASET_ADDRESS')
if data_address:
    # 使用数据集地址进行操作
    print(f"Using dataset address: {data_address}")
else:
    print("Dataset address not set.")


# 数据预处理
train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ]
)

# 加载数据集
train_dataset = datasets.ImageNet(
    root=data_address, split="train", transform=train_transform
)
val_dataset = datasets.ImageNet(
    root=data_address, split="val", transform=val_transform
)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
)

# 检测可用的GPU数量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
if num_gpus > 1:
    print(f"Using {num_gpus} GPUs for training.")

# 加载ResNet-18模型
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1000)  # ImageNet有1000个类别

if num_gpus > 1:
    model = nn.DataParallel(model)

model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 初始化GradScaler用于AMP训练
scaler = GradScaler()


# 定义top1和top5准确率计算函数
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# 训练和验证循环
best_top1 = 0
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_files = []

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    train_loss = 0
    train_top1 = 0
    train_top5 = 0
    num_batches = len(train_loader)

    with tqdm(
        train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", unit="batch"
    ) as pbar:
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            with autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            top1, top5 = accuracy(outputs, labels, topk=(1, 5))
            train_top1 += top1.item()
            train_top5 += top5.item()

    train_loss /= num_batches
    train_top1 /= num_batches
    train_top5 /= num_batches

    print(
        f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, "
        f"Train Top1: {train_top1:.2f}%, Train Top5: {train_top5:.2f}%"
    )

    # 验证阶段
    model.eval()
    val_loss = 0
    val_top1 = 0
    val_top5 = 0
    num_batches = len(val_loader)

    with tqdm(
        val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]", unit="batch"
    ) as pbar:
        with torch.no_grad():
            for images, labels in pbar:
                images = images.to(device)
                labels = labels.to(device)

                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                top1, top5 = accuracy(outputs, labels, topk=(1, 5))
                val_top1 += top1.item()
                val_top5 += top5.item()

    val_loss /= num_batches
    val_top1 /= num_batches
    val_top5 /= num_batches

    print(
        f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss:.4f}, "
        f"Val Top1: {val_top1:.2f}%, Val Top5: {val_top5:.2f}%"
    )

    # 保存检查点
    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_top1": val_top1,
    }
    checkpoint_path = os.path.join(
        checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
    torch.save(checkpoint, checkpoint_path)
    checkpoint_files.append(checkpoint_path)

    if len(checkpoint_files) > 3:
        oldest_checkpoint = checkpoint_files.pop(0)
        os.remove(oldest_checkpoint)

    # 保存全局最优模型
    if val_top1 > best_top1:
        best_top1 = val_top1
        best_checkpoint_path = os.path.join(
            checkpoint_dir, "best_checkpoint.pth")
        torch.save(checkpoint, best_checkpoint_path)
        print(f"Saved best model with Top1 accuracy: {best_top1:.2f}%")
