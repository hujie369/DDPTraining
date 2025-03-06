from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader


def get_dataloaders(path, rank=0, world_size=0, batch_size=128, num_workers=16, mode='train'):
    if mode == 'train':
        # 训练集数据预处理
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225])
        ])

        # 加载训练集
        train_dataset = datasets.ImageNet(
            root=path, split='train', transform=train_transform)
        # 创建训练集的数据加载器
        if rank != -1:
            # 创建训练集的分布式采样器
            train_sampler = DistributedSampler(
                train_dataset, num_replicas=world_size, rank=rank) if rank != -1 else None
            train_dataloader = DataLoader(train_dataset,
                                          batch_size=batch_size,
                                          sampler=train_sampler,
                                          num_workers=num_workers,
                                          pin_memory=True)
        else:
            train_dataloader = DataLoader(train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=num_workers,
                                          pin_memory=True)
        # train 数据集在所有进程中创建, 因此要使用DistributedSampler来对数据集进行分割
        return train_dataloader
    elif mode == 'val':
        # 验证集数据预处理
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225])
        ])
        # val 数据集只在主进程中创建一次
        val_dataset = datasets.ImageNet(
            root=path, split='val', transform=val_transform)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                    num_workers=num_workers, pin_memory=True)
        return val_dataloader
    else:
        raise ValueError("wrong mode for dataloader")
