import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from setup import setup, cleanup
from dataset import CIFAR10Dataset
from models import ResNet18


def train(rank, world_size):
    setup(rank, world_size)
    # 数据加载
    dataset = CIFAR10Dataset()
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)

    # 模型定义
    model = ResNet18().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001, momentum=0.9)

    # 训练循环
    for epoch in range(10):
        sampler.set_epoch(epoch)
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(rank), labels.to(rank)
            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if rank == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    cleanup()

if __name__ == "__main__":
    world_size = 1
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)