import torch
import torch.distributed as dist
import torch.nn as nn

def setup(rank, world_size=1,backend='gloo',init_method='tcp://localhost:29500'):
    # 环境变量设置
    # 初始化分布式环境（Gloo后端）
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size
    )


def cleanup():
    # 清理进程组
    dist.destroy_process_group()


def test(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for X, Y in test_loader:
            pred = torch.argmax(model(X), dim=1)
            correct += (pred == Y).sum().item()
            total += Y.size(0)
    print(f"Test Accuracy: {correct/total:.4f}")
    return correct/total


def uniform_init(model:nn.Module):
    if hasattr(model, 'weight'):
        nn.init.uniform_(model.weight, -0.1, 0.1)
    if hasattr(model, 'bias') and model.bias is not None:
        nn.init.constant_(model.bias, 0)