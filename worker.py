# worker.py (Rank 1 and 2)
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
from setup import setup


def compute_gradients(model, X, Y,device):
    model.train().to(device)
    X, Y = X.to(device), Y.to(device)
    outputs = model(X)
    loss = nn.CrossEntropyLoss()(outputs, Y)
    loss.backward()  # 计算梯度
    gradients = {name: param.grad.detach().cpu() for name, param in model.named_parameters()}
    return gradients

def run_worker(rank,model:nn.Module,dataset,world_size=2,backend='gloo',init_method='tcp://localhost:29500'):
    setup(rank, world_size=world_size, backend=backend, init_method=init_method)
    torch.manual_seed(0)  # 固定随机种子
    device = torch.device(f'cuda:{rank-1}' if torch.cuda.is_available() else 'cpu')
    # 加载训练集
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size-1, rank=rank-1)
    train_loader = DataLoader(dataset, batch_size=512, sampler=sampler)

    epoch = 0
    while True:
        sampler.set_epoch(epoch)
        epoch = epoch + 1
        for X, Y in train_loader:
            new_state = [None]
            dist.broadcast_object_list(new_state, src=0)
            if new_state[0] == "finish":
                return
            else:
                model.load_state_dict(new_state[0])

            # 计算本地梯度
            gradients = compute_gradients(model, X, Y,device)

            # 发送梯度给Server（确保张量在CPU）
            dist.send_object_list([gradients], dst=0)



if __name__ == "__main__":
    rank = 1
    run_worker(rank)