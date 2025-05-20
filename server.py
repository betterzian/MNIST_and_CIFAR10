# server.py (Rank 0)
import torch
import torch.distributed as dist
from torch.nn import Module
from torch.utils.data import DataLoader
from torchvision.models.quantization import resnet18
from dataset import MNISTDataset
from setup import uniform_init, test, setup


def run_server(rank,model:Module,dataset,world_size=2,backend='gloo',init_method='tcp://localhost:29500'):

    setup(rank,world_size=world_size,backend=backend,init_method=init_method)

    # 初始化模型参数
    model.apply(uniform_init)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

    # 加载测试集
    test_loader = DataLoader(dataset, batch_size=1000)

    num_epochs = 10
    batches_per_epoch = 10  # 假设每个epoch处理100个批次

    for epoch in range(num_epochs):
        # 评估模型
        test(model, test_loader)
        for _ in range(batches_per_epoch):
            # 广播参数给所有worker
            dist.broadcast_object_list([model.state_dict()], src=0)

            # 接收workers的梯度
            rec_grads = []
            for worker_rank in range(1,world_size):
                grads = [None]
                dist.recv_object_list(grads,src=worker_rank)
                rec_grads.extend(grads)

            # 平均梯度并更新参数
            avg_gradients = {}
            for param_name in rec_grads[0].keys():
                avg_grad = torch.mean(torch.stack([grads[param_name] for grads in rec_grads]), dim=0)
                avg_gradients[param_name] = avg_grad
            optimizer.zero_grad()
            for name, param in model.named_parameters():
                param.grad = avg_gradients[name]
            optimizer.step()

    dist.broadcast_object_list(["finish"], src=0)
    return

if __name__ == "__main__":
    run_server(model=resnet18(),dataset=MNISTDataset)