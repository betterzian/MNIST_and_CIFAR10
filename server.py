# server.py (Rank 0)
import torch
import torch.distributed as dist
from torch.nn import Module
from torch.utils.data import DataLoader
from torchvision.models.quantization import resnet18
from dataset import MNISTDataset
from setup import uniform_init, test, setup
from torch.utils.tensorboard import SummaryWriter

def run_server(rank,model:Module,dataset,path,num_epochs=1000,world_size=2,backend='gloo',init_method='tcp://localhost:29500'):

    setup(rank,world_size=world_size,backend=backend,init_method=init_method)
    device = (torch.device('cuda:0') if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # 初始化模型参数
    #model.apply(uniform_init)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005,momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400)
    writer = SummaryWriter('logs/'+path)
    # 加载测试集
    test_loader = DataLoader(dataset, batch_size=1000)

    num_epochs = num_epochs

    for epoch in range(num_epochs):
        # 评估模型
        accuracy = test(model, test_loader, writer, epoch,device)
        if accuracy > 0.92:
            dist.broadcast_object_list(["finish"], src=0)
            return
        # 广播参数给所有worker
        model.to('cpu')
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
            if param.grad is None:
                param.grad = avg_gradients[name]  # 确保梯度在正确设备
            else:
                param.grad += avg_gradients[name]
        model.to(device)
        optimizer.step()
        scheduler.step()

    dist.broadcast_object_list(["finish"], src=0)
    writer.close()
    return

if __name__ == "__main__":
    run_server(model=resnet18(),dataset=MNISTDataset)