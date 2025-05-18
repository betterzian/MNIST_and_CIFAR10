import os
import torch.distributed as dist


def setup(rank, world_size):
    # 环境变量设置
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    # 清理进程组
    dist.destroy_process_group()