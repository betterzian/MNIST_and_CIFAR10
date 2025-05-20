from argparse import Namespace
from dataset import MNISTDataset
from models import LogisticRegression
from server import run_server
from setup import cleanup
from worker import run_worker
import torch.multiprocessing as mp

def run(rank,args):
    model = LogisticRegression()
    if rank == 0:
        run_server(rank=rank, model=model, dataset=args.dataset,world_size=args.world_size)
    else:
        run_worker(rank=rank, model=model, dataset=args.dataset,world_size=args.world_size)
    cleanup()

if __name__ == "__main__":
    import sys
    #world_size = int(sys.argv[1])  # 通过命令行参数指定workers数量
    world_size = 2
    train_dataset = MNISTDataset(train=True)
    test_dataset = MNISTDataset(train=False)
    server_args = Namespace( dataset=test_dataset,world_size=world_size)
    worker_args = Namespace( dataset=train_dataset,world_size=world_size)
    mp.spawn(run,args=(server_args,),nprocs=world_size,join=True)
